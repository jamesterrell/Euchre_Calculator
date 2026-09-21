"""Build `static/cards.png` -- the 24 Euchre cards -- out of XP's `cards.dll`.

The page draws its cards as one sprite sheet rather than as CSS, so the faces
are the real ones. They come out of `cards.dll`, the card library Windows
shipped from 3.0 through XP, whose 52 faces are `RT_BITMAP` resources at
71x96. Resources are read with `LOAD_LIBRARY_AS_DATAFILE`, which maps the
file without running a line of it and works across bitness -- 64-bit Python
reads a 32-bit DLL's resources happily.

    python build_cards.py cards.dll

Two things about those bitmaps are older than Win32 and have to be handled.
They carry a 12-byte `BITMAPCOREHEADER` -- 16-bit dimensions, 3-byte palette
entries -- rather than the 40-byte `BITMAPINFOHEADER` everything since uses.
And an `RT_BITMAP` resource is a DIB with no `BITMAPFILEHEADER`, so one is
prepended to make the bytes loadable.

`cards.dll` is Microsoft's and is not redistributable, which is the reason
this is a build step and not a checked-in asset: the sheet it writes is
ignored by git, and anyone who wants the real cards supplies their own copy
of the DLL. Without it the page still runs -- `.pc` falls back to a drawn
card, see `static/index.html`.
"""
import ctypes
import io
import os
import struct
import sys

LOAD_LIBRARY_AS_DATAFILE = 0x00000002
RT_BITMAP = 2

CARD_W, CARD_H = 71, 96

# Resources 1-52 are the faces, thirteen to a suit in this order, each suit
# running A, 2..10, J, Q, K. Verified by eye against a contact sheet.
DLL_SUITS = "CDHS"
DLL_RANKS = "A23456789TJQK"

# The sheet is laid out the way the page's own deck is: a row per suit in
# `SUITS` order, a column per rank in `RANKS` order.
SHEET_SUITS = "SHDC"
SHEET_RANKS = "9TJQKA"

ENUMRESNAMEPROC = ctypes.WINFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p)


def _kernel32():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.LoadLibraryExW.restype = ctypes.c_void_p
    k32.FindResourceW.restype = ctypes.c_void_p
    k32.FindResourceW.argtypes = [ctypes.c_void_p] * 3
    k32.LoadResource.restype = ctypes.c_void_p
    k32.LoadResource.argtypes = [ctypes.c_void_p] * 2
    k32.LockResource.restype = ctypes.c_void_p
    k32.LockResource.argtypes = [ctypes.c_void_p]
    k32.SizeofResource.restype = ctypes.c_uint32
    k32.SizeofResource.argtypes = [ctypes.c_void_p] * 2
    return k32


def read_bitmap(path, ident):
    """The raw DIB bytes of one RT_BITMAP resource, by numeric id."""
    k32 = _kernel32()
    handle = k32.LoadLibraryExW(path, None, LOAD_LIBRARY_AS_DATAFILE)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    res = k32.FindResourceW(ctypes.c_void_p(handle), ctypes.c_void_p(ident),
                            ctypes.c_void_p(RT_BITMAP))
    if not res:
        raise LookupError("no bitmap resource %d in %s" % (ident, path))
    data = k32.LoadResource(ctypes.c_void_p(handle), ctypes.c_void_p(res))
    size = k32.SizeofResource(ctypes.c_void_p(handle), ctypes.c_void_p(res))
    return ctypes.string_at(k32.LockResource(ctypes.c_void_p(data)), size)


def to_bmp(dib):
    """Prepend the BITMAPFILEHEADER an RT_BITMAP resource leaves off."""
    header_size, = struct.unpack_from("<I", dib, 0)
    if header_size == 12:
        bpp, = struct.unpack_from("<H", dib, 10)
        palette = ((1 << bpp) if bpp <= 8 else 0) * 3
    else:
        bpp, = struct.unpack_from("<H", dib, 14)
        clr_used, = struct.unpack_from("<I", dib, 32)
        palette = ((clr_used or (1 << bpp)) if bpp <= 8 else clr_used) * 4
    offset = 14 + header_size + palette
    return struct.pack("<2sIHHI", b"BM", 14 + len(dib), 0, 0, offset) + dib


def cut_corners(card):
    """Clear the white pixels outside the card's rounded outline.

    Solitaire draws the corners as table, not as card, so they have to be
    transparent or every card sits in a little white notch. The fill starts
    at each corner and spreads over exactly-white pixels only; the outline is
    a closed loop in the suit's colour -- red for the red suits, which is why
    this cannot just clear a fixed corner block -- so nothing leaks inside.
    """
    card = card.convert("RGBA")
    px = card.load()
    w, h = card.size
    for start in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        stack = [start]
        while stack:
            x, y = stack.pop()
            if not (0 <= x < w and 0 <= y < h):
                continue
            r, g, b, a = px[x, y]
            if a == 0 or (r, g, b) != (255, 255, 255):
                continue
            px[x, y] = (255, 255, 255, 0)
            stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return card


def build(dll, out):
    from PIL import Image

    sheet = Image.new("RGBA", (CARD_W * len(SHEET_RANKS),
                               CARD_H * len(SHEET_SUITS)), (0, 0, 0, 0))
    for row, suit in enumerate(SHEET_SUITS):
        for col, rank in enumerate(SHEET_RANKS):
            ident = DLL_SUITS.index(suit) * 13 + DLL_RANKS.index(rank) + 1
            card = Image.open(io.BytesIO(to_bmp(read_bitmap(dll, ident))))
            if card.size != (CARD_W, CARD_H):
                raise ValueError("resource %d is %dx%d, not %dx%d"
                                 % ((ident,) + card.size + (CARD_W, CARD_H)))
            sheet.paste(cut_corners(card), (col * CARD_W, row * CARD_H))
    sheet.save(out)
    return sheet.size


def main():
    dll = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "cards.dll")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "static", "cards.png")
    if not os.path.isfile(dll):
        sys.exit("no such file: %s\n"
                 "Supply a copy of XP's cards.dll; see this module's "
                 "docstring." % dll)
    size = build(dll, out)
    print("wrote %s, %dx%d, %d bytes" % (out, size[0], size[1],
                                         os.path.getsize(out)))


if __name__ == "__main__":
    main()
