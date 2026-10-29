---
name: euchre-server
description: How the local web server and its page work -- server.py, static/index.html, build_cards.py and the XP cards.dll sprite sheet. Read before editing the server, the HTTP endpoints, the page, or the card artwork; covers the four deliberate design choices (localhost bind, warm-up gate, serialised queries, closing the connection on every refusal) and the Engine/table-sizing traps.
---

# The local server

```bash
python server.py --workers 10          # http://127.0.0.1:8000
python server.py --port 9000 --deals 2000
```

Standard library only -- `ThreadingHTTPServer`, no Flask, no bundler, and
`static/index.html` is one file of vanilla HTML, CSS and JS. The repo's
dependencies are still numpy, numba and jupyter.

The page is styled as Windows XP Solitaire, and the cards are XP's own. They
come from `cards.dll` -- the card library Windows shipped from 3.0 to XP --
whose 52 faces are `RT_BITMAP` resources at 71x96. `build_cards.py` reads the
24 a Euchre deck uses and writes `static/cards.png`, one sprite sheet, a row
per suit and a column per rank in the page's own `SUITS` / `RANKS` order:

```bash
python build_cards.py cards.dll     # -> static/cards.png, 426x384
```

Three things about it are worth knowing.

- **Both files are gitignored, and the page runs without them.** `cards.dll`
  is Microsoft's and not ours to redistribute, so it is a build step rather
  than a checked-in asset. `.pc` draws its own card -- rank over suit in two
  corners, a pip in the middle -- and the page asks for the sheet once on
  load, adding `real-cards` to the body only if it answers. Everything below
  that swap is identical, since the drawn card is the same 71x96 box.
- **Those bitmaps are older than Win32.** They carry a 12-byte
  `BITMAPCOREHEADER` -- 16-bit dimensions, 3-byte palette entries -- not the
  40-byte `BITMAPINFOHEADER`, and an `RT_BITMAP` resource has no
  `BITMAPFILEHEADER` at all. `build_cards.py` handles both.
- **The corners are cut to transparent.** Solitaire draws a card's rounded
  corners as table, so they are flood-filled from each corner over white
  pixels only. The outline is a closed loop in the card's own colour, which is
  why a fixed corner block would not do. XP draws that loop red on the red
  *pip* cards and black on the faces, so `blacken_outline` recolours it to
  black afterwards -- otherwise a hand shows red borders on 9H TH AH 9D TD AD
  and black ones on the other eighteen. Recolouring the sheet rather than
  adding a CSS border keeps every card exactly 71x96.

    GET  /                  the page
    GET  /api/health        {"ready", "busy", "workers", ...}
    POST /api/evaluate      {"hand", "up_card", "seat", "dealer", "deals",
                             "round": 1|2, "min_reached", "actions", ...}
    POST /api/solve         {"hands": [4], "up_card", "dealer", "seat"}

Both replies are `as_dict()` straight off the `api` result, so the server does
no arithmetic and the page does none either.

Four things about it are deliberate:

- **It binds to localhost.** No authentication, no rate limiting, no real size
  cap. `--host` exists so exposing it is a decision rather than an accident.
- **It warms up before it serves.** `api.Engine.warm()` runs in a background
  thread at boot; until it returns, `/api/health` says `ready: false` and the
  query endpoints answer 503 rather than hanging. That matters because a cold
  numba cache is ~80 s, not the ~25 s a warm one takes, and a user staring at
  a frozen page cannot tell the difference from a crash.
- **It serialises queries.** `Engine` holds a lock: one query already uses
  every thread it is given, and two at once oversubscribe numba's pool and
  make both slower. `ThreadingHTTPServer` is still right, because it lets
  `/api/health` answer *while* a query runs -- which is what lets the page say
  "busy" instead of appearing dead.
- **Every refusal closes the connection.** This is HTTP/1.1, so a connection
  is reused by default, and a request whose body was never read leaves bytes
  in the socket that the next request gets parsed out of. Rejecting without
  draining is the normal case here (a body too large to want, a query before
  the engine is warm), so `_fail` sends `Connection: close`. This was a real
  bug, found by `tests/test_server.py` and not by hand-testing with curl.

`api.Engine` is what a long-lived process should hold: one transposition table
for the life of the process instead of one allocated per call, and since the
table is never cleared the queries warm each other. It takes no deal count --
see the sizing note above, which is where `Engine(deals=...)` went. **The first query of a
process costs two to three times the ones after it** -- 10.4 s against 4.7 s
for three actions over 1,000 deals -- which is why `api` carries two rates,
`SECONDS_PER_DEAL` and `SECONDS_PER_DEAL_FIRST`, and `Engine.status` reports
how many queries have run so the page can quote the right one.

That same warming is a trap when measuring: runs get faster through a session,
and a careless A/B reads "no progress" as slower than "with progress". Warm a
fresh table before timing anything, and compare like with like.

`Engine.status` also carries live progress -- the fraction done and which
action is being priced -- which works without job ids precisely because
queries are serialised: there is only ever one to report on. Reporting it
means the sweep runs in blocks with a barrier at each one, so
`hand_ev.DEALS_PER_CHUNK_PER_BLOCK` sets how much work a chunk must have
inside a block to be worth the barrier. Without that floor a small sweep
collapses: 1,000 deals took 13.5 s against 5.6 s, because 20 blocks over 40
chunks is barely one deal per chunk before every thread stops to wait.

`tests/test_server.py` runs mostly against a stub engine, so the suite does not
pay for numba to load; one class at the end uses the real one at eight deals to
check the wiring. What it tests is what goes wrong between a browser and a
function call -- that `api`'s own error messages reach the caller verbatim
(the page prints them), that a query before warm-up is a 503 rather than a
hang, that `/static/..` cannot walk out of `static/`, and that a refusal does
not poison the next request on the same connection.
