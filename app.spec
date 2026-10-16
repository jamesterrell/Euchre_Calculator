# PyInstaller spec for the desktop app. Build with:
#
#     pip install pyinstaller pywebview
#     python build_cards.py cards.dll      # if static/cards.png is missing
#     pyinstaller app.spec                 # -> dist/EuchreCalculator.exe
#
# The repo's own modules ride along twice: compiled into the bundle, so the
# analysis finds numpy, numba and the stdlib they need, and as plain source
# under engine/, which is what app.py actually imports from -- see its
# docstring for why numba needs the files on disk at a stable path.
import glob
import hashlib
import os

ROOT = os.path.abspath(SPECPATH)
SKIP = {"app.py", "build_cards.py"}

if not os.path.exists(os.path.join(ROOT, "static", "cards.png")):
    raise SystemExit("static/cards.png is missing -- run "
                     "`python build_cards.py cards.dll` first")

sources = sorted(p for p in glob.glob(os.path.join(ROOT, "*.py"))
                 if os.path.basename(p) not in SKIP)
statics = sorted(glob.glob(os.path.join(ROOT, "static", "*")))

digest = hashlib.sha1()
for path in sources + statics:
    digest.update(os.path.relpath(path, ROOT).encode())
    with open(path, "rb") as handle:
        digest.update(handle.read())
build_file = os.path.join(workpath, "BUILD")
os.makedirs(workpath, exist_ok=True)
with open(build_file, "w") as handle:
    handle.write(digest.hexdigest()[:12])

datas = ([(p, "engine") for p in sources]
         + [(p, "engine/static") for p in statics]
         + [(build_file, "engine")])

a = Analysis(
    [os.path.join(ROOT, "app.py")],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=["server"],
    excludes=["jupyter", "notebook", "IPython", "matplotlib", "tkinter"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="EuchreCalculator",
    console=False,
    upx=False,
)
