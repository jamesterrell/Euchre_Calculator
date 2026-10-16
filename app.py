"""
The calculator as a desktop app: `server.py` on a private port, shown in a
window of its own rather than a browser.

    python app.py                 # from the repo, needs pywebview
    pyinstaller app.spec          # -> dist/EuchreCalculator.exe

The window is pywebview over Edge WebView2, which ships with Windows 10 and
11, so the page is `static/index.html` unchanged and the server is the one
`server.py` already is. Closing the window ends the process.

## Why the engine is copied out of the bundle

A one-file exe unpacks itself into a fresh temporary folder on every launch.
numba keys its on-disk cache on the source file's path and timestamp, so from
there every launch would be a cold cache -- eighty seconds of compiling, every
time. So the frozen app copies its own sources once, into a per-user folder
named for a hash of them, and imports from there. The first launch on a
machine compiles; every one after loads the cache in a second or two. A new
build has a new hash and so a fresh folder, which is the same invalidation
`fastsim._drop_stale_cache` exists for.
"""
import multiprocessing
import os
import shutil
import sys
import threading

TITLE = "Euchre Calculator"
HOME = os.path.join(os.environ.get("LOCALAPPDATA")
                    or os.path.expanduser("~"), "EuchreCalculator")


def _install_engine():
    """Copy the bundled sources somewhere stable and put them on sys.path."""
    bundled = os.path.join(sys._MEIPASS, "engine")
    with open(os.path.join(bundled, "BUILD")) as handle:
        build = handle.read().strip()
    target = os.path.join(HOME, "engine-" + build)
    if not os.path.isdir(target):
        staging = target + ".partial"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(bundled, staging)
        os.replace(staging, target)
    sys.path.insert(0, target)
    return target


def _log_to_file():
    """A windowed exe has no console, and `print(file=sys.stderr)` on None
    raises. Send both streams to a log the user can find."""
    os.makedirs(HOME, exist_ok=True)
    log = open(os.path.join(HOME, "app.log"), "w", buffering=1,
               encoding="utf-8")
    sys.stdout = sys.stderr = log


def main():
    frozen = getattr(sys, "frozen", False)
    if frozen:
        _log_to_file()
        engine_dir = _install_engine()

    import webview
    import server

    if frozen and not os.path.abspath(server.__file__).startswith(engine_dir):
        raise RuntimeError("server loaded from %s, not the installed engine; "
                           "numba would not cache" % server.__file__)

    httpd = server.make_server(port=0, workers=os.cpu_count() or 1,
                               quiet=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d/" % httpd.server_address[1]
    print("serving", url, "from", os.path.dirname(server.__file__))

    webview.create_window(TITLE, url, width=1240, height=900,
                          min_size=(760, 600))
    webview.start()
    httpd.shutdown()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
