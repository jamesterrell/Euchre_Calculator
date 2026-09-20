"""
A local web server over `api.py`. Standard library only.

    python server.py                  # http://127.0.0.1:8000
    python server.py --workers 10     # give it the machine
    python server.py --port 9000 --deals 2000

Three endpoints and a page:

    GET  /                  the page
    GET  /api/health        {"ready": bool, "busy": bool, ...}
    POST /api/evaluate      {"hand", "up_card", "seat", "dealer", "deals", ...}
    POST /api/solve         {"hands": [4], "up_card", "dealer", "seat"}

Both POST bodies are JSON and both replies are `as_dict()` straight off the
`api` result, so the page does no arithmetic and the server holds no opinions
about presentation.

## Why it is shaped like this

**It binds to localhost.** There is no authentication, no rate limiting and no
request size cap worth the name, because it is a calculator for one person on
one machine. Putting it on an interface anybody else can reach would need all
three, and none of them are here. `--host` exists so you can make that mistake
deliberately rather than by accident.

**It warms up before it serves.** The first call into the compiled engine
spends about a second loading cached machine code, and about eighty seconds
*compiling* it when the cache is cold -- which it is after any edit to
`bitcore.py` or `fastsim.py`. A background thread does that at boot; until it
finishes, `/api/health` reports `ready: false` and the query endpoints answer
503 rather than hanging. The page polls and says what it is waiting for.

**It serialises queries.** `api.Engine` holds a lock, because one query
already uses every thread it is given and two at once just oversubscribe
numba's pool. `ThreadingHTTPServer` is still the right base -- it lets
`/api/health` answer while a query is running, which is what makes the page
able to say "busy" instead of freezing.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import api

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

# Bodies are small -- five cards and a handful of integers. Anything larger is
# a mistake or a probe, and reading it into memory first is how you turn one
# into the other.
MAX_BODY = 64 * 1024


class Handler(BaseHTTPRequestHandler):
    """One request. The engine it talks to is set on the server object."""

    server_version = "EuchreCalculator/1.0"
    protocol_version = "HTTP/1.1"

    # -------------------------------------------------------------- replies

    def _send(self, status: int, body: bytes, kind: str, close: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: dict, close: bool = False):
        self._send(status, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8", close)

    def _fail(self, status: int, message: str):
        """
        Refuse, and hang up.

        This is HTTP/1.1, so the connection would otherwise be reused -- and a
        request whose body we did not read leaves unread bytes in the socket
        that the next request would be parsed out of. Rejecting without
        draining is exactly the case that happens (a body too large to want, a
        query that arrives before the engine is warm), so every refusal closes.
        Errors are rare; a connection is cheap.
        """
        self._json(status, {"error": message}, close=True)

    def log_message(self, fmt, *args):
        if self.server.quiet:
            return
        sys.stderr.write("  %s %s\n" % (self.address_string(), fmt % args))

    # --------------------------------------------------------------- routes

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            engine = self.server.engine
            return self._json(200, {
                "ready": engine.ready, "busy": engine.busy,
                "workers": engine.workers,
                "default_deals": self.server.deals,
                "seconds_per_deal": api.SECONDS_PER_DEAL,
            })
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        return self._fail(404, "no such path: %s" % path)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        route = {"/api/evaluate": self._evaluate,
                 "/api/solve": self._solve}.get(path)
        if route is None:
            return self._fail(404, "no such path: %s" % path)

        engine = self.server.engine
        if not engine.ready:
            return self._fail(503, "still warming up -- the compiled engine "
                                   "is loading. Try again in a moment.")
        try:
            body = self._body()
        except ValueError as why:
            return self._fail(400, str(why))

        started = time.time()
        try:
            payload = route(body)
        except ValueError as why:
            # Every rejection `api` makes is a ValueError with a message
            # written for a person, so it is safe and useful to pass through.
            return self._fail(400, str(why))
        except Exception as why:
            # Anything that is not a ValueError is a bug here rather than bad
            # input, so it goes to the log as well as to the caller.
            self.log_error("%s: %s", type(why).__name__, why)
            return self._fail(500, "%s: %s" % (type(why).__name__, why))
        payload["took_seconds"] = round(time.time() - started, 3)
        return self._json(200, payload)

    # ---------------------------------------------------------- the work

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("a JSON body is required")
        if length > MAX_BODY:
            # Read a little of it so the client is not writing into a socket
            # nobody is reading while it waits for the reply; `_fail` closes
            # the connection rather than draining the rest.
            self.rfile.read(min(length, MAX_BODY))
            raise ValueError("body too large: %d bytes, limit is %d"
                             % (length, MAX_BODY))
        try:
            got = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as why:
            raise ValueError("body is not JSON: %s" % why)
        if not isinstance(got, dict):
            raise ValueError("body must be a JSON object")
        return got

    def _evaluate(self, body: dict) -> dict:
        deals = int(body.get("deals") or self.server.deals)
        kw = {}
        for name in ("play_sims", "bid_sims", "discard_sims"):
            if body.get(name) is not None:
                kw[name] = int(body[name])
        if body.get("actions"):
            kw["actions"] = tuple(body["actions"])
        if body.get("seed") is not None:
            kw["seed"] = int(body["seed"])
        got = self.server.engine.evaluate(
            body.get("hand", ""), body.get("up_card", ""),
            seat=int(body.get("seat", 0)), dealer=int(body.get("dealer", 3)),
            deals=deals, **kw)
        return got.as_dict()

    def _solve(self, body: dict) -> dict:
        hands = body.get("hands")
        if not isinstance(hands, (list, tuple)):
            raise ValueError("hands must be a list of four hands")
        seat = body.get("seat")
        got = self.server.engine.solve(
            hands, body.get("up_card", ""),
            dealer=int(body.get("dealer", 3)),
            seat=None if seat is None else int(seat))
        return got.as_dict()

    # ------------------------------------------------------------- static

    def _static(self, name: str):
        # Resolve and confirm the result is still inside `static/`, so a path
        # with `..` in it cannot walk out of the directory.
        path = os.path.realpath(os.path.join(STATIC, name))
        if not path.startswith(os.path.realpath(STATIC) + os.sep):
            return self._fail(403, "not yours to read")
        if not os.path.isfile(path):
            return self._fail(404, "no such file: %s" % name)
        kind = {".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".svg": "image/svg+xml"}.get(os.path.splitext(path)[1],
                                             "application/octet-stream")
        with open(path, "rb") as handle:
            return self._send(200, handle.read(), kind)


def serve(host: str = "127.0.0.1", port: int = 8000, workers: int = 1,
          deals: int = api.DEALS, tt_bits=None, quiet: bool = False):
    """Start the server and block. Warms the engine in the background."""
    engine = api.Engine(workers=workers, tt_bits=tt_bits, deals=deals)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.engine = engine
    httpd.deals = deals
    httpd.quiet = quiet
    httpd.daemon_threads = True

    def warm():
        started = time.time()
        engine.warm()
        print("  engine ready in %.1fs" % (time.time() - started),
              file=sys.stderr)

    threading.Thread(target=warm, daemon=True).start()

    print("Euchre calculator on http://%s:%d  (%d thread%s, %d deals by "
          "default)" % (host, port, engine.workers,
                        "" if engine.workers == 1 else "s", deals),
          file=sys.stderr)
    print("  warming up -- the page will say when it is ready", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped", file=sys.stderr)
    finally:
        httpd.server_close()
    return httpd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind. The default is localhost, "
                             "and there is no authentication here -- changing "
                             "it should be a deliberate act")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1,
                        help="threads for one query. One query uses all of "
                             "them, and queries are serialised")
    parser.add_argument("--deals", type=int, default=api.DEALS,
                        help="default layouts per action when a request does "
                             "not say (default %d)" % api.DEALS)
    parser.add_argument("--tt-bits", type=int, default=None,
                        help="log2 of the shared transposition table")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="no per-request log line")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    serve(host=args.host, port=args.port, workers=args.workers,
          deals=args.deals, tt_bits=args.tt_bits, quiet=args.quiet)


if __name__ == "__main__":
    main()
