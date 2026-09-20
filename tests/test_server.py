"""
The HTTP layer, which holds no opinions and must not acquire any.

`server.py` parses a body, calls `api`, and serialises the answer. So this
tests the things that go wrong between a browser and a function call, and
almost all of them are about refusing input politely:

  * a bad hand comes back **400 with the message `api` wrote**, because those
    messages were written for a person and the page shows them verbatim;
  * a query before the engine is warm comes back **503**, not a ninety-second
    hang;
  * `/static/..` cannot walk out of `static/`;
  * `/api/health` answers **while a query is running**, which is what lets the
    page say "busy" instead of appearing to freeze.

Most of it runs against a stub engine, so the suite does not pay for numba to
load. One test at the end uses the real one at two deals, to check the wiring
end to end.
"""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import api
import rotation as r
import server


class StubEngine:
    """`api.Engine`'s interface, answering instantly and recording the call."""

    def __init__(self, ready=True):
        self.ready = ready
        self.busy = False
        self.workers = 3
        self.calls = []
        self.raise_with = None

    # Real cards, because the answers still go through `as_dict()` on the way
    # out and a stub that cannot be serialised tests the wrong thing.
    HAND = tuple(r.parse_hand("TH AS AD KD JD"))
    UP = r.parse_card("9H")

    def evaluate(self, hand, up_card, seat=0, dealer=3, deals=api.DEALS, **kw):
        self.calls.append(("evaluate", hand, up_card, seat, dealer, deals, kw))
        if self.raise_with:
            raise self.raise_with
        return api.Evaluation(hand=self.HAND, up_card=self.UP, seat=seat,
                              dealer=dealer, deals=deals,
                              actions=(api.ActionEV(api.ORDER, 0.5, 0.1,
                                                    deals, deals),))

    def solve(self, hands, up_card, dealer=3, **kw):
        self.calls.append(("solve", hands, up_card, dealer, kw))
        if self.raise_with:
            raise self.raise_with
        return api.Solution(hands=(self.HAND,) * 4, up_card=self.UP,
                            dealer=dealer, auction=("all pass",), caller=None,
                            trump=None, alone=False, discard=None, value=0,
                            caller_score=0, caller_tricks=0, tricks=())


class ServerCase(unittest.TestCase):
    """One server per class, on a port the OS picks."""

    engine_factory = StubEngine

    @classmethod
    def setUpClass(cls):
        cls.engine = cls.engine_factory()
        if isinstance(cls.engine, api.Engine):
            cls.engine.warm()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.httpd.engine = cls.engine
        cls.httpd.deals = 7
        cls.httpd.quiet = True
        cls.httpd.daemon_threads = True
        cls.url = "http://127.0.0.1:%d" % cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=30) as reply:
            return reply.status, reply.read()

    def post(self, path, body, raw=None):
        data = raw if raw is not None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.url + path, data=data,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as reply:
                return reply.status, json.loads(reply.read())
        except urllib.error.HTTPError as why:
            return why.code, json.loads(why.read())


class TestRouting(ServerCase):

    def test_health_says_what_the_page_needs(self):
        status, body = self.get("/api/health")
        got = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(got["ready"], True)
        self.assertEqual(got["workers"], 3)
        self.assertEqual(got["default_deals"], 7)
        self.assertEqual(got["seconds_per_deal"], api.SECONDS_PER_DEAL)

    def test_the_page_is_served(self):
        status, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>Euchre calculator</title>", body)

    def test_unknown_paths(self):
        for path in ("/nope", "/api/nope"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.get(path)
            self.assertEqual(caught.exception.code, 404)
        self.assertEqual(self.post("/api/nope", {})[0], 404)

    def test_static_cannot_walk_out_of_static(self):
        for path in ("/static/../server.py", "/static/../../etc/passwd",
                     "/static/%2e%2e/server.py"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.get(path)
            self.assertIn(caught.exception.code, (403, 404), path)


class TestBodies(ServerCase):

    def test_a_body_is_required(self):
        status, got = self.post("/api/evaluate", None, raw=b"")
        self.assertEqual(status, 400)
        self.assertIn("JSON body", got["error"])

    def test_not_json(self):
        status, got = self.post("/api/evaluate", None, raw=b"{nope")
        self.assertEqual(status, 400)
        self.assertIn("not JSON", got["error"])

    def test_not_an_object(self):
        status, got = self.post("/api/evaluate", [1, 2, 3])
        self.assertEqual(status, 400)
        self.assertIn("JSON object", got["error"])

    def test_too_large(self):
        status, got = self.post("/api/evaluate", None,
                                raw=b"{" + b"x" * (server.MAX_BODY + 1))
        self.assertEqual(status, 400)
        self.assertIn("too large", got["error"])
        # and the connection survives well enough for the next one
        self.assertEqual(self.get("/api/health")[0], 200)

    def test_api_s_own_message_reaches_the_caller(self):
        """
        The page prints `error` verbatim, so it has to be worth reading.
        """
        self.engine.raise_with = ValueError("9Z is not a Euchre card")
        try:
            status, got = self.post("/api/evaluate",
                                    {"hand": "TH AS AD KD JD", "up_card": "9Z"})
        finally:
            self.engine.raise_with = None
        self.assertEqual(status, 400)
        self.assertEqual(got["error"], "9Z is not a Euchre card")

    def test_an_unexpected_failure_is_a_500(self):
        self.engine.raise_with = RuntimeError("the wheels came off")
        try:
            status, got = self.post("/api/evaluate", {"hand": "x", "up_card": "y"})
        finally:
            self.engine.raise_with = None
        self.assertEqual(status, 500)
        self.assertIn("the wheels came off", got["error"])


class TestTranslation(ServerCase):
    """What the body turns into, on the way to `api`."""

    def test_evaluate_defaults_come_from_the_server(self):
        self.engine.calls.clear()
        self.post("/api/evaluate", {"hand": "TH AS AD KD JD", "up_card": "9H"})
        kind, hand, up, seat, dealer, deals, kw = self.engine.calls[-1]
        self.assertEqual((kind, hand, up, seat, dealer), ("evaluate",
                         "TH AS AD KD JD", "9H", 0, 3))
        self.assertEqual(deals, 7, "the server's default should apply")
        self.assertEqual(kw, {})

    def test_evaluate_passes_the_knobs_through(self):
        self.engine.calls.clear()
        self.post("/api/evaluate", {
            "hand": "TH AS AD KD JD", "up_card": "9H", "seat": 2, "dealer": 0,
            "deals": 40, "play_sims": 8, "bid_sims": 9, "discard_sims": 10,
            "seed": 5, "actions": ["order", "pass"]})
        _, _, _, seat, dealer, deals, kw = self.engine.calls[-1]
        self.assertEqual((seat, dealer, deals), (2, 0, 40))
        self.assertEqual(kw, {"play_sims": 8, "bid_sims": 9,
                              "discard_sims": 10, "seed": 5,
                              "actions": ("order", "pass")})

    def test_solve_needs_a_list_of_hands(self):
        status, got = self.post("/api/solve", {"hands": "not a list",
                                               "up_card": "9H"})
        self.assertEqual(status, 400)
        self.assertIn("four hands", got["error"])

    def test_solve_passes_the_seat_through(self):
        self.engine.calls.clear()
        self.post("/api/solve", {"hands": ["a", "b", "c", "d"],
                                 "up_card": "9H", "dealer": 1, "seat": 2})
        kind, hands, up, dealer, kw = self.engine.calls[-1]
        self.assertEqual((kind, hands, up, dealer), ("solve",
                         ["a", "b", "c", "d"], "9H", 1))
        self.assertEqual(kw, {"seat": 2})

        self.post("/api/solve", {"hands": ["a", "b", "c", "d"],
                                 "up_card": "9H"})
        self.assertEqual(self.engine.calls[-1][4], {"seat": None})

    def test_every_reply_says_how_long_it_took(self):
        status, got = self.post("/api/evaluate",
                                {"hand": "TH AS AD KD JD", "up_card": "9H"})
        self.assertEqual(status, 200)
        self.assertIsInstance(got["took_seconds"], float)


class TestNotReady(ServerCase):
    """Before the engine is warm, a query is refused rather than queued."""

    engine_factory = lambda: StubEngine(ready=False)          # noqa: E731

    def test_health_still_answers(self):
        status, body = self.get("/api/health")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["ready"])

    def test_queries_are_refused(self):
        for path in ("/api/evaluate", "/api/solve"):
            status, got = self.post(path, {})
            self.assertEqual(status, 503, path)
            self.assertIn("warming up", got["error"])


class TestTheRealEngine(ServerCase):
    """One pass through the wiring with nothing stubbed out."""

    engine_factory = api.Engine

    def test_evaluate_and_solve_answer(self):
        status, got = self.post("/api/evaluate", {
            "hand": "TH AS AD KD JD", "up_card": "9H", "seat": 2, "dealer": 0,
            "deals": 8, "play_sims": 1, "bid_sims": 1, "discard_sims": 1})
        self.assertEqual(status, 200)
        self.assertEqual(got["hand"], ["TH", "AS", "AD", "KD", "JD"])
        self.assertEqual([a["action"] for a in got["actions"]],
                         list(api.ACTIONS))
        self.assertEqual(json.loads(json.dumps(got)), got)

        status, got = self.post("/api/solve", {
            "hands": ["AS KD JS JH AD", "TC KH AC KS TH",
                      "QS TS 9S JD 9D", "9C AH JC KC QH"],
            "up_card": "9H", "dealer": 3, "seat": 0})
        self.assertEqual(status, 200)
        self.assertEqual(got["trump"], "hearts")
        self.assertEqual(len(got["tricks"]), 5)
        self.assertEqual(json.loads(json.dumps(got)), got)


if __name__ == "__main__":
    unittest.main()
