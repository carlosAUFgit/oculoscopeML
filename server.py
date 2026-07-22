from __future__ import annotations

import logging
from typing import Callable

from flask import Flask, Response

log = logging.getLogger(__name__)

_PAGE = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>Oculoscope</title></head>
<body style="margin:0;background:#111;text-align:center">
<img id="view" src="/latest" alt="latest capture"
     style="max-width:100%;max-height:100vh">
<script>
setInterval(function () {
  document.getElementById('view').src = '/latest?t=' + Date.now();
}, 1500);
</script>
</body>
</html>"""


def create_app(get_latest_fn: Callable[[], bytes | None],
               trigger_fn: Callable[[], None] | None = None) -> Flask:
    app = Flask(__name__)

    @app.get("/latest")
    def latest() -> Response:
        data = get_latest_fn()
        if data is None:
            return Response("no image yet", status=503, mimetype="text/plain")
        return Response(data, mimetype="image/jpeg")

    @app.get("/")
    def index() -> Response:
        return Response(_PAGE, mimetype="text/html")

    if trigger_fn is not None:
        @app.post("/trigger")
        def trigger() -> Response:
            trigger_fn()
            return Response("triggered", status=202, mimetype="text/plain")

    return app


def start_server(get_latest_fn: Callable[[], bytes | None], port: int,
                 trigger_fn: Callable[[], None] | None = None) -> None:
    log.info("server: listening on 0.0.0.0:%d", port)
    create_app(get_latest_fn, trigger_fn).run(host="0.0.0.0", port=port,
                                              threaded=True)
