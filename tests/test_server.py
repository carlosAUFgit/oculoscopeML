import server


def _client(latest=None, trigger=None):
    app = server.create_app(lambda: latest, trigger)
    app.testing = True
    return app.test_client()


def test_latest_503_when_no_image():
    assert _client(None).get("/latest").status_code == 503


def test_latest_serves_jpeg_bytes():
    resp = _client(b"\xff\xd8fakejpeg").get("/latest")
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert resp.data == b"\xff\xd8fakejpeg"


def test_index_autorefresh_page():
    resp = _client().get("/")
    html = resp.data.decode()
    assert resp.status_code == 200
    assert "/latest" in html
    assert "1500" in html
    assert "localStorage" not in html
    assert "sessionStorage" not in html


def test_trigger_fires_callback_when_wired():
    calls = []
    resp = _client(trigger=lambda: calls.append(1)).post("/trigger")
    assert resp.status_code == 202
    assert calls == [1]


def test_trigger_absent_when_not_wired():
    assert _client().post("/trigger").status_code == 404
