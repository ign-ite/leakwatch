"""Smoke test: the app renders without errors on the bundled data."""
from streamlit.testing.v1 import AppTest


def test_app_renders():
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert at.metric[0].label == "Recoverable value flagged"
    assert len(at.expander) >= 4
