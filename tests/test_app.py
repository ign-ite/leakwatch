"""Smoke tests: the app renders without errors on the bundled data."""
import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def run_app(monkeypatch, extractions_path):
    monkeypatch.setenv("LEAKWATCH_EXTRACTIONS", str(extractions_path))
    return AppTest.from_file("../app.py", default_timeout=60).run()


def perfect_extractions(tmp_path) -> Path:
    """What a flawless model would return for every document, plus one failed document."""
    names = dict(pd.read_csv(ROOT / "shared" / "vendors.csv").values)
    key = json.loads((ROOT / "data" / "contract_answer_key.json").read_text())
    docs = []
    for d in key:
        text = (ROOT / "data" / "documents" / f"{d['doc_id']}.txt").read_text()
        quote = text.splitlines()[0]
        terms = [t | {"vendor": names[t["vendor_id"]], "source": d["type"], "confidence": 0.99,
                      "evidence_quote": quote, "evidence_ok": True} for t in d["terms"]]
        docs.append({"doc_id": d["doc_id"], "status": "ok", "terms": terms, "error": None,
                     "attempts": 1, "prompt_tokens": 500, "output_tokens": 80, "latency_s": 1.2})
    docs[-1] = docs[-1] | {"status": "failed", "terms": [], "error": "RuntimeError: 429"}
    path = tmp_path / "extractions.json"
    path.write_text(json.dumps({"model": "test-model", "documents": docs}))
    return path


def test_app_renders_tab1_only_without_extractions(monkeypatch, tmp_path):
    at = run_app(monkeypatch, tmp_path / "missing.json")
    assert not at.exception
    assert at.metric[0].label == "Recoverable value flagged"
    assert len(at.tabs) == 1


def test_app_renders_contract_tab_with_extractions(monkeypatch, tmp_path):
    at = run_app(monkeypatch, perfect_extractions(tmp_path))
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Invoice leakage", "Contract terms"]
    labels = {m.label: m.value for m in at.metric}
    assert labels["Fields exactly right"] == "100%"
    assert labels["Extra terms"] == "0"
    assert labels["Documents that failed"] == "1 of 40"
    # with perfect extraction, the trusted-terms path must recover exactly what the true terms do
    assert labels["Recoverable with extracted terms"] == labels["Recoverable with the true terms"]
