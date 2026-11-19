import json

from contracts.extract import evidence_ok, extract_document, run_all

DOC = """RATE CONTRACT
1.1 The agreed price for malt (per kg) is   Rs. 52.40 per unit.
A freight surcharge of 2% applies to orders below 500 units."""


def term(quote="The agreed price for malt (per kg) is Rs. 52.40 per unit.", **overrides):
    t = {"vendor": "Konkan Agro Foods", "item": "malt (per kg)", "unit_price": 52.4,
         "rebate_pct": None, "rebate_threshold_qty": None, "effective_from": "2025-04-01",
         "effective_to": "2026-03-31", "source": "contract", "confidence": 0.95,
         "evidence_quote": quote}
    return t | overrides


def fake(*replies):
    """A stand-in for Gemini: returns the replies in order; an Exception reply is raised."""
    replies = list(replies)

    def call(prompt):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, {"prompt_tokens": 100, "output_tokens": 20}
    return call


def ok_reply(*terms):
    return json.dumps({"terms": list(terms)})


def test_evidence_ok_ignores_whitespace_but_not_wording():
    assert evidence_ok("price for malt (per kg) is Rs. 52.40", DOC)       # doc has extra spaces
    assert not evidence_ok("malt costs Rs. 52.40 per kg", DOC)            # paraphrase
    assert not evidence_ok("   ", DOC)


def test_extracts_terms_and_checks_evidence():
    record = extract_document("C01", DOC, fake(ok_reply(term(), term(quote="malt is cheap"))))
    assert record["status"] == "ok" and record["attempts"] == 1
    assert [t["evidence_ok"] for t in record["terms"]] == [True, False]
    assert record["terms"][0]["unit_price"] == 52.4


def test_malformed_json_is_retried_once():
    record = extract_document("C01", DOC, fake("{not json", ok_reply(term())))
    assert record["status"] == "ok" and record["attempts"] == 2
    assert record["prompt_tokens"] == 200 and record["output_tokens"] == 40


def test_schema_violation_counts_as_malformed():
    bad = json.dumps({"terms": [{"vendor": "X"}]})        # required fields missing
    record = extract_document("C01", DOC, fake(bad, bad))
    assert record["status"] == "failed" and record["terms"] == []
    assert "ValidationError" in record["error"]


def test_api_errors_fail_after_one_retry():
    record = extract_document("C01", DOC, fake(RuntimeError("429 quota"), RuntimeError("429 quota")))
    assert record["status"] == "failed" and record["attempts"] == 2
    assert "429 quota" in record["error"]
    assert record["latency_s"] >= 0


def test_run_all_writes_every_document(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "C01.txt").write_text(DOC)
    (docs / "C02.txt").write_text(DOC)
    out = tmp_path / "extractions.json"
    run_all(docs, out, fake(ok_reply(term()), RuntimeError("x"), RuntimeError("x")), model="fake-model")
    saved = json.loads(out.read_text())
    assert saved["model"] == "fake-model"
    assert [(d["doc_id"], d["status"]) for d in saved["documents"]] == [("C01", "ok"), ("C02", "failed")]
