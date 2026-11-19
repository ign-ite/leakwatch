"""Extract pricing and rebate terms from documents with Gemini, and verify each one.

Every extracted term must carry an evidence_quote that appears verbatim (ignoring
whitespace) in the document. A term whose quote can't be found is treated as possibly
hallucinated and sent to the review queue instead of being trusted.

Run once, offline (results are committed so the app needs no API key):
    uv run python -m contracts.extract
The key is read from .streamlit/secrets.toml (gitignored): GEMINI_API_KEY = "..."
"""
import json
import time
import tomllib
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent
MODEL = "gemini-2.5-flash"
PAUSE_S = 6.0  # stay under the free tier's requests-per-minute limit
RETRY_WAIT_S = 30.0  # after an API error (usually a rate limit), wait before the one retry

# prompt -> (JSON text, {"prompt_tokens": n, "output_tokens": n})
ModelCall = Callable[[str], tuple[str, dict]]


class Term(BaseModel):
    vendor: str
    item: str
    unit_price: float | None
    rebate_pct: float | None
    rebate_threshold_qty: int | None
    effective_from: date | None
    effective_to: date | None
    source: Literal["contract", "email"]
    confidence: float
    evidence_quote: str


class Extraction(BaseModel):
    terms: list[Term]


PROMPT = """You extract commercial terms from one supplier document (a contract or an email).

Return every term of these two kinds:
1. A unit price for an item (unit_price in INR as a plain number; rebate fields null).
2. A volume rebate: a percentage credited once purchases of an item reach a quantity
   (rebate_pct, rebate_threshold_qty as a whole number of units; unit_price null).

Rules:
- vendor is the supplier's company name; item is the product as written in the document.
- Dates are YYYY-MM-DD. Dates written like 01/08/2025 are day/month/year.
- effective_from / effective_to: the dates the term applies. If the document gives no end
  date (e.g. a revision "until further notice"), effective_to is null.
- Lines starting with ">" quote an earlier message. They are history, not current terms.
- Surcharges, payment terms and late-payment interest are NOT rebates. Ignore them.
- evidence_quote: copy, character for character, the sentence that states the term.
- confidence: 0 to 1, how sure you are the term is stated exactly as extracted.

Document:
\"\"\"
{text}
\"\"\"
"""


def _squash(s: str) -> str:
    return " ".join(s.split())


def evidence_ok(quote: str, text: str) -> bool:
    """The quote appears verbatim (ignoring whitespace) in the document's current text.
    Lines starting with '>' quote earlier messages, so they are not evidence."""
    current = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))
    return bool(quote.strip()) and _squash(quote) in _squash(current)


def value_in_quote(term: dict) -> bool:
    """The extracted price or rebate percentage is written in the quote itself."""
    quote = term["evidence_quote"]
    if term["unit_price"] is not None and f"{term['unit_price']:,.2f}" not in quote:
        return False
    if term["rebate_pct"] is not None and f"{term['rebate_pct']:g}%" not in quote:
        return False
    return True


def extract_document(doc_id: str, text: str, call_model: ModelCall,
                     retry_wait_s: float = RETRY_WAIT_S) -> dict:
    """Call the model (retrying once on any error or invalid output) and check evidence.
    Invalid output is retried at once; an API error (usually a rate limit) after a wait."""
    record = {"doc_id": doc_id, "status": "failed", "terms": [], "error": None,
              "attempts": 0, "prompt_tokens": 0, "output_tokens": 0, "latency_s": 0.0}
    for attempt in (1, 2):
        if attempt == 2 and record["error"] and not record["error"].startswith("ValidationError"):
            time.sleep(retry_wait_s)   # an API error (usually a rate limit): wait, untimed
        record["attempts"] = attempt
        start = time.perf_counter()
        try:
            raw, usage = call_model(PROMPT.format(text=text))
            record["prompt_tokens"] += usage["prompt_tokens"]
            record["output_tokens"] += usage["output_tokens"]
            parsed = Extraction.model_validate_json(raw)
        except ValidationError as err:  # malformed JSON or schema violation
            record["error"] = f"ValidationError: {err}"[:300]
            continue
        except Exception as err:  # API error
            record["error"] = f"{type(err).__name__}: {err}"[:300]
            continue
        finally:
            record["latency_s"] = round(record["latency_s"] + time.perf_counter() - start, 3)
        terms = [t.model_dump(mode="json") for t in parsed.terms]   # dates back to "YYYY-MM-DD"
        record["terms"] = [t | {"evidence_ok": evidence_ok(t["evidence_quote"], text) and value_in_quote(t)}
                           for t in terms]
        record.update(status="ok", error=None)
        break
    return record


def run_all(doc_dir: Path, out_path: Path, call_model: ModelCall, model: str = MODEL,
            pause_s: float = 0.0, retry_wait_s: float = 0.0) -> None:
    """Extract every document. Documents already extracted successfully in out_path are kept,
    so a run cut short by a daily quota can be finished later. Saves after each document."""
    results = {}   # doc_id -> record; every saved record stays until it is replaced
    if out_path.exists():
        results = {d["doc_id"]: d for d in json.loads(out_path.read_text())["documents"]}

    def save() -> None:
        records = [results[k] for k in sorted(results)]
        out_path.write_text(json.dumps({"model": model, "documents": records}, indent=2) + "\n")

    for path in sorted(doc_dir.glob("*.txt")):
        if results.get(path.stem, {}).get("status") == "ok":
            continue
        results[path.stem] = extract_document(path.stem, path.read_text(), call_model, retry_wait_s)
        print(f"{path.stem}: {results[path.stem]['status']}", flush=True)
        save()
        time.sleep(pause_s)
    save()


def gemini_call(api_key: str, model: str = MODEL) -> ModelCall:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(response_mime_type="application/json",
                                         response_schema=Extraction, temperature=0)

    def call(prompt: str) -> tuple[str, dict]:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        usage = response.usage_metadata
        # Thinking tokens are billed as output but not included in candidates_token_count.
        return response.text, {"prompt_tokens": usage.prompt_token_count or 0,
                               "output_tokens": (usage.candidates_token_count or 0)
                               + (usage.thoughts_token_count or 0)}
    return call


if __name__ == "__main__":
    secrets = tomllib.loads((ROOT / ".streamlit" / "secrets.toml").read_text())
    model = secrets.get("GEMINI_MODEL", MODEL)
    run_all(ROOT / "data" / "documents", Path(__file__).parent / "extractions.json",
            gemini_call(secrets["GEMINI_API_KEY"], model), model=model, pause_s=PAUSE_S,
            retry_wait_s=RETRY_WAIT_S)
