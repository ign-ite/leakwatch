# Leakwatch

Finds money leaking out of accounts payable: invoices paid twice and invoices billed above the contract price.

**Live app:** _link added after deploy_

## What it does

Three ERP exports, each with its own formats (European dates and decimals, uppercase vendor names
with `PVT LTD`, `(India) Private Limited` variants and typos), are:

1. **Cleaned** into one table with pandas: dates and numbers parsed, vendor names matched to a
   master list (exact after normalizing, then fuzzy with `difflib`; weak matches are marked for review
   and names that match nothing are excluded and counted),
   and OCR misreads in invoice numbers undone (`O→0`, `I→1`, `S→5`, leading zeros).
2. **Checked** with three SQL rules in SQLite (`leakage/sql/`, commented step by step):
   - exact duplicates: `ROW_NUMBER()` over vendor + invoice number + amount
   - OCR duplicates: `LAG()` to the previous payment with the same cleaned number, within 30 days
   - price overcharges: `JOIN` to the contract price list
   Duplicates already reversed by a credit note are excluded.
3. **Ranked** by priority = recoverable value × confidence, and summarized in a downloadable audit memo.

The data is synthetic (`leakage/generate.py`, fixed seed) with planted problems recorded in an
answer key, so every rule has a measured precision and recall. It also plants legitimate payments that
look like leakage: duplicates already reversed by a credit note, recurring same-amount invoices, a
vendor reusing an invoice number a year later, and look-alike numbers (`O/0`, `I/1`, `S/5`). The first
two are ignored; the last two are flagged, which is why exact and OCR precision are about 0.93–0.94
rather than 1.0. The app shows why each planted problem was missed.

## Run it

```bash
uv sync
uv run streamlit run app.py
uv run pytest
```

Regenerate the data with `uv run python -m leakage.generate`.

## Limitations

- Synthetic data is cleaner than real AP data.
- No multi-currency invoices or partial payments.
- Confidence scores are hand-set, not learned.
- Exact matches have no time limit, so a reused invoice number a year later is flagged.
