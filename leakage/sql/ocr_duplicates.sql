-- OCR DUPLICATES
-- The same invoice paid twice, where scanning misread the invoice number the
-- second time (0 -> O, 1 -> I, 5 -> S, a dropped leading zero). The raw numbers
-- differ, but invoice_no_clean (misreads undone in Python) is the same.

-- Step 1: credit notes, exactly as in exact_duplicates.sql.
WITH credit_notes AS (
    SELECT vendor_id, invoice_no_clean, -amount AS reversed_amount
    FROM invoices
    WHERE doc_type = 'CRN'
),

-- Step 2: within each group of same vendor + cleaned number + amount, sorted by
-- payment date, LAG() looks one row back: "what was the previous payment?"
with_previous AS (
    SELECT
        invoice_id, vendor_id, vendor_needs_review, invoice_no, invoice_no_clean, amount, paid_date,
        LAG(invoice_id) OVER w AS prev_id,
        LAG(invoice_no) OVER w AS prev_invoice_no,
        LAG(paid_date)  OVER w AS prev_paid
    FROM invoices
    WHERE doc_type = 'INV' AND vendor_id IS NOT NULL
    WINDOW w AS (PARTITION BY vendor_id, invoice_no_clean, amount ORDER BY paid_date, invoice_id)
)

-- Step 3: flag a payment when there IS a previous one, its raw number is
-- different (identical numbers are exact duplicates, handled elsewhere), and it
-- came within 30 days. julianday() turns a date into a day count so we can subtract.
SELECT
    p.invoice_id,
    p.vendor_id,
    'ocr_duplicate' AS check_type,
    p.amount AS recoverable,
    p.vendor_needs_review,
    'Invoice no. ' || p.invoice_no || ' looks like a misread of ' || p.prev_invoice_no
        || ' (' || p.prev_id || ', same vendor and amount, '
        || CAST(julianday(p.paid_date) - julianday(p.prev_paid) AS INTEGER) || ' days apart)' AS reason
FROM with_previous AS p
WHERE p.prev_id IS NOT NULL
  AND p.invoice_no <> p.prev_invoice_no
  AND julianday(p.paid_date) - julianday(p.prev_paid) <= 30
  AND NOT EXISTS (
      SELECT 1 FROM credit_notes AS c
      WHERE c.vendor_id = p.vendor_id
        AND c.invoice_no_clean = p.invoice_no_clean
        AND ROUND(c.reversed_amount, 2) = ROUND(p.amount, 2)
  );
