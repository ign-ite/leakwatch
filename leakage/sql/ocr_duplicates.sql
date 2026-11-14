-- OCR DUPLICATES
-- The same invoice paid twice, where scanning misread the invoice number the
-- second time (0 -> O, 1 -> I, 5 -> S, a dropped leading zero). The raw numbers
-- differ, but invoice_no_clean (misreads undone in Python) is the same.

-- Step 1: count credit notes per invoice, exactly as in exact_duplicates.sql.
WITH credits AS (
    SELECT vendor_id, invoice_no_clean, -amount AS amount, COUNT(*) AS n_credits
    FROM invoices
    WHERE doc_type = 'CRN'
    GROUP BY vendor_id, invoice_no_clean, -amount
),

-- Step 2: within each invoice (same vendor + cleaned number + amount), sorted
-- by payment date:
--   LAG() looks one row back: "what was the previous payment?"
--   repeat_no numbers the repeats exactly as in exact_duplicates.sql.
--   copy_no counts identical raw numbers; copy_no > 1 is an exact duplicate,
--   which the other check reports, so it is skipped here (no double counting).
numbered AS (
    SELECT
        invoice_id, vendor_id, vendor_needs_review, invoice_no, invoice_no_clean, amount, paid_date,
        LAG(invoice_id)  OVER same_invoice AS prev_id,
        LAG(invoice_no)  OVER same_invoice AS prev_invoice_no,
        LAG(paid_date)   OVER same_invoice AS prev_paid,
        ROW_NUMBER()     OVER same_invoice - 1 AS repeat_no,
        ROW_NUMBER()     OVER same_number AS copy_no
    FROM invoices
    WHERE doc_type = 'INV' AND vendor_id IS NOT NULL
    WINDOW
        same_invoice AS (PARTITION BY vendor_id, invoice_no_clean, amount ORDER BY paid_date, invoice_id),
        same_number  AS (PARTITION BY vendor_id, invoice_no, amount ORDER BY paid_date, invoice_id)
)

-- Step 3: flag a payment when it is the first appearance of its raw number,
-- there IS a previous payment with a different raw number, it came within
-- 30 days (julianday() turns a date into a day count so we can subtract),
-- and it is not one of the repeats already reversed by a credit note.
SELECT
    n.invoice_id,
    n.vendor_id,
    'ocr_duplicate' AS check_type,
    n.amount AS recoverable,
    n.vendor_needs_review,
    'Invoice no. ' || n.invoice_no || ' looks like a misread of ' || n.prev_invoice_no
        || ' (' || n.prev_id || ', same vendor and amount, '
        || CAST(julianday(n.paid_date) - julianday(n.prev_paid) AS INTEGER) || ' days apart)' AS reason
FROM numbered AS n
LEFT JOIN credits AS c
       ON c.vendor_id = n.vendor_id
      AND c.invoice_no_clean = n.invoice_no_clean
      AND c.amount = n.amount
WHERE n.copy_no = 1
  AND n.prev_id IS NOT NULL
  AND n.invoice_no <> n.prev_invoice_no
  AND julianday(n.paid_date) - julianday(n.prev_paid) <= 30
  AND n.repeat_no > COALESCE(c.n_credits, 0);
