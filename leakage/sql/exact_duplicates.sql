-- EXACT DUPLICATES
-- The same invoice (same vendor, invoice number and amount) paid more than once,
-- typically because it was entered in two different ERPs.

-- Step 1: count credit notes per invoice. A credit note reverses one payment,
-- so N credit notes cancel N repeat payments. GROUP BY collapses them to one
-- row per vendor + invoice number + amount, with the count.
WITH credits AS (
    SELECT vendor_id, invoice_no_clean, -amount AS amount, COUNT(*) AS n_credits
    FROM invoices
    WHERE doc_type = 'CRN'
    GROUP BY vendor_id, invoice_no_clean, -amount
),

-- Step 2: number the payments with two windows (window functions keep every row
-- and add a column computed across related rows).
--   same_number:  rows with the identical raw invoice number. copy_no > 1 means
--                 this exact number was already paid.
--   same_invoice: rows that are the same invoice once OCR misreads are undone
--                 (invoice_no_clean). repeat_no = 0 is the original payment,
--                 1, 2, ... are the repeats, in date order. This numbering is
--                 shared with ocr_duplicates.sql, so a credit note cancels a
--                 repeat no matter which check would have caught it.
numbered AS (
    SELECT
        invoice_id, vendor_id, vendor_needs_review, invoice_no, invoice_no_clean, amount,
        ROW_NUMBER()            OVER same_number  AS copy_no,
        FIRST_VALUE(invoice_id) OVER same_number  AS first_id,
        FIRST_VALUE(paid_date)  OVER same_number  AS first_paid,
        ROW_NUMBER()            OVER same_invoice - 1 AS repeat_no
    FROM invoices
    WHERE doc_type = 'INV' AND vendor_id IS NOT NULL
    WINDOW
        same_number  AS (PARTITION BY vendor_id, invoice_no, amount ORDER BY paid_date, invoice_id),
        same_invoice AS (PARTITION BY vendor_id, invoice_no_clean, amount ORDER BY paid_date, invoice_id)
)

-- Step 3: flag repeats of an identical number, skipping the first n_credits
-- repeats (those were already reversed). LEFT JOIN keeps invoices with no
-- credit notes; COALESCE turns their missing count into 0.
SELECT
    n.invoice_id,
    n.vendor_id,
    'exact_duplicate' AS check_type,
    n.amount AS recoverable,
    n.vendor_needs_review,
    'Same vendor, invoice no. ' || n.invoice_no || ' and amount as ' || n.first_id
        || ' (paid ' || n.first_paid || ')' AS reason
FROM numbered AS n
LEFT JOIN credits AS c
       ON c.vendor_id = n.vendor_id
      AND c.invoice_no_clean = n.invoice_no_clean
      AND c.amount = n.amount
WHERE n.copy_no > 1
  AND n.repeat_no > COALESCE(c.n_credits, 0);
