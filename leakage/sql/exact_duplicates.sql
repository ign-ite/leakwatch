-- EXACT DUPLICATES
-- The same invoice (same vendor, invoice number and amount) paid more than once,
-- typically because it was entered in two different ERPs.

-- Step 1: credit notes. A credit note reverses a payment, so a duplicate that
-- already has a matching credit note is NOT money lost. Collect them to exclude later.
WITH credit_notes AS (
    SELECT vendor_id, invoice_no_clean, -amount AS reversed_amount
    FROM invoices
    WHERE doc_type = 'CRN'
),

-- Step 2: number every copy of each invoice.
--   PARTITION BY groups rows with the same vendor + invoice number + amount.
--   ORDER BY paid_date makes the first payment copy_no = 1.
--   Any copy_no > 1 is a repeat payment of the same invoice.
numbered AS (
    SELECT
        invoice_id, vendor_id, vendor_needs_review, invoice_no, invoice_no_clean, amount,
        ROW_NUMBER()             OVER w AS copy_no,
        FIRST_VALUE(invoice_id)  OVER w AS first_id,
        FIRST_VALUE(paid_date)   OVER w AS first_paid
    FROM invoices
    WHERE doc_type = 'INV' AND vendor_id IS NOT NULL
    WINDOW w AS (PARTITION BY vendor_id, invoice_no, amount ORDER BY paid_date, invoice_id)
)

-- Step 3: keep the repeats, minus any a credit note already reversed.
SELECT
    n.invoice_id,
    n.vendor_id,
    'exact_duplicate' AS check_type,
    n.amount AS recoverable,
    n.vendor_needs_review,
    'Same vendor, invoice no. ' || n.invoice_no || ' and amount as ' || n.first_id
        || ' (paid ' || n.first_paid || ')' AS reason
FROM numbered AS n
WHERE n.copy_no > 1
  AND NOT EXISTS (
      SELECT 1 FROM credit_notes AS c
      WHERE c.vendor_id = n.vendor_id
        AND c.invoice_no_clean = n.invoice_no_clean
        AND ROUND(c.reversed_amount, 2) = ROUND(n.amount, 2)
  );
