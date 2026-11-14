-- PRICE OVERCHARGES
-- Invoice lines billed above the unit price agreed in the vendor's contract.

-- Step 1: JOIN attaches the agreed price to each invoice line, matching on
-- vendor + item. Lines with no contract for that vendor + item drop out (inner join).
WITH priced AS (
    SELECT
        i.invoice_id, i.vendor_id, i.vendor_needs_review, i.qty, i.unit_price,
        c.contract_price
    FROM invoices AS i
    JOIN contracts AS c
      ON c.vendor_id = i.vendor_id
     AND c.item = i.item
    WHERE i.doc_type = 'INV'
)

-- Step 2: keep lines billed above contract. The + 0.005 ignores rounding noise
-- (prices are stored as floats). Recoverable = overcharge per unit x quantity.
SELECT
    invoice_id,
    vendor_id,
    'price_overcharge' AS check_type,
    ROUND((unit_price - contract_price) * qty, 2) AS recoverable,
    vendor_needs_review,
    'Billed ' || printf('%.2f', unit_price) || ' per unit vs contract '
        || printf('%.2f', contract_price) || ', ' || qty || ' units' AS reason
FROM priced
WHERE unit_price > contract_price + 0.005;
