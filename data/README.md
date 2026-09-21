# Data

**Source:** UCI Machine Learning Repository — [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (dataset id 350).

**License:** CC BY 4.0 — verified on the UCI dataset page. Redistribution with attribution is permitted, so the CSV is committed directly in this folder rather than fetched by a download script.

**Citation:**
> Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

**File:** `default_of_credit_card_clients.csv` — converted from the source `.xls` (header row 2, `ID` column preserved, target column renamed from `default payment next month` to `default_payment_next_month`); values are otherwise unchanged. 30,000 rows, 25 columns, ~2.8 MB.

**Columns:** `ID`; `LIMIT_BAL` (credit limit, NT$); `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` (demographic — protected attributes, see `.ai/PROJECT_SPEC.md`); `PAY_0, PAY_2–PAY_6` (repayment status, past 6 months); `BILL_AMT1–6`, `PAY_AMT1–6` (bill and payment amounts, past 6 months); `default_payment_next_month` (target, 1 = default). No direct customer identifiers.
