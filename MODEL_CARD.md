# Model Card — Fair Explainable Credit Risk

## Purpose

This model estimates probability of default for credit card accounts and supports a model-risk-style approval review. The intended use is portfolio demonstration and analyst workflow review: compare a traditional WoE scorecard with a gradient boosting challenger, explain decline decisions with reason codes, audit fairness, and choose a threshold by expected cost.

The selected challenger is XGBoost because it outperformed the scorecard on held-out ranking metrics while still supporting additive contribution reason codes.

## Data

- Source: UCI Default of Credit Card Clients, dataset id 350, CC BY 4.0.
- Population: 30,000 Taiwan credit card accounts from 2005.
- Target: `default_payment_next_month`, with a 22.1% default rate.
- Split: 22,500 train rows and 7,500 held-out test rows, stratified by target, seed `20260101`.
- Model features: 19 repayment, bill amount, payment amount, and credit limit fields.
- Protected fields retained for audit and excluded from model features: `SEX`, `AGE`, `MARRIAGE`, `EDUCATION`.

## Performance

Held-out test results:

| Model | ROC-AUC | KS | Gini |
|---|---:|---:|---:|
| Logistic scorecard | 0.7752 | 0.4191 | 0.5504 |
| XGBoost | 0.7835 | 0.4365 | 0.5670 |

The strongest scorecard IV feature is `PAY_0` (`0.8628`), followed by prior repayment-status variables. SHAP global importance for XGBoost also ranks `PAY_0` first (`0.4706` mean absolute contribution).

## Decision Threshold

The threshold is selected by expected cost, with a missed default costing 5x a wrongly declined good account.

| Policy | Threshold | Approval rate | Approved default rate | Expected cost | Expected profit |
|---|---:|---:|---:|---:|---:|
| Naive cutoff | 0.5000 | 87.5% | 15.7% | 0.7263 | 0.0525 |
| Cost-based threshold | 0.1508 | 53.1% | 9.4% | 0.5473 | 0.2315 |

Expected profit is revenue from approved good accounts minus loss from approved defaulting accounts. Expected cost additionally counts declined good accounts as lost opportunity cost.

Cost/profit values are per account in model cost units: one approved default costs 5 units and one wrongly declined good account costs 1 unit. In units of one default's loss, divide the table values by 5. At a €4,000 average loss per default and 1,000 applicants, the cost-based threshold's expected profit is about €185,200.

## Explainability

Reason codes use XGBoost SHAP contribution values. Example declined applicant `ID 10745` has predicted default probability `88.5%`; top plain-language reasons are:

1. Most recent repayment status shows delinquency or delayed payment.
2. Prior-month repayment status shows delinquency or delayed payment.
3. Recent bill balance is high relative to the portfolio pattern.
4. Six-month repayment status shows delinquency or delayed payment.

Reason code examples are generated in `notebooks/03_explainability.ipynb` and saved to `reports/tables/declined_reason_codes.csv`.

## Fairness Findings

At the cost-based threshold:

| Attribute | Worst group by approval parity | Disparate impact ratio | TPR gap | Flag below 0.8 |
|---|---|---:|---:|---|
| SEX | Male | 0.914 | 0.002 | No |
| AGE band | <=25 | 0.762 | 0.131 | Yes |
| MARRIAGE | other | 0.782 | 0.308 | Yes |

Calibration is close by SEX and most age/marriage groups, but small groups are noisier. The undocumented marriage group has only 14 held-out accounts and should be monitored rather than overinterpreted.

Proxy-effect test: excluding protected attributes does not remove all disparities. The candidate model already excludes protected attributes and still flags AGE band and MARRIAGE approval-rate gaps below 0.8. A test model that includes protected attributes worsens several parity metrics, so exclusion helps but is not sufficient by itself.

## Limitations

- The data is historical and geographically specific; patterns may not transfer to a new issuer or time period.
- Approval recommendations are illustrative and do not incorporate income verification, affordability policy, adverse-action compliance, or operational override rules.
- The cost ratio is documented and reproducible, but not fit to real loss-given-default or margin data.
- Protected attributes are excluded from model features, but repayment and balance variables can still carry correlated proxy signal.
- Small audit groups have unstable fairness metrics.

## Monitoring Plan

- Monthly: ROC-AUC, KS, Gini, approval rate, approved default rate, and expected cost/profit.
- Monthly fairness: approval rate, disparate impact ratio, TPR gap, and calibration error by SEX, AGE band, MARRIAGE, and EDUCATION.
- Drift: population stability on top IV/SHAP features, especially `PAY_0`, `PAY_2`, `LIMIT_BAL`, and recent payment amounts.
- Explainability: monitor top reason-code distribution for sudden shifts.
- Governance trigger: retrain or review if KS drops by more than 0.05, any major group DI ratio remains below 0.8 for two consecutive months, calibration error exceeds 5 percentage points, or reason-code mix changes materially.
