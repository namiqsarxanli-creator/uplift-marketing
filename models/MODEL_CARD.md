# Model card: SMS uplift model

| | |
|---|---|
| File | `uplift_model.pkl` (+ `model_meta.json`), produced by `notebooks/06_test_and_business.ipynb` |
| Type | T-learner: two logistic regressions (SMS / no SMS) in one scikit-learn Pipeline |
| Output | predicted change in purchase probability caused by the SMS (uplift), per client |
| Training data | 160,031 X5 RetailHero loyalty clients (train + valid), randomised SMS campaign, March 2019 |
| Features | 16 pre-campaign attributes: age, gender, card age, points redemption, purchase count, spend, basket, stores, recency |

## Intended use

Ranking existing loyalty-programme customers for an SMS or offer campaign, and choosing how many to contact for given
contact and offer costs (`uplift.business.plan_contacts`, API `/plan`).

## Performance (held-out test split, 40,008 clients, evaluated once)

| Metric | Value (95% bootstrap CI) |
|---|---|
| Extra purchase rate in the top 10% | +7.9 pp [4.8, 10.9] (campaign average +3.3 pp) |
| Extra purchase rate in the top 30% | +5.4 pp [3.6, 7.2] |
| Qini ×1000 | 1.99 [0.75, 3.30] |
| vs response model ("who will buy") | Qini +3.3 ×1000 [+1.2, +5.3], uplift@10% +7.2 pp [+3.9, +10.7] |

Cross-validated estimates during model selection were higher (Qini 3.37); the gap is within the test interval and
consistent with selection optimism.

## Calibration and limits

- The order of predicted deciles matches observed effects; the top decile is predicted at +8.9 pp and observed at
  +7.9 pp. The API reports effects through an isotonic calibration fitted on these deciles
  (`model_meta.json`), which never goes below +1.8 pp.
- Predictions are compressed at the bottom: the lowest decile is predicted negative but observed positive. **Do not
  use the model to label clients as harmed by contact** ("sleeping dogs"); no such segment was detectable.
- Trained on *active* customers only (all had purchased within 23 days before the campaign). Not validated for
  reactivating lapsed customers.
- One retailer, one campaign, one channel. A different message, offer or season changes who reacts; keep a random
  holdout in every campaign and retrain on it.
- Effect sizes are small (+3.3 pp average); decisions should use ranges, not single numbers.

## Leakage safeguards

- `first_redeem_date` is censored at the campaign start (raw data contains post-campaign dates).
- Basket totals are de-duplicated per transaction.
- Both are enforced by tests in `tests/`.

## Ethical notes

The model only decides who receives a marketing message; it does not affect prices or access to services. Age and
gender are used as features because they carry part of the effect; if the deployment context restricts such
targeting, retrain without them (step 4 shows the purchase-behaviour features keep most of the signal).
