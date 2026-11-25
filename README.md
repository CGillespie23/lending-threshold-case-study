# Setting a lending threshold: one additive score, tested out of sample

*Synthetic data. No real lender, bank, applicant or dollar figure appears here.*

## The question

A consumer lender funds applications across three bank types, four credit
grades and five fraud-vendor star ratings. Zero-pay at 100 days runs around
30% overall and varies several-fold across cells. Where should the funding
threshold sit, and how much does each rung of tightening cost in volume and
return in avoided loss?

## What this repository shows

<!-- Fill in after the notebooks are done. Three to five sentences, written
     for an executive who will not open the notebooks. Lead with the finding. -->

## Approach

1. **Population first.** One row per loan, confirmed. Build window November to
   February; March held out. Composition shift by bank type decomposed from
   within-group change before any model is fitted.
2. **One additive score** (logistic regression on bank type, grade, stars)
   instead of raw cell rates, because small cells overfit and an additive
   model borrows strength across them.
3. **Out-of-sample test** on March, with the noise stated.
4. **Threshold economics.** A ladder of zero-pay targets, net contribution
   before and after an explicit per-line cost, and the marginal line at each rung.

## Results

<!-- Table: target | volume | observed zero pay (build) | observed (March) |
     net before cost | net after cost | marginal per added line -->

## What this cannot see

<!-- Declined applicants, recoveries after day 100, the fair-lending exposure
     of a bank-type term. Say these plainly. -->

## Reproduce

```
pip install -r requirements.txt
python src/generate_data.py
jupyter lab notebooks/
```
