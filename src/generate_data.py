"""
Synthetic consumer-lending data for the threshold case study.

Nothing here is real. The generator writes applications with a bank type, a
credit grade, a fraud-vendor star rating and a 100-day zero-pay outcome. The
outcome is drawn from a logistic model with ADDITIVE log-odds effects, so a
simple additive score can recover the structure and a raw cell-rate approach
will overfit the small cells. March carries extra drift for neobank lines so an
out-of-sample test on March has something to find.

Run:  python src/generate_data.py
Writes: data/applications.csv
"""
import numpy as np
import pandas as pd

SEED = 2026
N_PER_MONTH = {"2025-11": 3400, "2025-12": 3900, "2026-01": 4600, "2026-02": 4900, "2026-03": 3100}

BANK_TYPES = ["traditional", "other_flagged", "neobank"]
BANK_SHARE = {"2025-11": [0.78, 0.14, 0.08], "2025-12": [0.74, 0.16, 0.10],
              "2026-01": [0.66, 0.19, 0.15], "2026-02": [0.60, 0.21, 0.19],
              "2026-03": [0.58, 0.22, 0.20]}
BANK_OFFSET = {"traditional": 0.0, "other_flagged": 0.90, "neobank": 1.37}
# a few institutions per type, so routing-level cells are small and noisy
BANKS = {"traditional": [f"T{i:02d}" for i in range(1, 13)],
         "other_flagged": [f"F{i:02d}" for i in range(1, 6)],
         "neobank": ["N01", "N02", "N03"]}

GRADES = ["A", "B", "C", "D"]
GRADE_P = [0.38, 0.32, 0.20, 0.10]
GRADE_OFFSET = {"A": 0.0, "B": 0.50, "C": 0.90, "D": 1.30}

STARS = [5, 4, 3, 2, 1]
STAR_P = {"traditional": [0.45, 0.28, 0.15, 0.08, 0.04],
          "other_flagged": [0.30, 0.28, 0.20, 0.13, 0.09],
          "neobank": [0.25, 0.27, 0.22, 0.15, 0.11]}
STAR_OFFSET = {5: 0.0, 4: 0.20, 3: 0.45, 2: 0.60, 1: 0.90}

BASE_LOGIT = -2.0          # traditional / A / 5 stars  ->  about 12% zero pay
MONTH_DRIFT = {"2025-11": 0.0, "2025-12": 0.0, "2026-01": 0.05, "2026-02": 0.10,
               "2026-03": {"traditional": 0.10, "other_flagged": 0.25, "neobank": 0.45}}


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    loan_id = 100000
    for month, n in N_PER_MONTH.items():
        bank_type = rng.choice(BANK_TYPES, size=n, p=BANK_SHARE[month])
        for bt in bank_type:
            grade = rng.choice(GRADES, p=GRADE_P)
            stars = int(rng.choice(STARS, p=STAR_P[bt]))
            bank_id = rng.choice(BANKS[bt])
            # card ping correlates with bank type but adds nothing beyond it
            card_ping = int(rng.random() < {"traditional": 0.92, "other_flagged": 0.78, "neobank": 0.70}[bt])
            amount = float(np.round(np.exp(rng.normal(5.9, 0.35)), -1))   # ~$300-$600, not predictive
            drift = MONTH_DRIFT[month]
            drift = drift[bt] if isinstance(drift, dict) else drift
            logit = BASE_LOGIT + BANK_OFFSET[bt] + GRADE_OFFSET[grade] + STAR_OFFSET[stars] + drift
            p = 1 / (1 + np.exp(-logit))
            zero_pay = int(rng.random() < p)
            # paid share of first draw at 100 days: 0 if zero pay, otherwise skewed high
            paid_share = 0.0 if zero_pay else float(np.clip(rng.beta(5, 2), 0, 1))
            rows.append(dict(loan_id=loan_id, applied_month=month, bank_type=bt, bank_id=bank_id,
                             grade=grade, stars=stars, card_ping_passed=card_ping,
                             first_draw=amount, zero_pay_100d=zero_pay,
                             paid_share_100d=round(paid_share, 3)))
            loan_id += 1
    df = pd.DataFrame(rows)
    df.to_csv("data/applications.csv", index=False)
    print(f"wrote {len(df):,} rows to data/applications.csv")
    print(df.groupby(["applied_month"])["zero_pay_100d"].agg(["count", "mean"]).round(3))
    print(df.groupby("bank_type")["zero_pay_100d"].mean().round(3))


if __name__ == "__main__":
    main()
