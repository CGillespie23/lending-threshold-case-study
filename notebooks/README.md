# Notebooks — what each one has to answer

Write these yourself. The outlines are the questions, not the answers.

## 01_population.ipynb — what process created this data?
- One row per loan? Prove it. Duplicates, nulls, month counts.
- Base rate overall and by month. Which months are complete?
- Composition over time: bank-type share by month. Is a rising aggregate rate
  composition or deterioration? Decompose it (within-group vs between-group).
- Cell sizes: bank_id x grade x stars. How many cells have under 30 lines?
- Decide the build window (Nov-Feb) and the holdout (Mar) and say why.

## 02_score.ipynb — one additive score
- Fit logistic regression on the build window: bank_type + grade + stars.
- Read the coefficients as log-odds offsets. Compare the bank offsets to the
  raw gap between bank types. Are they roughly constant across grades?
- Does card_ping_passed add anything beyond bank_type and grade? Test it
  properly (held-out log-loss or AUC with and without), then say what you'd do.
- Calibration: predicted vs observed by decile on the build window AND on March.
  Intercept shift? Slope problem? Both?
- Raw cell rates vs the additive score on March: which is closer, where, and
  how big is the noise on 3,100 lines?

## 03_threshold.ipynb — where to cut, and what it costs
- State the economics explicitly: contribution on a paying line, loss on a
  zero-pay line, origination cost per line. Invent them, label them invented.
- Build the settings ladder: for each zero-pay target, which cells fund,
  volume, expected loss rate, net contribution before and after cost.
- Marginal analysis: per-line contribution of the lines each rung adds.
  Where does the marginal line go negative after cost?
- Show the ladder on March. Did every target overshoot? By how much? Which
  rung would you pick "if recent months are the guide", and why?
- One paragraph on what this analysis cannot see: declined applicants,
  recoveries after day 100, fair-lending exposure of a bank-type term.
