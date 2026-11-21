[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/spy26/gta6-trophy-rarity-forecast-r3/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/spy26/gta6-trophy-rarity-forecast-r3/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast — R3: calibrated hybrid B+C, jump-aware

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its
Exophase earn rate at the case's target as five quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95), from the house's recorded reads of the
trophy page before the checkpoint. One file, `main.py` — Python 3.13
standard library only, no network, deterministic (two runs on the same
case write byte-identical output).

## The model

Two per-trophy estimators are blended in logit space:

- **B — rate trend.** Ordinary least squares of logit(rate) on
  ln(1 + hours since release midnight), over the last 24 h of reads, read
  at the target. The best smooth extrapolator when the tracked base is
  growing steadily.
- **C — E/N decomposition.** Earners E(t) and players-tracked N(t) are fit
  separately as log-linear in log-time — each slope the conservative one
  (the smaller-|slope| of the full-window fit and the last-third fit,
  because growth is concave in log-time), the earner slope shrunk toward
  its tier mean (shrinkage k = 3.0) — and both extrapolated with a
  saturating step (tau = 2.0 in log-time) to the target; the rate is
  100 * E / N. This sees through dilution: a surge of new tracked players
  pushes every shown rate down together, which a rate-level fit misreads
  as trophies getting rarer.

Two calibrations, both measured on the benchmark (see below), set this
entry apart from the R2 hybrid:

- **Jump-aware blend.** The C weight is 0.5 normally and rises to 0.85
  when the tracked base jumped inside the window: the jump statistic is
  the largest one-step log-N increase minus the median *positive*
  one-step increase (between the 6-hourly refreshes the page is flat, so
  the background must be measured on positive steps only — the plain
  median over all steps fires on every normal refresh). A catch-up scan
  dilutes every rate at once; the rate-level trend misreads that dip as
  trophies getting rarer, so the decomposition earns the extra weight.
  This fixes the weakest regime (late N-jump) without touching the
  short-horizon cuts.
- **Frozen history** (every read in the window identical — no refresh seen
  yet, as on launch night): the next refresh brings more tracked players,
  so rates dilute. The median is C's median shifted down by **1.5 logit**
  with a **0.7 logit** spread. The frozen branch never fires on the final
  (two days of moving history); it only sets the practice score.
- **Spread ×0.45.** The unscaled spread over-covers badly on every
  synthetic regime (90% coverage ~1.00). 0.45 is the robust choice at the
  improved median: it still beats the unscaled spread when the real
  median is off by up to ~0.4 logit of extra error (independent, common,
  and per-tier shifts included in the stress tests), and its worst
  harsh-scenario penalty against the R2 spread (0.6) is ~0.01 logit of
  pinball loss, against a ~0.02–0.03 gain in the likely world.

A trophy at 100% gets a fixed near-100 row; a trophy with too little
history to fit falls back to persistence.

## Benchmark (pinball loss, logit scale, lower is better)

| scenario | R3 (this entry) | R2 hybrid (L) | house starter |
|---|---:|---:|---:|
| practice scenarios (dilution) | **0.231** | 0.260 | 1.398 |
| synthetic 4.5-day mean, 4 regimes (FINAL-like) | **0.117** | 0.151 | 0.458 |
| synthetic holdout mean, 4 unseen regimes | **0.152** | 0.175 | — |
| practice, holdout multipliers | **0.192** | 0.233 | 1.383 |

Per-regime synthetic 4.5-day losses: R1 0.111, R2 0.121, R3 0.109,
R4 0.129 (R2 hybrid: 0.138 / 0.148 / 0.137 / 0.182). The entry wins every
4.5-day regime individually, main and holdout. Full tables, per-variant
results, and the spread-robustness analysis live in the build notes
(`BENCHMARK.md` in the builder's working directory, not part of this repo).

Caveats, stated plainly: the practice score is measured against dilution
scenarios (the true practice answers are the 04:30/05:00 page, unknown to
us); the frozen shift wins 9 of 10 scenario cells and across the whole
plausible multiplier range (1.0–2.5× earner growth), losing only the
highest-multiplier cell. The synthetic win is driven substantially by the
spread calibration, which is only as good as the synthetic world's median
accuracy — hence ×0.45 rather than the synthetic-optimal ×0.40 (whose
worst harsh-scenario penalty is ~3× larger for ~0.009 more synthetic
gain), and no global median shift (it buys 0.003 synthetic but costs up
to 0.04 under a systematic over-prediction).

## Run it

```sh
# the grader's way (one case per run, sandbox):
python3 -P -s main.py
# reads $MAKEFUN_INPUT_DIR (default /input/) for every *.in case,
# writes one <id>.out per case to $MAKEFUN_OUTPUT_DIR (default /output/)

# locally on the kit's practice cases:
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

Output is one JSON object per case, `{"trophies": [{"id": n, "q": [q05,
q25, q50, q75, q95]}, ...]}`, 55 rows, five non-decreasing quantiles in
0–100 per trophy. Runtime is ~0.06 s for two practice cases on one CPU.

## License

MIT — see `LICENSE`.
