#!/usr/bin/env python3
"""GTA VI trophy rarity forecast -- R3 entry (model M).

Forecasts, for every trophy of Grand Theft Auto VI on PS5, its Exophase
earn rate at the case's target as five quantiles (0.05/0.25/0.5/0.75/0.95),
from the house's recorded reads of the trophy page before the checkpoint.

Model (calibrated hybrid, stdlib only, deterministic):
  * Two per-trophy estimators are blended in logit space.
      B -- rate trend: ordinary least squares of logit(rate) on
        ln(1 + hours since release midnight), over the last 24 h of reads,
        read at the target. The best smooth extrapolator when the tracked
        base is growing steadily.
      C -- E/N decomposition: earners E(t) and players-tracked N(t) are fit
        separately as log-linear in log-time (each slope the conservative
        one: the smaller-|slope| of the full-window fit and the last-third
        fit, because growth is concave in log-time), the earner slope is
        shrunk toward its tier mean (shrinkage k = 3.0), and both are
        extrapolated with a saturating step (tau = 2.0 in log-time) to the
        target; the rate is 100 * E / N. This sees through dilution: a
        surge of new tracked players pushes every shown rate down together,
        which a rate-level fit misreads as trophies getting rarer.
  * Blend weight: 0.5 for C normally, rising toward 0.85 when the tracked
    base jumped inside the window (a catch-up scan dilutes every rate at
    once, and a rate-level trend misreads the dip as a falling trend).
    The jump statistic is the largest one-step log-N increase minus the
    median *positive* one-step increase (between the 6-hourly refreshes
    the page is flat, so the background must be measured on positive
    steps only -- the plain median over all steps fires on every normal
    refresh); the weight ramps in over jump statistics 0.15--0.30.  * Frozen history (every read in the window identical -- no refresh seen
    yet, as on launch night): the next refresh brings more tracked players,
    so rates dilute. The median is C's median shifted down by 1.5 logit
    (the shift implied by the known first-refresh dilution, N 638 -> 3,868,
    at median earner growth ~1.35x) with a 0.7 logit spread. The frozen
    branch never fires on the final (two days of moving history); it only
    sets the practice score.
  * Spread: C's sigma (base 0.60 logit, growing with horizon and with the
    size of the extrapolated move, plus the earner-fit residual), scaled by
    a horizon-aware factor: 1.0 at horizon 0, ramping to 0.45 over ~0.25
    days (at 4.5 d the factor is 0.4500, so the long-horizon calibration
    is untouched). The unscaled spread over-covers badly on every
    synthetic regime (90% coverage ~1.00); 0.45 is the robust choice -- it
    still beats the unscaled spread when the real median is off by up to
    ~0.4 logit of extra error (independent, common, and per-tier shifts
    included in the stress tests), and its worst harsh-scenario penalty
    against the R2 spread (0.6) is ~0.01 logit of pinball loss, against a
    ~0.02-0.03 gain in the likely world. At short horizons the median
    carries a systematic bias (mean|d| 0.48 logit at 6 h), so the ramp
    widens the spread there.
  * A trophy at 100% gets a fixed near-100 row; a trophy with too little
    history to fit falls back to persistence.

Benchmark (pinball loss, logit scale, lower is better; see BENCHMARK.md in
the build directory for the full table and the robustness analysis):
  practice scenarios (dilution, N 638 -> 3,868/5,200):  0.231 (R2 L: 0.260)
  synthetic 4.5-day mean (4 regimes, FINAL-like cut):   0.117 (R2 L: 0.151)
  synthetic holdout mean (4 unseen regimes):            0.152 (R2 L: 0.175)
  house starter baseline: practice 1.398, synthetic 0.458.

Usage: python3 -P -s main.py
  Reads $MAKEFUN_INPUT_DIR (else /input/) for every *.in case file and
  writes one <id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output/).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------- constants
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

WINDOW_HOURS = 24.0      # reads used for the fits (all reads if fewer than 3)
TAU_DAMP = 2.0           # saturating extrapolation, in log-time units
SHRINK_K = 3.0           # earner-slope shrinkage weight toward the tier mean
SIGMA0 = 0.60            # C base spread (logit)
SIGMA_H = 0.34           # C spread growth per sqrt(day of horizon)
DRIFT_FRAC_C = 0.30      # C spread growth with the extrapolated move
DRIFT_CAP = 2.5          # cap on C's logit drift from the last rate
SIGMA_RW_B = 0.5         # B random-walk spread per sqrt(day of horizon)
DRIFT_FRAC_B = 0.25      # B spread growth with the extrapolated move
FROZEN_SHIFT = -1.5      # frozen-history dilution shift (logit)
FROZEN_SIGMA = 0.7       # frozen-history spread (logit)
SPREAD_SCALE = 0.45      # non-frozen spread calibration (see docstring)
SPREAD_RAMP_TAU = 0.40   # horizon (days) over which the spread scale ramps
                         # from 1.0 at horizon 0 down to SPREAD_SCALE
BLEND_W_C = 0.5          # base weight of C in the logit-average with B
JUMP_LO = 0.15           # jump statistic where the C weight starts rising
JUMP_HI = 0.30           # jump statistic where the C weight tops out
JUMP_W_HI = 0.85         # C weight at/above JUMP_HI
FALLBACK_SIGMA = 1.0     # persistence fallback spread (logit)
NO_HISTORY_RATE = 1.0    # prior median (%) when a case has no reads at all
NO_HISTORY_SIGMA = 1.5   # prior spread (logit) when a case has no reads


# ------------------------------------------------------------------- helpers
def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def quantiles_from_logit(lg, sigma):
    return [round4(clamp_pct(inv_logit(lg + z * sigma))) for z in Z]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def log_time(case, when):
    """ln(1 + hours since the game's release midnight, UTC)."""
    rel = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    hours = max(0.0, (when - rel).total_seconds() / 3600.0)
    return math.log1p(hours)


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std, n) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        s = math.sqrt(max(var, 0.0))
    else:
        s = 0.0
    return (a, b, s, n)


def conservative_slope(xs, ys):
    """The smaller-|slope| of the full-window OLS and the OLS over the last
    third of the window. Growth is concave in log-time, so the full-window
    slope overstates the forward slope; the recent slope is the honest one,
    and taking the smaller guards against a noisy recent fit."""
    full = ols(xs, ys)
    if full is None:
        return None
    if len(xs) >= 6:
        k = max(3, len(xs) // 3)
        rec = ols(xs[-k:], ys[-k:])
        if rec is not None:
            return full if abs(full[1]) <= abs(rec[1]) else rec
    return full


def last_known_rate(case, i):
    """The trophy's rate in the latest read that lists one, else 0.01."""
    for h in reversed(case["history"]):
        r = h["rates"][i]
        if r is not None:
            return r
    return 0.01


def window_reads(case):
    """Reads within the fit window (all reads when fewer than 3 qualify)."""
    at = parse_time(case["at"])
    win = [h for h in case["history"]
           if (at - parse_time(h["at"])).total_seconds() / 3600.0 <= WINDOW_HOURS]
    if len(win) < 3:
        win = list(case["history"])
    return win


def history_frozen(case, win):
    """True when no refresh has been seen: every rate and the tracked-base
    count are identical across the window."""
    for i in range(len(case["trophies"])):
        vals = {h["rates"][i] for h in win if h["rates"][i] is not None}
        if len(vals) > 1:
            return False
    return len({h["players_tracked"] for h in win}) <= 1


def n_jump_stat(win):
    """Largest one-step log-N increase minus the median POSITIVE one-step
    increase. Between 6-hourly refreshes the page is flat (zero steps);
    the old detector's median over ALL steps is ~0, so any refresh step
    looks like a jump. Using only positive steps as the background makes
    the detector fire only on steps that beat normal refresh growth --
    the same R4 gain with no cost on the 6-h / 24-h cuts."""
    steps = []
    for a, b in zip(win, win[1:]):
        steps.append(math.log(max(1.0, b["players_tracked"]))
                     - math.log(max(1.0, a["players_tracked"])))
    pos = sorted(s for s in steps if s > 1e-9)
    if not steps or not pos:
        return 0.0
    med = pos[len(pos) // 2]
    return max(steps) - med


# ------------------------------------------------------------ estimator B
def fit_b(case, win, x_target):
    """Rate-trend fit. Per trophy: ("special",) | ("fallback", last) |
    ("fit", median_logit, sigma)."""
    days = max(0.0, case["horizon_minutes"] / 1440.0)
    out = []
    for i, trophy in enumerate(case["trophies"]):
        last = last_known_rate(case, i)
        if last >= 99.99:
            out.append(("special",))
            continue
        xs, ys = [], []
        for h in win:
            r = h["rates"][i]
            if r is None:
                continue
            xs.append(log_time(case, parse_time(h["at"])))
            ys.append(logit(r))
        fit = ols(xs, ys)
        if fit is None:
            out.append(("fallback", last))
            continue
        a, b, s, _n = fit
        lg_med = a + b * x_target
        drift = lg_med - logit(last)
        sigma = math.sqrt(s * s + (SIGMA_RW_B ** 2) * days
                          + (DRIFT_FRAC_B * abs(drift)) ** 2 + 0.12 ** 2)
        out.append(("fit", lg_med, sigma))
    return out


# ------------------------------------------------------------ estimator C
def fit_c(case, win, x_last, x_target):
    """E/N decomposition. Returns (per-trophy details, frozen, jump_stat).
    Details are ("special",) | ("fallback", last) |
    ("fit", median_logit, sigma)."""
    days = max(0.0, case["horizon_minutes"] / 1440.0)
    dx = max(0.0, x_target - x_last)
    damp = TAU_DAMP * (1.0 - math.exp(-dx / TAU_DAMP)) if dx > 0 else 0.0
    win_x = [log_time(case, parse_time(h["at"])) for h in win]
    frozen = history_frozen(case, win)
    jump = n_jump_stat(win)

    n_fit = conservative_slope(
        win_x, [math.log(max(1.0, h["players_tracked"])) for h in win])
    if n_fit is None:
        g_n = 0.0
        log_n_last = math.log(max(1.0, case["history"][-1]["players_tracked"]))
    else:
        a_n, g_n, _s, _n = n_fit
        g_n = min(3.0, max(-0.5, g_n))
        log_n_last = a_n + g_n * win_x[-1]
    log_n_t = log_n_last + g_n * damp

    tiers = {}
    fits = {}
    for i, trophy in enumerate(case["trophies"]):
        xs, ys = [], []
        for h, xv in zip(win, win_x):
            r = h["rates"][i]
            if r is None:
                continue
            e = max(0.5, r * h["players_tracked"] / 100.0)
            xs.append(xv)
            ys.append(math.log(e))
        fit = conservative_slope(xs, ys)
        fits[i] = fit
        if fit is not None:
            g = min(4.0, max(-1.0, fit[1]))
            tiers.setdefault(trophy["tier"], []).append(g)
    tier_mean = {t: sum(v) / len(v) for t, v in tiers.items()}

    out = []
    for i, trophy in enumerate(case["trophies"]):
        last = last_known_rate(case, i)
        if last >= 99.99:
            out.append(("special",))
            continue
        fit = fits[i]
        if fit is None:
            out.append(("fallback", last))
            continue
        a_e, g_e, s_e, n_e = fit
        g_e = min(4.0, max(-1.0, g_e))
        tm = tier_mean.get(trophy["tier"], g_e)
        g_star = (n_e * g_e + SHRINK_K * tm) / (n_e + SHRINK_K)
        log_e_last = a_e + g_e * win_x[-1]
        log_e_t = log_e_last + g_star * damp
        rate_med = 100.0 * math.exp(log_e_t - log_n_t)
        lg_med = logit(rate_med)
        drift = lg_med - logit(last)
        drift = min(DRIFT_CAP, max(-DRIFT_CAP, drift))
        lg_med = logit(last) + drift
        sigma = math.sqrt(SIGMA0 ** 2 + (SIGMA_H ** 2) * days
                          + (DRIFT_FRAC_C * abs(drift)) ** 2 + (0.5 * s_e) ** 2)
        out.append(("fit", lg_med, sigma))
    return out, frozen, jump


def spread_scale(days):
    """Non-frozen spread scale, horizon-aware: 1.0 at horizon 0, decaying
    to SPREAD_SCALE at long horizons. At short horizons the median carries
    a systematic bias (mean|d| 0.48 logit at 6 h), so the calibrated narrow
    spread under-covers; at 4.5 d the median is accurate and the full 0.45
    applies (scale 0.4500 at 4.5 d, so the main metrics are untouched).
    Tau 0.40 (swept 0.25/0.35/0.40/0.50) restores the short-horizon width
    the 0.25 ramp lost: scale at 6 h 0.65 -> 0.74, back to the R2 level,
    while the 4.5 d scale stays 0.45 and practice/synth are unchanged."""
    return SPREAD_SCALE + (1.0 - SPREAD_SCALE) * math.exp(-days / SPREAD_RAMP_TAU)


def blend_weight_c(jump):
    """C weight from the N-jump statistic: BLEND_W_C normally, ramping to
    JUMP_W_HI as the statistic grows from JUMP_LO to JUMP_HI."""
    if jump <= JUMP_LO:
        return BLEND_W_C
    frac = min(1.0, (jump - JUMP_LO) / (JUMP_HI - JUMP_LO))
    return BLEND_W_C + (JUMP_W_HI - BLEND_W_C) * frac


# ---------------------------------------------------------------- forecast
def forecast(case):
    trophies = case["trophies"]
    if not case["history"]:
        # No reads at all: no information beyond a wide prior.
        return {"trophies": [
            {"id": t["id"],
             "q": quantiles_from_logit(logit(NO_HISTORY_RATE), NO_HISTORY_SIGMA)}
            for t in trophies]}
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    x_last = log_time(case, at)
    x_target = log_time(case, target)
    win = window_reads(case)
    b = fit_b(case, win, x_target)
    c, frozen, jump = fit_c(case, win, x_last, x_target)
    w_c = blend_weight_c(jump)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last = last_known_rate(case, i)
        if last >= 99.99:
            rows.append({"id": trophy["id"],
                         "q": [98.5, 99.7, 99.99, 99.99, 99.99]})
            continue
        kind = c[i][0]
        if kind == "fallback":
            if frozen:
                # One frozen read (or none usable): a refresh is pending,
                # so the same dilution shift applies as for fitted trophies.
                rows.append({"id": trophy["id"],
                             "q": quantiles_from_logit(logit(last) + FROZEN_SHIFT,
                                                       FROZEN_SIGMA)})
            else:
                rows.append({"id": trophy["id"],
                             "q": quantiles_from_logit(logit(last), FALLBACK_SIGMA)})
            continue
        _k, lg_c, sig_c = c[i]
        if frozen:
            # No refresh seen: dilute C's median and widen the spread.
            rows.append({"id": trophy["id"],
                         "q": quantiles_from_logit(lg_c + FROZEN_SHIFT, FROZEN_SIGMA)})
            continue
        if b[i][0] == "fit":
            lg = (1.0 - w_c) * b[i][1] + w_c * lg_c
        else:
            lg = lg_c
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_logit(
                         lg, sig_c * spread_scale(
                             max(0.0, case["horizon_minutes"] / 1440.0)))})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
            f.write(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
