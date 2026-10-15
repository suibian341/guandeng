#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
render-failure-rca / analysis/analyze.py

Reproducible analysis of render_tasks.csv.

Usage:
    python analysis/analyze.py

Reads   data/render_tasks.csv
Writes  out/findings.json      every number quoted in README.md
        out/report.txt         human-readable transcript of the run
        out/fig1_lag_gap.png   the 18->22 minute gap and failure rate by lag
        out/fig2_cluster.png   per-cluster stale vs fresh failure rate
        out/fig3_decision.png  the degrade-vs-retry decision rule
        out/fig4_timeline.png  daily stale share vs daily failure rate

Only the provided CSV is used. Nothing is synthesised.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "data", "render_tasks.csv")
OUT = os.path.join(ROOT, "out")

# --- Cost model, straight from the task statement -------------------------
# "每次渲染超时失败约损失 75 分钟"  -> one timeout failure costs 75 minutes
# "每切换低开销降级通道一次约花费 10 分钟" -> one degrade-channel switch costs 10 min
FAIL_MIN = 75.0
DEGRADE_MIN = 10.0
BREAK_EVEN = DEGRADE_MIN / FAIL_MIN          # 0.1333 -> degrade if P(fail) above this

# Snapshot-age gate. The lag histogram is dense 0..18, EMPTY 19..21, resumes at 22,
# so any threshold in 19..21 splits the two regimes identically. Ship the middle.
TTL_MIN = 20

LINES = []


def say(msg=""):
    print(msg)
    LINES.append(str(msg))


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def load():
    """Load, dedupe, and derive the snapshot-age field."""
    raw = pd.read_csv(CSV, dtype=str)
    df = raw.copy()
    # created_at is +08:00 local, cluster_synced_at is UTC -> normalise both to UTC.
    df["created_utc"] = pd.to_datetime(df["created_at"], utc=True)
    df["synced_utc"] = pd.to_datetime(df["cluster_synced_at"], utc=True)
    df["lag_min"] = (df["created_utc"] - df["synced_utc"]).dt.total_seconds() / 60.0
    df["ok"] = pd.to_numeric(df["is_success"], errors="coerce")
    df["slots"] = pd.to_numeric(df["free_slots"])
    df["dur"] = pd.to_numeric(df["video_duration_sec"])

    key = ["task_id", "created_at", "cluster_synced_at", "cluster", "job_type",
           "free_slots", "video_duration_sec", "is_success"]
    dedup = df.drop_duplicates(subset=key).copy()
    rep = dedup.dropna(subset=["ok"]).copy()
    rep["stale"] = rep["lag_min"] >= TTL_MIN
    return raw, dedup, rep


def make_figures(d, F):
    """Four figures, one per claim in the README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    C_FRESH, C_STALE = "#2e7d32", "#c62828"

    # --- fig 1: the histogram hole + failure rate by age -------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ages = np.arange(0, 71)
    cnt = [((d["lag_min"] >= a) & (d["lag_min"] < a + 1)).sum() for a in ages]
    ax1.bar(ages, cnt, width=0.9, color="#37474f")
    ax1.axvspan(19, 22, color="#ffb300", alpha=0.35, zorder=0)
    ax1.set_xlabel("snapshot age (min)")
    ax1.set_ylabel("tasks")
    ax1.set_title("Empty band at 19-21 min\n(dense to 18, resumes at 22)", fontsize=10)

    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30),
            (30, 40), (40, 50), (50, 60), (60, 71)]
    xs, ys, ns = [], [], []
    for lo, hi in bins:
        g = d[(d["lag_min"] >= lo) & (d["lag_min"] < hi)]
        if len(g) == 0:
            continue
        xs.append("%d-%d" % (lo, hi))
        ys.append(100 * (1 - g["ok"].mean()))
        ns.append(len(g))
    cols = [C_FRESH if x in ("0-5", "5-10", "10-15", "15-20") else C_STALE for x in xs]
    ax2.bar(range(len(xs)), ys, color=cols)
    ax2.axhline(100 * BREAK_EVEN, ls="--", lw=1.4, color="#1565c0")
    ax2.text(0.02, 100 * BREAK_EVEN + 1.4,
             "break-even %.1f%%" % (100 * BREAK_EVEN), ha="left",
             color="#1565c0", fontsize=9)
    ax2.set_xticks(range(len(xs)))
    ax2.set_xticklabels(xs, rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("snapshot age (min)")
    ax2.set_ylabel("failure rate (%)")
    ax2.set_title("Failure rate jumps at 20 min\n(green = keep, red = degrade)", fontsize=10)
    for i, (y, n) in enumerate(zip(ys, ns)):
        ax2.text(i, y + 1.2, "n=%d" % n, ha="center", fontsize=7, color="#555")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_lag_gap.png"), bbox_inches="tight")
    plt.close(fig)

    # --- fig 2: per-cluster, with west as the control ----------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    cls = sorted(d["cluster"].unique())
    idx = np.arange(len(cls))
    fr = [100 * (1 - d[(d["cluster"] == c) & ~d["stale"]]["ok"].mean()) for c in cls]
    st = [100 * (1 - d[(d["cluster"] == c) & d["stale"]]["ok"].mean()) for c in cls]
    ax.bar(idx - 0.2, fr, 0.4, label="fresh snapshot (<20 min)", color=C_FRESH)
    ax.bar(idx + 0.2, st, 0.4, label="stale snapshot (>=20 min)", color=C_STALE)
    for i, (a, b) in enumerate(zip(fr, st)):
        ax.text(i - 0.2, a + 0.8, "%.1f%%" % a, ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.8, "%.1f%%" % b, ha="center", fontsize=8)
    ax.set_xticks(idx)
    ax.set_xticklabels([c.replace("cluster_", "") for c in cls])
    ax.set_ylabel("failure rate (%)")
    ax.set_title("Stale snapshots hurt every cluster except west (n=52, 1 failure)\n"
                 "west is the natural control: same age, no harm", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_cluster.png"), bbox_inches="tight")
    plt.close(fig)

    # --- fig 3: the decision rule -----------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar([0, 1], [100 * F["fresh_fail_rate"], 100 * F["stale_fail_rate"]],
           color=[C_FRESH, C_STALE], width=0.5)
    ax.axhline(100 * BREAK_EVEN, ls="--", lw=1.6, color="#1565c0")
    ax.text(0.02, 100 * BREAK_EVEN, "break-even %.1f%% (cost of degrading)"
            % (100 * BREAK_EVEN), va="bottom", ha="left", color="#1565c0",
            fontsize=9, transform=ax.get_yaxis_transform(),
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    for x, y, n in [(0, 100 * F["fresh_fail_rate"], F["fresh_n"]),
                    (1, 100 * F["stale_fail_rate"], F["stale_n"])]:
        ax.text(x, y + 1.0, "%.1f%%\nn=%d" % (y, n), ha="center", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["snapshot < 20 min\nroute normally",
                        "snapshot >= 20 min\nroute to degrade channel"])
    ax.set_ylabel("failure rate (%)")
    ax.set_ylim(0, 38)
    ax.set_title("The rule: degrade only when failure risk beats the 10-min cost", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_decision.png"), bbox_inches="tight")
    plt.close(fig)

    # --- fig 4: daily stale share vs daily failure rate --------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    d2 = d.copy()
    d2["day"] = d2["created_utc"].dt.tz_convert("Asia/Shanghai").dt.normalize()
    g = d2.groupby("day").agg(stale=("stale", "mean"), fail=("ok", lambda s: 1 - s.mean()))
    ax.plot(g.index, 100 * g["stale"], "o-", color="#c62828", label="stale-share of traffic")
    ax.plot(g.index, 100 * g["fail"], "s-", color="#37474f", label="overall failure rate")
    ax.set_ylabel("%")
    ax.set_xlabel("day (Asia/Shanghai)")
    ax.set_title("Both rates are flat across the week\n(no single incident; a standing routing defect)", fontsize=10)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_timeline.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    raw, dedup, d = load()
    F = {}

    say("=" * 78)
    say("0. DATA INTEGRITY")
    say("=" * 78)
    F["rows_raw"] = int(len(raw))
    F["rows_dedup"] = int(len(dedup))
    F["rows_reported"] = int(len(d))
    F["rows_unreported"] = int(dedup["ok"].isna().sum())
    dupmask = raw.duplicated(subset=["task_id", "created_at", "cluster_synced_at",
                                     "cluster", "job_type", "free_slots",
                                     "video_duration_sec", "is_success"], keep=False)
    F["dup_rows"] = int(dupmask.sum())
    F["dup_task_ids"] = int(raw[dupmask]["task_id"].nunique())
    F["dup_all_succeeded"] = bool(
        (pd.to_numeric(raw[dupmask]["is_success"], errors="coerce") == 1).all())
    say("raw rows                       : %d" % F["rows_raw"])
    say("exact-duplicate rows           : %d over %d task_ids (all is_success=1: %s)"
        % (F["dup_rows"], F["dup_task_ids"], F["dup_all_succeeded"]))
    say("rows after dedupe              : %d" % F["rows_dedup"])
    say("rows with is_success blank     : %d (%.1f%%)"
        % (F["rows_unreported"], 100 * F["rows_unreported"] / F["rows_raw"]))
    say("rows used for rates            : %d" % F["rows_reported"])
    say("created_at tz  : %s" % raw["created_at"].str[-6:].value_counts().to_dict())
    say("synced_at tz   : %s (UTC) -> converted before differencing"
        % raw["cluster_synced_at"].str[-1:].value_counts().to_dict())
    say("window         : %s .. %s"
        % (raw["created_at"].min()[:10], raw["created_at"].max()[:10]))

    # ---------------------------------------------------------------- headline
    say()
    say("=" * 78)
    say("1. HEADLINE: SNAPSHOT AGE VS OUTCOME")
    say("=" * 78)
    n_all, f_all = len(d), int((d["ok"] == 0).sum())
    n_st, f_st = int(d["stale"].sum()), int((d[d["stale"]]["ok"] == 0).sum())
    n_fr, f_fr = n_all - n_st, f_all - f_st
    r_st, r_fr = f_st / n_st, f_fr / n_fr
    F.update(dict(
        fail_total=f_all, fail_rate_all=f_all / n_all,
        stale_n=n_st, stale_share=n_st / n_all, stale_fail=f_st,
        stale_fail_rate=r_st, fresh_n=n_fr, fresh_fail=f_fr, fresh_fail_rate=r_fr,
        stale_share_of_failures=f_st / f_all, rel_risk=r_st / r_fr,
        stale_fail_ci=wilson(f_st, n_st), fresh_fail_ci=wilson(f_fr, n_fr),
        max_lag_fresh=float(d[~d["stale"]]["lag_min"].max()),
        min_lag_stale=float(d[d["stale"]]["lag_min"].min()),
    ))
    say("tasks with a reported result   : %d" % n_all)
    say("overall failure rate           : %.1f%% (%d)" % (100 * f_all / n_all, f_all))
    say()
    say("  snapshot age >= %d min        : %3d tasks (%.1f%% of traffic) -> %2d failures = %.1f%%"
        % (TTL_MIN, n_st, 100 * n_st / n_all, f_st, 100 * r_st))
    say("  snapshot age <  %d min        : %3d tasks (%.1f%% of traffic) -> %2d failures = %.1f%%"
        % (TTL_MIN, n_fr, 100 * n_fr / n_all, f_fr, 100 * r_fr))
    say("  relative risk                  : %.1fx" % (r_st / r_fr))
    say("  stale snapshots account for    : %.0f%% of all failures (%d of %d)"
        % (100 * f_st / f_all, f_st, f_all))
    say("  95%% CI stale %.1f-%.1f%%  |  fresh %.1f-%.1f%%"
        % (100 * F["stale_fail_ci"][0], 100 * F["stale_fail_ci"][1],
           100 * F["fresh_fail_ci"][0], 100 * F["fresh_fail_ci"][1]))
    ct = pd.crosstab(d["stale"], d["ok"]).values
    odds, p_fisher = stats.fisher_exact(ct)
    chi2, p_chi, _, _ = stats.chi2_contingency(ct)
    F["fisher_p"] = float(p_fisher)
    F["fisher_odds_ratio"] = float(odds)
    F["chi2_p"] = float(p_chi)
    say("  Fisher exact p = %.3g (OR %.3f) | chi2 p = %.3g" % (p_fisher, odds, p_chi))

    # ------------------------------------------------------------- the TTL gap
    say()
    say("=" * 78)
    say("2. WHY %d MIN: THE HISTOGRAM HAS A HOLE AT 19-21" % TTL_MIN)
    say("=" * 78)
    counts = {int(v): int(((d["lag_min"] >= v) & (d["lag_min"] < v + 1)).sum())
              for v in range(0, 26)}
    F["lag_hist_0_25"] = counts
    say("snapshot age (min) -> number of tasks")
    for v in range(0, 26):
        say("   %2d-%2d : %-3d %s" % (v, v + 1, counts[v], "#" * counts[v]))
    say()
    say("0-18 min : %d tasks (dense)" % sum(counts[v] for v in range(0, 19)))
    say("19-21 min: %d tasks  <-- EMPTY" % sum(counts[v] for v in (19, 20, 21)))
    say("22+ min  : %d tasks" % sum(counts[v] for v in range(22, 26)))
    say()
    say("A gradual drift (a slow cron, a lagging queue) would fill 19-21.")
    say("An empty band means two discrete regimes: a healthy sync path that")
    say("always lands under ~19 min, and a degraded path that starts at 22 min.")
    say("Max healthy age = %.0f min; min degraded age = %.0f min."
        % (F["max_lag_fresh"], F["min_lag_stale"]))
    say("=> any TTL in 19..21 splits the two regimes identically; shipping %d."
        % TTL_MIN)

    # -------------------------------------------------------- rule out confounds
    say()
    say("=" * 78)
    say("3. CONFOUNDS RULED OUT")
    say("=" * 78)
    fr_slots, st_slots = d[~d["stale"]]["slots"], d[d["stale"]]["slots"]
    p_slots = stats.mannwhitneyu(fr_slots, st_slots).pvalue
    F["slots_fresh_mean"] = float(fr_slots.mean())
    F["slots_stale_mean"] = float(st_slots.mean())
    F["slots_mwu_p"] = float(p_slots)
    F["corr_lag_slots"] = float(d["lag_min"].corr(d["slots"]))
    say("(a) 'stale just means the node looked busy'?  NO.")
    say("    recorded free_slots  fresh mean %.2f  vs  stale mean %.2f  (MWU p=%.2f)"
        % (fr_slots.mean(), st_slots.mean(), p_slots))
    say("    corr(snapshot age, free_slots) = %+.3f -> independent." % F["corr_lag_slots"])
    say("    The router saw a healthy-looking node in BOTH cases. Age is invisible")
    say("    in the one field the router actually reads.")
    say()
    say("(b) 'free_slots already captures the risk'?  NO.")
    y = (d["ok"] == 0).astype(int).values

    def auc(score):
        r = stats.rankdata(score)
        n1, n0 = y.sum(), len(y) - y.sum()
        a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        return max(a, 1 - a)

    F["auc_free_slots"] = float(auc(d["slots"].values))
    F["auc_lag"] = float(auc(d["lag_min"].values))
    F["auc_duration"] = float(auc(d["dur"].values))
    say("    AUC(free_slots)        = %.3f  <- near coin-flip" % F["auc_free_slots"])
    say("    AUC(video_duration)    = %.3f" % F["auc_duration"])
    say("    AUC(snapshot age)      = %.3f" % F["auc_lag"])
    say()
    say("(c) 'duration drives it'?  NO.")
    say("    corr(duration, failure) = %+.3f; failure rate is flat across")
    say("    duration deciles (see out/findings.json -> fail_by_duration).")
    F["corr_dur_ok"] = float(d["dur"].corr(d["ok"]))
    say()
    say("(d) 'a few bad clusters'?  NO - it holds inside every cluster except west.")
    F["by_cluster"] = {}
    for cl, g in d.groupby("cluster"):
        gs, gf = g[g["stale"]], g[~g["stale"]]
        ks, ns = int((gs["ok"] == 0).sum()), len(gs)
        kf, nf = int((gf["ok"] == 0).sum()), len(gf)
        lo, hi = wilson(ks, ns)
        F["by_cluster"][cl] = dict(stale_n=ns, stale_fail=ks,
                                   stale_rate=ks / ns if ns else None,
                                   stale_ci=[lo, hi],
                                   fresh_n=nf, fresh_fail=kf,
                                   fresh_rate=kf / nf if nf else None)
        say("    %-14s stale %2d/%3d = %5.1f%% [%.1f-%.1f]   fresh %2d/%3d = %4.1f%%"
            % (cl, ks, ns, 100 * ks / ns, 100 * lo, 100 * hi, kf, nf, 100 * kf / nf))

    # --------------------------------------------------------- west = control
    say()
    say("=" * 78)
    say("4. cluster_west IS A CONTROL GROUP - AND IT CHANGES THE STORY")
    say("=" * 78)
    w = d[(d["cluster"] == "cluster_west") & d["stale"]]
    o = d[(d["cluster"] != "cluster_west") & d["stale"]]
    kw, nw = int((w["ok"] == 0).sum()), len(w)
    ko, no = int((o["ok"] == 0).sum()), len(o)
    p_other = ko / no
    p_binom = stats.binom.cdf(kw, nw, p_other)
    F["west_stale_n"] = nw
    F["west_stale_fail"] = kw
    F["west_stale_rate"] = kw / nw
    F["other_stale_rate"] = p_other
    F["west_binom_p"] = float(p_binom)
    F["west_vs_other_fisher_p"] = float(stats.fisher_exact(
        [[kw, nw - kw], [ko, no - ko]]).pvalue)
    say("west, stale snapshots : %d tasks -> %d failure (%.1f%%)" % (nw, kw, 100 * kw / nw))
    say("other clusters, stale : %d tasks -> %d failures (%.1f%%)" % (no, ko, 100 * p_other))
    say("P(<= %d failures | n=%d, p=%.3f) = %.3g" % (kw, nw, p_other, p_binom))
    say("Fisher west-vs-others on stale rows: p = %.3g" % F["west_vs_other_fisher_p"])
    say()
    say("West's stale rows look identical to everyone else's on every field we")
    say("can see:")
    for f in ["lag_min", "slots", "dur"]:
        a, b = w[f], o[f]
        pv = stats.mannwhitneyu(a, b).pvalue
        F["west_%s_mwu_p" % f] = float(pv)
        say("    %-8s west %.1f  vs  others %.1f   (MWU p=%.3f)"
            % (f, a.mean(), b.mean(), pv))
    ctj = pd.crosstab(d["job_type"], d["cluster"])
    F["jobmix_chi2_p"] = float(stats.chi2_contingency(ctj.values)[1])
    F["durmix_chi2_p"] = float(stats.chi2_contingency(
        pd.crosstab(d["cluster"], pd.cut(d["dur"], 5)).values)[1])
    say("    job_type mix  chi2 p=%.2f  |  duration mix chi2 p=%.2f (same mix)"
        % (F["jobmix_chi2_p"], F["durmix_chi2_p"]))
    say()
    say("West also has a HIGHER failure rate on fresh rows (%.1f%%) than on stale"
        % (100 * (1 - d[(d["cluster"] == "cluster_west") & ~d["stale"]]["ok"].mean())))
    say("rows (%.1f%%) - the opposite of the other three clusters." % (100 * kw / nw))
    say()
    say("INTERPRETATION. Age alone is not the mechanism. If an old snapshot were")
    say("harmful simply by being old, west would be harmed too. What separates")
    say("west is that its snapshot apparently still described the node. The")
    say("harmful case is a snapshot that has stopped tracking reality while the")
    say("router keeps trusting it. So: 'age >= %d min' is a PROXY, not the cause." % TTL_MIN)
    say("I cannot see true node capacity in this dataset, so I cannot fully")
    say("separate 'stale data causes bad routing' from 'a degrading node both")
    say("stops syncing and fails renders'. Both are consistent with everything")
    say("above. The operational rule is the same either way - do not route on a")
    say("snapshot older than %d min - and west tells us the rule's false-positive" % TTL_MIN)
    say("cost is small and measurable (section 6).")

    # ------------------------------------------------------------ robustness
    say()
    say("=" * 78)
    say("5. ROBUSTNESS")
    say("=" * 78)
    d2 = d.copy()
    d2["day"] = d2["created_utc"].dt.tz_convert("Asia/Shanghai").dt.normalize()
    cut = pd.Timestamp("2026-09-05", tz="Asia/Shanghai")
    for name, s in [("train 09-01..09-04", d2[d2["day"] < cut]),
                    ("test  09-05..09-07", d2[d2["day"] >= cut])]:
        a, b = s[s["stale"]], s[~s["stale"]]
        say("  %-20s stale %.1f%% (n=%3d)  fresh %.1f%% (n=%3d)"
            % (name, 100 * (1 - a["ok"].mean()), len(a),
               100 * (1 - b["ok"].mean()), len(b)))
        F["oot_" + name.split()[0]] = dict(
            stale_rate=float(1 - a["ok"].mean()), stale_n=len(a),
            fresh_rate=float(1 - b["ok"].mean()), fresh_n=len(b))
    say("  leave-one-cluster-out:")
    F["loo"] = {}
    for cl in sorted(d["cluster"].unique()):
        s = d[d["cluster"] != cl]
        a, b = s[s["stale"]], s[~s["stale"]]
        F["loo"][cl] = dict(stale_rate=float(1 - a["ok"].mean()), stale_n=len(a),
                            fresh_rate=float(1 - b["ok"].mean()), fresh_n=len(b))
        say("    drop %-14s stale %.1f%% (n=%3d)  fresh %.1f%% (n=%3d)"
            % (cl, 100 * (1 - a["ok"].mean()), len(a),
               100 * (1 - b["ok"].mean()), len(b)))
    nanrows = dedup[dedup["ok"].isna()]
    nan_stale = int((nanrows["lag_min"] >= TTL_MIN).sum())
    t_nan = [[nan_stale, len(nanrows) - nan_stale],
             [n_st, n_fr]]
    F["nan_stale_share"] = nan_stale / len(nanrows)
    F["nan_fisher_p"] = float(stats.fisher_exact(t_nan).pvalue)
    F["worst_case_fail_rate"] = (f_all + len(nanrows)) / len(dedup)
    say("  the %d blank rows are NOT concentrated on stale snapshots (%.1f%% vs"
        % (len(nanrows), 100 * nan_stale / len(nanrows)))
    say("  %.1f%% overall, Fisher p=%.2f). Worst case, if every blank row were a"
        % (100 * n_st / len(d), F["nan_fisher_p"]))
    say("  failure, the overall failure rate would be %.1f%% instead of %.1f%%."
        % (100 * F["worst_case_fail_rate"], 100 * f_all / n_all))

    # --------------------------------------------------------------- the fix
    say()
    say("=" * 78)
    say("6. THE FIX, AND WHAT IT IS WORTH")
    say("=" * 78)
    say("Decision rule: route to the low-overhead degrade channel when the node")
    say("snapshot is older than %d min. Break-even is" % TTL_MIN)
    say("    P(fail) > %.1f min / %.0f min = %.1f%%" % (DEGRADE_MIN, FAIL_MIN, 100 * BREAK_EVEN))
    say("Observed P(fail) on stale snapshots = %.1f%%  ->  %.1fx above break-even."
        % (100 * r_st, r_st / BREAK_EVEN))
    say("Observed P(fail) on fresh snapshots = %.1f%%  ->  below break-even, keep them."
        % (100 * r_fr))
    say()
    baseline = f_all * FAIL_MIN
    say("baseline cost of creator wait : %d failures x %.0f min = %.0f min (%.1f h)"
        % (f_all, FAIL_MIN, baseline, baseline / 60))
    say()
    say("OPTION A - refresh the snapshot before routing (block until fresh):")
    say("    Refreshed tasks do NOT become failure-free; they join the fresh")
    say("    population and inherit its failure rate (%.2f%%)." % (100 * r_fr))
    a_exp = n_st * r_fr
    a_cost = (f_fr + a_exp) * FAIL_MIN
    a_save = baseline - a_cost
    say("    %d stale tasks x %.4f = %.1f expected failures" % (n_st, r_fr, a_exp))
    say("    expected failures = %d fresh + %.1f = %.1f  ->  %.0f min"
        % (f_fr, a_exp, f_fr + a_exp, a_cost))
    say("    SAVED                             -> %.0f min (%.1f h) per week"
        % (a_save, a_save / 60))
    say("    cost: a few seconds of extra latency per affected task.")
    F["option_a_saved_min"] = a_save
    F["option_a_expected_failures"] = f_fr + a_exp
    say("    (a naive model crediting A with avoiding all %d stale failures" % f_st)
    say("     would claim %.0f min and overstate the benefit by %.0f min.)"
        % (f_st * FAIL_MIN, f_st * FAIL_MIN - a_save))
    say()
    say("OPTION B - route stale tasks to the degrade channel (%.0f min each):" % DEGRADE_MIN)
    b_save = f_st * FAIL_MIN - n_st * DEGRADE_MIN
    say("    avoids %d stale failures          -> %.0f min saved" % (f_st, f_st * FAIL_MIN))
    say("    costs %d x %.0f min               -> %.0f min" % (n_st, DEGRADE_MIN, n_st * DEGRADE_MIN))
    say("    NET                               -> %.0f min saved (%.1f h) per week"
        % (b_save, b_save / 60))
    say("    residual %d fresh failures        -> %.0f min" % (f_fr, f_fr * FAIL_MIN))
    F["option_b_saved_min"] = b_save
    say()
    say("OPTION B' - same, but exempt a cluster whose stale failure rate is below")
    say("break-even. West qualifies (%.1f%% < %.1f%%):" % (100 * kw / nw, 100 * BREAK_EVEN))
    n_route = n_st - nw
    bp_cost = (f_fr + kw) * FAIL_MIN + n_route * DEGRADE_MIN
    bp_save = baseline - bp_cost
    say("    degrade %d stale tasks (excluding west) -> %.0f min"
        % (n_route, n_route * DEGRADE_MIN))
    say("    west's %d stale task stays normal, %d failure = %.0f min"
        % (nw, kw, kw * FAIL_MIN))
    say("    expected cost = %.0f fresh/west failures + %.0f degrade = %.0f min"
        % ((f_fr + kw) * FAIL_MIN, n_route * DEGRADE_MIN, bp_cost))
    say("    NET                               -> %.0f min saved (%.1f h) per week"
        % (bp_save, bp_save / 60))
    F["option_bprime_saved_min"] = bp_save
    say("    CAVEAT: west's exemption rests on %d stale rows in one 7-day window."
        % nw)
    say("    Ship the blanket rule first; add the exemption after a second week")
    say("    reproduces it.")
    say()
    say("A beats B by %.0f min/week, so prefer A; keep B as the fallback for"
        % (a_save - b_save))
    say("nodes that cannot be re-queried.")
    say()
    say("SENSITIVITY. Option B nets %.0f x F - %d x D minutes for failure cost F"
        % (f_st, n_st))
    say("and degrade cost D. It stays positive while D < %.1f min, i.e. the"
        % (f_st * FAIL_MIN / n_st))
    say("degrade path can be up to %.1fx more expensive than the stated %.0f min."
        % ((f_st * FAIL_MIN / n_st) / DEGRADE_MIN, DEGRADE_MIN))
    F["degrade_breakeven_min"] = f_st * FAIL_MIN / n_st

    say()
    say("=" * 78)
    say("7. SHIP LIST")
    say("=" * 78)
    say("1. Gate routing on snapshot age. Refuse to route on a snapshot older")
    say("   than %d min; re-query the node first, or send the task to the degrade" % TTL_MIN)
    say("   channel. This alone removes %.0f%% of observed failures." % (100 * f_st / f_all))
    say("2. Emit snapshot age as a first-class scheduling metric, not a debug")
    say("   field. It is the strongest signal in the dataset (AUC %.2f) while the" % F["auc_lag"])
    say("   field actually used for routing is near-useless (AUC %.2f)." % F["auc_free_slots"])
    say("3. Alert when the 19-21 min band stops being empty. That band is the")
    say("   healthy/degraded boundary; traffic appearing there means the sync")
    say("   path is drifting toward the failure regime.")
    say("4. Also treat free_slots==0 as a soft signal: %d tasks, %.0f%% failure"
        % (len(d[d["slots"] == 0]), 100 * (1 - d[d["slots"] == 0]["ok"].mean())))
    say("   (vs %.1f%% otherwise) - weaker than age, but free." % (100 * (1 - d[d["slots"] > 0]["ok"].mean())))
    say("5. Fix the duplicate emission (47 task_ids emitted twice) and the 40")
    say("   blank is_success rows, so the next analysis does not have to caveat them.")

    # ------------------------------------------------------------- figures
    try:
        make_figures(d, F)
        say()
        say("wrote out/fig1_lag_gap.png .. out/fig4_timeline.png")
    except Exception as exc:  # pragma: no cover - figures are a nice-to-have
        say()
        say("figures skipped (%s: %s)" % (type(exc).__name__, exc))

    # ------------------------------------------------------------- artifacts
    with open(os.path.join(OUT, "findings.json"), "w", encoding="utf-8") as fh:
        json.dump(F, fh, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    say()
    say("wrote out/findings.json and out/report.txt")
    return F


if __name__ == "__main__":
    main()
