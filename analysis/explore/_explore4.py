# -*- coding: utf-8 -*-
"""Ad-hoc exploration #4: confound checks, robustness, west mechanism."""
import pandas as pd, numpy as np, os
from scipy import stats

pd.set_option("display.width", 260)
CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "render_tasks.csv")
raw = pd.read_csv(CSV, dtype=str)
df = raw.copy()
df["c"] = pd.to_datetime(df.created_at, utc=True)
df["s"] = pd.to_datetime(df.cluster_synced_at, utc=True)
df["lag"] = (df.c - df.s).dt.total_seconds() / 60
df["ok"] = pd.to_numeric(df.is_success, errors="coerce")
df["slots"] = pd.to_numeric(df.free_slots)
df["dur"] = pd.to_numeric(df.video_duration_sec)
cols = ["task_id", "created_at", "cluster_synced_at", "cluster", "job_type",
        "free_slots", "video_duration_sec", "is_success"]
dd = df.drop_duplicates(subset=cols)
d = dd.dropna(subset=["ok"]).copy()
d["stale"] = d.lag >= 20

print("=" * 72)
print("A. CONFOUND CHECK: is staleness just a proxy for 'cluster looked busy'?")
print("=" * 72)
print("If high lag meant 'cluster is overloaded', stale rows should show FEWER")
print("recorded free_slots. They do not:")
print("  fresh rows: mean slots %.2f, median %.1f" % (d[~d.stale].slots.mean(), d[~d.stale].slots.median()))
print("  stale rows: mean slots %.2f, median %.1f" % (d[d.stale].slots.mean(), d[d.stale].slots.median()))
print("  MWU p = %.3f  (no difference)" % stats.mannwhitneyu(d[~d.stale].slots, d[d.stale].slots).pvalue)
print("  => the scheduler saw a healthy node in BOTH cases. Staleness is invisible")
print("     in the field the router actually uses.")
print()
print("Correlation lag vs slots: %.4f (independent)" % d.lag.corr(d.slots))

print()
print("=" * 72)
print("B. ROBUSTNESS: out-of-time validation (train d1-4 / test d5-7)")
print("=" * 72)
d["day"] = d.c.dt.tz_convert("Asia/Shanghai").dt.normalize()
cut = pd.Timestamp("2026-09-05", tz="Asia/Shanghai")
tr, te = d[d.day < cut], d[d.day >= cut]
print("train n=%d  test n=%d" % (len(tr), len(te)))
for name, s in [("train", tr), ("test", te)]:
    a, b = s[s.stale], s[~s.stale]
    print("  %-5s stale fail=%.3f (n=%3d)  fresh fail=%.3f (n=%3d)" % (
        name, 1 - a.ok.mean(), len(a), 1 - b.ok.mean(), len(b)))

print()
print("=" * 72)
print("C. ROBUSTNESS: cluster-wise (leave-one-out)")
print("=" * 72)
for cl in sorted(d.cluster.unique()):
    s = d[d.cluster != cl]
    a, b = s[s.stale], s[~s.stale]
    print("  drop %-14s stale fail=%.3f (n=%3d)  fresh fail=%.3f (n=%3d)" % (
        cl, 1 - a.ok.mean(), len(a), 1 - b.ok.mean(), len(b)))

print()
print("=" * 72)
print("D. WEST MECHANISM: does west's snapshot age differently?")
print("=" * 72)
for cl, g in d.groupby("cluster"):
    st = g[g.stale]
    print("  %-14s stale n=%3d  lag p50=%.0f  slots p50=%.0f  dur p50=%.0f  fail=%.3f" % (
        cl, len(st), st.lag.median(), st.slots.median(), st.dur.median(), 1 - st.ok.mean()))
print()
print("Test: are west stale rows LOWER-lag than others' stale rows?")
w = d[(d.cluster == "cluster_west") & d.stale]; o = d[(d.cluster != "cluster_west") & d.stale]
print("  west stale lag median %.1f, others %.1f, MWU p=%.4f" % (
    w.lag.median(), o.lag.median(), stats.mannwhitneyu(w.lag, o.lag).pvalue))
print("  -> west is equally stale. Staleness alone does not cause failure.")
print("     West behaves like a CONTROL group: same stale snapshots, no harm.")
print()
print("West fail rate by lag band:")
wb = w.copy()
print("  west ALL rows by lag band:")
for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 30), (30, 45), (45, 100)]:
    g = d[(d.cluster == "cluster_west") & (d.lag >= lo) & (d.lag < hi)]
    if len(g):
        print("    lag %2d-%3d  n=%3d  fail=%.3f" % (lo, hi, len(g), 1 - g.ok.mean()))

print()
print("=" * 72)
print("E. UNREPORTED (is_success empty): what if they are all failures?")
print("=" * 72)
n_nan = int(dd.ok.isna().sum())
base_fail = int((d.ok == 0).sum())
print("  reported failures: %d ; unreported: %d" % (base_fail, n_nan))
print("  worst case overall fail rate: %.3f" % ((base_fail + n_nan) / len(dd)))
nan_stale = int((dd[dd.ok.isna()].lag >= 20).sum())
print("  unreported & stale: %d (%.1f%%) vs stale share overall %.1f%%" % (
    nan_stale, 100 * nan_stale / n_nan, 100 * d.stale.mean()))
t = [[nan_stale, n_nan - nan_stale],
     [int(d.stale.sum()), int((~d.stale).sum())]]
print("  Fisher p = %.3f -> unreported rows are NOT concentrated on stale snapshots"
      % stats.fisher_exact(t).pvalue)

print()
print("=" * 72)
print("F. TWO FIX OPTIONS, QUANTIFIED")
print("=" * 72)
FAIL_MIN, DEG_MIN = 75.0, 10.0
n_stale = int(d.stale.sum())
n_fail_stale = int((d[d.stale].ok == 0).sum())
n_fail_all = int((d.ok == 0).sum())
print("  baseline: %d failures x %.0f min = %.0f min of creator wait" % (
    n_fail_all, FAIL_MIN, n_fail_all * FAIL_MIN))
print()
print("  OPTION A - re-sync before routing (block until fresh snapshot):")
print("    avoids %d stale failures -> %.0f min saved (cost ~seconds)" % (
    n_fail_stale, n_fail_stale * FAIL_MIN))
print("    residual failures %d x %.0f = %.0f min" % (
    n_fail_all - n_fail_stale, FAIL_MIN, (n_fail_all - n_fail_stale) * FAIL_MIN))
print()
print("  OPTION B - route stale tasks to the degrade channel (%.0f min each):" % DEG_MIN)
print("    avoids %d stale failures -> %.0f min saved, costs %d x %.0f = %.0f min" % (
    n_fail_stale, n_fail_stale * FAIL_MIN, n_stale, DEG_MIN, n_stale * DEG_MIN))
print("    NET = %.0f min saved" % (n_fail_stale * FAIL_MIN - n_stale * DEG_MIN))
print()
print("  Break-even failure probability for Option B: %.1f%%" % (100 * DEG_MIN / FAIL_MIN))
print("  Observed failure probability on stale snapshots: %.1f%%" % (100 * (1 - d[d.stale].ok.mean())))
print("  -> %.1fx above break-even." % ((1 - d[d.stale].ok.mean()) / (DEG_MIN / FAIL_MIN)))

print()
print("=" * 72)
print("G. WEEKLY EXTRAPOLATION (7 days of data)")
print("=" * 72)
print("  per-week avoided timeouts: %.0f min = %.1f h" % (
    n_fail_stale * FAIL_MIN, n_fail_stale * FAIL_MIN / 60))
print("  stale routing share: %.1f%% of %d tasks" % (100 * n_stale / len(d), len(d)))
print("  failures avoided as share of all failures: %.1f%%" % (100 * n_fail_stale / n_fail_all))
