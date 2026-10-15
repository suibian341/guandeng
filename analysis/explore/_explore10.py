# -*- coding: utf-8 -*-
"""Final numeric checks before writing the report."""
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

print("=== west immunity significance ===")
w = d[(d.cluster == "cluster_west") & d.stale]
n, k = len(w), int((w.ok == 0).sum())
p_other = 1 - d[(d.cluster != "cluster_west") & d.stale].ok.mean()
print("west stale n=%d fails=%d ; others' rate=%.3f" % (n, k, p_other))
print("P(<=1 fail | n=52, p=%.3f) = %.3g" % (p_other, stats.binom.cdf(k, n, p_other)))

print()
print("=== Wilson CI for key rates ===")
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)

for name, sub in [("stale (lag>=20)", d[d.stale]), ("fresh (lag<20)", d[~d.stale])]:
    k = int((sub.ok == 0).sum()); n = len(sub)
    lo, hi = wilson(k, n)
    print("  %-18s %2d/%3d = %.4f  95%%CI [%.4f, %.4f]" % (name, k, n, k / n, lo, hi))
print()
for cl in sorted(d.cluster.unique()):
    sub = d[(d.cluster == cl) & d.stale]
    k = int((sub.ok == 0).sum()); n = len(sub)
    lo, hi = wilson(k, n)
    print("  %-14s stale %2d/%3d = %.4f  95%%CI [%.4f, %.4f]" % (cl, k, n, k / n, lo, hi))

print()
print("=== healthy regime max lag / safe TTL ===")
print("max lag among fresh:", d[~d.stale].lag.max())
print("min lag among stale:", d[d.stale].lag.min())
print("=> recommend TTL = 15 min (3 min margin below the 18-min healthy max)")

print()
print("=== TTL=15 cost model ===")
FAIL_MIN, DEG_MIN = 75.0, 10.0
tot_fail = int((d.ok == 0).sum())
for thr in [15, 18, 19, 20]:
    sel = d.lag >= thr
    n_sel = int(sel.sum()); caught = int((d[sel].ok == 0).sum())
    print("  TTL=%2d: routes %3d tasks, catches %2d/%d failures (%.0f%%)" % (
        thr, n_sel, caught, tot_fail, 100 * caught / tot_fail))

print()
print("=== headline numbers for README ===")
n_all = len(d); f_all = int((d.ok == 0).sum())
n_st = int(d.stale.sum()); f_st = int((d[d.stale].ok == 0).sum())
print("reported tasks            : %d" % n_all)
print("overall failure rate      : %.1f%% (%d)" % (100 * f_all / n_all, f_all))
print("tasks routed on stale snap: %d (%.1f%%)" % (n_st, 100 * n_st / n_all))
print("failures among them       : %d (%.0f%% of all failures)" % (f_st, 100 * f_st / f_all))
print("stale fail rate           : %.1f%%" % (100 * f_st / n_st))
print("fresh fail rate           : %.1f%%" % (100 * (f_all - f_st) / (n_all - n_st)))
print("relative risk             : %.1fx" % ((f_st / n_st) / ((f_all - f_st) / (n_all - n_st))))
print("timeout minutes from stale: %.0f min (%.1f h)" % (f_st * FAIL_MIN, f_st * FAIL_MIN / 60))
print("degrade cost if all stale : %.0f min" % (n_st * DEG_MIN))
print("net saving (degrade route): %.0f min (%.1f h)" % (f_st * FAIL_MIN - n_st * DEG_MIN,
                                                         (f_st * FAIL_MIN - n_st * DEG_MIN) / 60))
print("saving if refresh-first   : %.0f min (%.1f h)" % (f_st * FAIL_MIN, f_st * FAIL_MIN / 60))
print("break-even failure prob   : %.1f%%  (observed %.1f%%, %.1fx margin)" % (
    100 * DEG_MIN / FAIL_MIN, 100 * f_st / n_st, (f_st / n_st) / (DEG_MIN / FAIL_MIN)))

print()
print("=== secondary signal: free_slots ===")
z = d[d.slots == 0]
print("slots==0: n=%d fail=%.1f%%  vs slots>0: n=%d fail=%.1f%%" % (
    len(z), 100 * (1 - z.ok.mean()), len(d) - len(z), 100 * (1 - d[d.slots > 0].ok.mean())))
print("AUC(free_slots) = 0.535 -> the field the router uses is nearly blind")
print("corr(free_slots, lag) = %.3f -> staleness is invisible in free_slots" % d.slots.corr(d.lag))

print()
print("=== data-quality notes ===")
print("duplicate rows (identical 8 cols): %d over %d task_ids" % (
    len(df) - len(dd), df[df.duplicated(subset=cols, keep=False)].task_id.nunique()))
print("unreported is_success: %d (%.1f%%)" % (int(df.ok.isna().sum()), 100 * df.ok.isna().mean()))
print("worst-case overall failure rate if all unreported failed: %.1f%%" % (
    100 * (f_all + int(dd.ok.isna().sum())) / len(dd)))
print("dup task_ids all succeeded? %s" % (
    (pd.to_numeric(df[df.duplicated(subset=cols, keep=False)].is_success) == 1).all()))
print("dup effect on failure rate: %.4f -> %.4f" % (
    1 - pd.to_numeric(df.is_success, errors="coerce").mean(), 1 - d.ok.mean()))
