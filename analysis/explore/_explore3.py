# -*- coding: utf-8 -*-
"""Ad-hoc exploration #3: mechanism, temporal clustering, west anomaly."""
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
print("A. IS THE LAG GAP REAL? (threshold discovery)")
print("=" * 72)
alllag = sorted(d.lag.unique())
print("sorted unique lag values:", alllag)
print()
print("max lag < 20 :", d[d.lag < 20].lag.max())
print("min lag >= 20:", d[d.lag >= 20].lag.min())
print("rows with 18 < lag < 22:", ((d.lag > 18) & (d.lag < 22)).sum())

print()
print("=" * 72)
print("B. THRESHOLD SWEEP: cost model")
print("=" * 72)
FAIL_MIN, DEG_MIN = 75.0, 10.0
n = len(d)
print("%6s %8s %8s %10s %12s %12s %12s" % (
    "thr", "n_route", "n_fail", "fail_rate", "cur_cost", "new_cost", "saving"))
best = None
for thr in range(5, 61):
    sel = d.lag >= thr
    ns = int(sel.sum())
    fails_caught = int((d[sel].ok == 0).sum())
    # current: every selected task keeps running on normal channel
    cur = int((d.ok == 0).sum()) * FAIL_MIN          # baseline: all failures
    # proposed: selected tasks go to degrade channel (10 min each), rest unchanged
    new = (int((d.ok == 0).sum()) - fails_caught) * FAIL_MIN + ns * DEG_MIN
    saving = cur - new
    if thr in (15, 18, 19, 20, 21, 22, 25, 30, 40):
        print("%6d %8d %8d %10.3f %12.0f %12.0f %12.0f" % (
            thr, ns, fails_caught, fails_caught / ns if ns else 0, cur, new, saving))
    if best is None or saving > best[1]:
        best = (thr, saving, ns, fails_caught)
print()
print("OPTIMAL threshold = %d min -> saving %.0f min, routes %d tasks, catches %d/%d failures" % (
    best[0], best[1], best[2], best[3], int((d.ok == 0).sum())))

print()
print("=" * 72)
print("C. TEMPORAL CLUSTERING OF STALE EVENTS (sync outage bursts?)")
print("=" * 72)
d2 = d.sort_values("c").copy()
for cl, g in d2.groupby("cluster"):
    g = g.sort_values("c")
    gaps = g.c.diff().dt.total_seconds() / 60
    stale_idx = g.stale.values
    # find runs of consecutive stale rows
    runs, cur = [], 0
    for v in stale_idx:
        if v:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    print("%-14s stale rows=%3d  runs=%2d  run lengths=%s" % (
        cl, stale_idx.sum(), len(runs), sorted(runs, reverse=True)[:10]))
print()
print("stale share by hour-of-day (all clusters):")
d2["h"] = d2.c.dt.tz_convert("Asia/Shanghai").dt.hour
print(d2.groupby("h")["stale"].agg(["count", "sum", "mean"]).T)

print()
print("=" * 72)
print("D. WEST: is its sync record qualitatively different?")
print("=" * 72)
w = d[d.cluster == "cluster_west"]; o = d[d.cluster != "cluster_west"]
print("west lag:   ", np.round(np.percentile(w.lag, [5, 25, 50, 75, 95]), 1))
print("others lag: ", np.round(np.percentile(o.lag, [5, 25, 50, 75, 95]), 1))
print("KS test west vs others lag: p=%.4f" % stats.ks_2samp(w.lag, o.lag).pvalue)
print()
print("west stale vs west fresh: fail %.4f (n=%d) vs %.4f (n=%d)" % (
    1 - w[w.stale].ok.mean(), w.stale.sum(), 1 - w[~w.stale].ok.mean(), (~w.stale).sum()))
print("=> west stale is NOT harmful. Others stale IS.")
print()
print("Check: does west's stale share vary over time (i.e. a fixed sync cadence)?")
w2 = w.copy(); w2["day"] = w2.c.dt.tz_convert("Asia/Shanghai").dt.date
print(w2.groupby("day")["stale"].agg(["count", "sum", "mean"]))

print()
print("=" * 72)
print("E. THE 40 UNREPORTED ROWS")
print("=" * 72)
u = dd[dd.ok.isna()]
print("n =", len(u))
print("their lag:", np.round(np.percentile(u.lag, [10, 50, 90]), 1))
print("their stale share: %.3f (vs %.3f in reported)" % ((u.lag >= 20).mean(), d.stale.mean()))
print("by cluster:"); print(u.groupby("cluster").size())
print("overlap with duplicate task_ids:",
      len(set(u.task_id.astype(int)) & set(dd[dd.duplicated("task_id", keep=False)].task_id.astype(int))))

print()
print("=" * 72)
print("F. DUPLICATES")
print("=" * 72)
dup = df[df.duplicated(subset=cols, keep=False)]
print("fully-identical duplicate rows:", len(dup), "distinct task_ids:", dup.task_id.nunique())
print("all duplicates have ok==1?", (pd.to_numeric(dup.is_success) == 1).all())
print("effect of dedup on overall fail rate: %.4f -> %.4f" % (
    1 - pd.to_numeric(df.is_success, errors="coerce").mean(),
    1 - pd.to_numeric(dd.is_success, errors="coerce").mean()))
