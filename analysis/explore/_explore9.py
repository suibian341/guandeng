# -*- coding: utf-8 -*-
"""Ad-hoc exploration #9: is cluster_synced_at a FROZEN HEARTBEAT?

If a node's sync agent stops reporting, cluster_synced_at freezes at the last
successful report and lag grows monotonically. That produces:
  - a dense band of small lags (healthy reporting)
  - a gap just before the staleness threshold
  - long runs of stale rows sharing the SAME cluster_synced_at
"""
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
d = dd.dropna(subset=["ok"]).copy().sort_values("c").reset_index(drop=True)
d["stale"] = d.lag >= 20

print("=" * 74)
print("TEST A: how many DISTINCT cluster_synced_at values are there?")
print("=" * 74)
print("rows=%d  distinct synced_at=%d  distinct created_at=%d" % (
    len(d), d.s.nunique(), d.c.nunique()))
print("ratio rows/synced_at = %.2f  (if each sync served ~1 task, this is ~1)" % (
    len(d) / d.s.nunique()))
print()
print("Tasks sharing one synced_at value (i.e. one snapshot serving N tasks):")
vc = d.s.value_counts()
print(vc.value_counts().sort_index().rename("n_snapshots").to_frame().assign(
    tasks_served=lambda x: x.index * x.n_snapshots))
print()
print("Stale rows: do they cluster onto FEW synced_at values?")
for name, sub in [("FRESH", d[~d.stale]), ("STALE", d[d.stale])]:
    v = sub.s.value_counts()
    print("  %-6s rows=%3d  distinct_synced=%3d  max_tasks_per_snapshot=%d" % (
        name, len(sub), sub.s.nunique(), v.max()))

print()
print("=" * 74)
print("TEST B: within a stale RUN, is synced_at FROZEN (identical)?")
print("=" * 74)
for cl, g in d.groupby("cluster"):
    g = g.sort_values("c").reset_index(drop=True)
    runs = []
    cur = []
    for _, r in g.iterrows():
        if r.stale:
            cur.append(r)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    frozen = sum(1 for r in runs if len(set(x.s for x in r)) == 1)
    multi = [r for r in runs if len(r) >= 3]
    print("  %-14s runs=%2d  frozen(single synced_at)=%2d  runs>=3 rows=%d" % (
        cl, len(runs), frozen, len(multi)))
    for r in multi[:3]:
        span = (max(x.c for x in r) - min(x.c for x in r)).total_seconds() / 60
        print("      run of %2d tasks over %5.1f min, %d distinct synced_at, fail=%d/%d" % (
            len(r), span, len(set(x.s for x in r)), sum(1 for x in r if x.ok == 0), len(r)))

print()
print("=" * 74)
print("TEST C: the 18-22 gap - evidence of a threshold, not drift")
print("=" * 74)
print("lag histogram (1-min bins, 0..24):")
for lo in range(0, 25):
    n = int(((d.lag >= lo) & (d.lag < lo + 1)).sum())
    print("   %2d-%2d : %-3d %s" % (lo, lo + 1, n, "#" * n))
print()
print("=> dense 0-18, EMPTY 19-21, resumes at 22. A gradual drift process")
print("   would populate 19-21. This is a hard cutover between two regimes.")

print()
print("=" * 74)
print("TEST D: healthy vs stuck - per cluster share of time stuck")
print("=" * 74)
d["day"] = d.c.dt.tz_convert("Asia/Shanghai").dt.date
print(d.pivot_table(index="cluster", columns="day", values="stale", aggfunc="mean").round(2).to_string())

print()
print("=" * 74)
print("TEST E: does a node recover? (stale -> fresh transitions)")
print("=" * 74)
for cl, g in d.groupby("cluster"):
    g = g.sort_values("c").reset_index(drop=True)
    tr = 0
    for i in range(1, len(g)):
        if g.stale[i - 1] and not g.stale[i]:
            tr += 1
    print("  %-14s stale->fresh recoveries = %d" % (cl, tr))
print()
print("If nodes recover, staleness is an intermittent health episode.")
print("If synced_at NEVER advances again for a cluster, it is a dead agent.")
for cl, g in d.groupby("cluster"):
    g = g.sort_values("c")
    st = g[g.stale]
    if len(st):
        print("  %-14s stale synced_at range: %s .. %s" % (
            cl, st.s.min(), st.s.max()))
