# -*- coding: utf-8 -*-
"""Ad-hoc exploration #8: common-cause framing.
Hypothesis H1 (data staleness causes failure): the ROUTER acts on an old
  snapshot and picks a node that is actually full -> render times out.
Hypothesis H2 (common cause): cluster_synced_at is a HEALTH HEARTBEAT.
  When a node degrades, its sync agent stops reporting AND its renders fail.
  Staleness is then a symptom/leading indicator, not the mechanism.

Distinguishing test: under H1, harm depends on the snapshot being WRONG
(load moved). Under H2, harm depends on the node being UNHEALTHY, which
should show up as temporal BURSTS of both staleness and failure.
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
print("TEST 1: Do failures BURST in time (common cause) or spread out?")
print("=" * 74)
for cl, g in d.groupby("cluster"):
    g = g.sort_values("c").reset_index(drop=True)
    fails = g.index[g.ok == 0].tolist()
    if len(fails) < 3:
        continue
    # gaps between successive failures, in position units
    gaps = np.diff(fails)
    # compare to random expectation
    exp = (len(g) - 1) / (len(fails) - 1)
    # count failures that are "adjacent" (within 3 positions of another failure)
    adj = sum(1 for i in range(len(fails)) if any(
        abs(fails[i] - fails[j]) <= 3 for j in range(len(fails)) if j != i))
    print("  %-14s fails=%2d  mean_gap=%.1f (random=%.1f)  adjacent(<=3)=%d/%d=%.2f" % (
        cl, len(fails), gaps.mean(), exp, adj, len(fails), adj / len(fails)))

print()
print("=" * 74)
print("TEST 2: Does the CLUSTER-WIDE failure rate move over time?")
print("=" * 74)
d["day"] = d.c.dt.tz_convert("Asia/Shanghai").dt.date
piv = d.pivot_table(index="day", columns="cluster", values="ok", aggfunc=["count", "mean"])
print("daily fail rate by cluster:")
fr = d.pivot_table(index="day", columns="cluster", values="ok", aggfunc="mean")
print((1 - fr).round(3).to_string())
print()
print("correlation of daily fail rates ACROSS clusters (common-cause => positive):")
c = (1 - fr).corr()
print(c.round(3).to_string())

print()
print("=" * 74)
print("TEST 3: Are stale events clustered in bursts per cluster?")
print("=" * 74)
print("If sync simply runs on a slow cron, staleness is spread evenly.")
print("If it is a health signal, staleness comes in bursts.")
for cl, g in d.groupby("cluster"):
    g = g.sort_values("c").reset_index(drop=True)
    st = g.index[g.stale].tolist()
    if len(st) < 3:
        continue
    gaps = np.diff(st)
    exp = (len(g) - 1) / (len(st) - 1)
    print("  %-14s stale=%2d  mean_gap=%.1f (random=%.1f)  max_run=%d" % (
        cl, len(st), gaps.mean(), exp,
        max((len(list(grp)) for _, grp in
             __import__("itertools").groupby(g.stale.values)), default=0)))

print()
print("=" * 74)
print("TEST 4: DECISIVE - does the snapshot being WRONG matter?")
print("=" * 74)
print("Under H1, what matters is (recorded slots) vs (true slots).")
print("We approximate 'true load' by how many tasks were actually routed to")
print("the cluster in the following 30 min (demand) and the recorded slots.")
print()
d2 = d.sort_values("c").reset_index(drop=True)
fut = []
for i, r in d2.iterrows():
    nxt = d2[(d2.cluster == r.cluster) &
             (d2.c > r.c) & (d2.c <= r.c + pd.Timedelta("30min"))]
    fut.append(len(nxt))
d2["future_demand"] = fut
for name, sub in [("FRESH", d2[~d2.stale]), ("STALE", d2[d2.stale])]:
    sub = sub.copy()
    sub["db"] = pd.qcut(sub.future_demand, 3, duplicates="drop")
    print("  %s future demand vs fail:" % name)
    print(sub.groupby("db", observed=True)["ok"].agg(["count", "mean"]).assign(
        fail=lambda x: 1 - x["mean"]).round(3).to_string())
    print()
print("If H1 were the mechanism, STALE rows with HIGH future demand should fail")
print("far more (the snapshot understated the rush).")

print()
print("=" * 74)
print("TEST 5: west vs others - failure rate of FRESH rows on the SAME days")
print("=" * 74)
for cl in sorted(d.cluster.unique()):
    g = d[d.cluster == cl]
    print("  %-14s fresh fail=%.3f  stale fail=%.3f" % (
        cl, 1 - g[~g.stale].ok.mean(), 1 - g[g.stale].ok.mean()))
print()
print("West fresh fail (%.3f) is HIGHER than its stale fail (%.3f) - the exact" % (
    1 - d[(d.cluster == "cluster_west") & ~d.stale].ok.mean(),
    1 - d[(d.cluster == "cluster_west") & d.stale].ok.mean()))
print("opposite of H1. On west, an old snapshot is not a risk at all.")
