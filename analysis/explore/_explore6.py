# -*- coding: utf-8 -*-
"""Ad-hoc exploration #6: does the stale snapshot still tell the truth?
Idea: for a stale row, compare its recorded free_slots with the free_slots
recorded by FRESH rows of the same cluster shortly before/after. If the
stale snapshot's value still matches, the snapshot is old but accurate.
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
print("A. SNAPSHOT FIDELITY: stale row's slots vs nearby fresh rows' slots")
print("=" * 74)
print("For each STALE row, take fresh rows of the same cluster within +-60 min")
print("and compare the recorded free_slots. Large divergence = the stale")
print("snapshot was lying about node capacity.")
print()
rows = []
for cl, g in d.groupby("cluster"):
    g = g.sort_values("c")
    fresh = g[~g.stale]
    for _, r in g[g.stale].iterrows():
        near = fresh[(fresh.c >= r.c - pd.Timedelta("60min")) &
                     (fresh.c <= r.c + pd.Timedelta("60min"))]
        if len(near) == 0:
            continue
        rows.append({
            "cluster": cl, "task_id": r.task_id, "lag": r.lag,
            "slots_snapshot": r.slots, "slots_nearby": near.slots.mean(),
            "n_nearby": len(near), "ok": r.ok,
            "gap": (r.slots - near.slots.mean()),
            "absgap": abs(r.slots - near.slots.mean()),
        })
fid = pd.DataFrame(rows)
print("matched stale rows: %d" % len(fid))
print()
print("Per cluster: mean |snapshot slots - nearby fresh slots|")
print(fid.groupby("cluster").agg(
    n=("task_id", "size"),
    mean_abs_gap=("absgap", "mean"),
    median_abs_gap=("absgap", "median"),
    fail=("ok", lambda s: 1 - s.mean()),
    mean_lag=("lag", "mean")).round(3))
print()
print("Test: is the gap LARGER for the failing (non-west) clusters?")
for cl in sorted(fid.cluster.unique()):
    g = fid[fid.cluster == cl]
    print("  %-14s mean_abs_gap=%.2f  fail=%.3f" % (cl, g.absgap.mean(), 1 - g.ok.mean()))

print()
print("=" * 74)
print("B. DOES THE GAP PREDICT FAILURE? (within stale rows, non-west)")
print("=" * 74)
nw = fid[fid.cluster != "cluster_west"].copy()
print("n =", len(nw), " fail =", round(1 - nw.ok.mean(), 3))
nw["gb"] = pd.cut(nw.absgap, [-0.01, 2, 5, 10, 40])
print(nw.groupby("gb", observed=True)["ok"].agg(["count", "mean"]).assign(
    fail=lambda x: 1 - x["mean"]))
print()
print("corr(|gap|, fail) on non-west stale rows: %.3f" % nw.absgap.corr(1 - nw.ok))
print("corr(lag,   fail) on non-west stale rows: %.3f" % nw.lag.corr(1 - nw.ok))
print()
print("West:")
ww = fid[fid.cluster == "cluster_west"]
print("  n=%d mean_abs_gap=%.2f fail=%.3f" % (len(ww), ww.absgap.mean(), 1 - ww.ok.mean()))
print("  corr(|gap|, fail) = %.3f" % ww.absgap.corr(1 - ww.ok))

print()
print("=" * 74)
print("C. SANITY: same metric on FRESH rows (should be near zero)")
print("=" * 74)
rows2 = []
for cl, g in d.groupby("cluster"):
    g = g.sort_values("c")
    fresh = g[~g.stale]
    for _, r in g[~g.stale].iterrows():
        near = fresh[(fresh.task_id != r.task_id) &
                     (fresh.c >= r.c - pd.Timedelta("60min")) &
                     (fresh.c <= r.c + pd.Timedelta("60min"))]
        if len(near) == 0:
            continue
        rows2.append({"cluster": cl, "absgap": abs(r.slots - near.slots.mean()), "ok": r.ok})
f2 = pd.DataFrame(rows2)
print("fresh matched rows: %d, mean |gap| = %.2f" % (len(f2), f2.absgap.mean()))
print("stale matched rows: %d, mean |gap| = %.2f" % (len(fid), fid.absgap.mean()))
print("MWU p = %.4g" % stats.mannwhitneyu(fid.absgap, f2.absgap).pvalue)
print()
print("per-cluster mean |gap|, FRESH vs STALE:")
cmp = pd.DataFrame({
    "fresh": f2.groupby("cluster").absgap.mean(),
    "stale": fid.groupby("cluster").absgap.mean(),
})
cmp["delta"] = cmp.stale - cmp.fresh
print(cmp.round(2))

print()
print("=" * 74)
print("D. FREE SLOTS TRAJECTORY AROUND A STALE EVENT (west vs north)")
print("=" * 74)
for cl in ["cluster_west", "cluster_north"]:
    g = d[d.cluster == cl].sort_values("c")
    st = g[g.stale]
    if len(st) == 0:
        continue
    pick = st.iloc[len(st) // 2]
    near = g[(g.c >= pick.c - pd.Timedelta("90min")) &
             (g.c <= pick.c + pd.Timedelta("90min"))]
    print("--", cl, "center task", pick.task_id, "lag", pick.lag, "slots", pick.slots)
    print(near[["task_id", "created_at", "slots", "lag", "ok"]].to_string(index=False))
    print()
