# -*- coding: utf-8 -*-
"""Ad-hoc exploration #7: decisive test of the west anomaly.
Hypothesis: an old snapshot only causes harm if the node's load actually
MOVED during the staleness window. Test: local volatility of free_slots
around each stale row, per cluster, vs failure.
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
print("LOCAL VOLATILITY AT STALE MOMENTS")
print("=" * 74)
rows = []
for cl, g in d.groupby("cluster"):
    g = g.sort_values("c")
    for _, r in g[g.stale].iterrows():
        near = g[(g.c >= r.c - pd.Timedelta("30min")) & (g.c <= r.c + pd.Timedelta("30min"))]
        rows.append({"cluster": cl, "task_id": r.task_id, "lag": r.lag, "ok": r.ok,
                     "vol": near.slots.std(), "n_near": len(near),
                     "rng": near.slots.max() - near.slots.min()})
v = pd.DataFrame(rows)
print(v.groupby("cluster").agg(n=("task_id", "size"), vol_mean=("vol", "mean"),
                               rng_mean=("rng", "mean"),
                               fail=("ok", lambda s: 1 - s.mean())).round(2))
print()
print("West vs others local volatility at stale moments:")
w = v[v.cluster == "cluster_west"]; o = v[v.cluster != "cluster_west"]
print("  west vol=%.2f  others vol=%.2f  MWU p=%.4f" % (
    w.vol.mean(), o.vol.mean(), stats.mannwhitneyu(w.vol, o.vol).pvalue))
print("  west range=%.2f others range=%.2f MWU p=%.4f" % (
    w.rng.mean(), o.rng.mean(), stats.mannwhitneyu(w.rng, o.rng).pvalue))

print()
print("=" * 74)
print("DOES VOLATILITY PREDICT FAILURE? (stale rows only, non-west)")
print("=" * 74)
o2 = o.dropna(subset=["vol"]).copy()
o2["vb"] = pd.qcut(o2.vol, 3, duplicates="drop")
print(o2.groupby("vb", observed=True)["ok"].agg(["count", "mean"]).assign(
    fail=lambda x: 1 - x["mean"]))
print("corr(vol, fail) = %.3f" % o2.vol.corr(1 - o2.ok))

print()
print("=" * 74)
print("STRUCTURAL: is west's lag generated differently?")
print("=" * 74)
for cl, g in d.groupby("cluster"):
    lag = g.lag
    print("  %-14s n=%3d  stale%%=%.2f  lag mod 5 hist=%s" % (
        cl, len(g), g.stale.mean(),
        np.bincount((lag.astype(int) % 5).values, minlength=5)))
print()
print("lag value counts, west vs others (values >= 20):")
wc = d[(d.cluster == "cluster_west") & d.stale].lag.value_counts().sort_index()
oc = d[(d.cluster != "cluster_west") & d.stale].lag.value_counts().sort_index()
print(pd.DataFrame({"west": wc, "others": oc}).fillna(0).astype(int).T.to_string())
