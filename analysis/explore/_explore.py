# -*- coding: utf-8 -*-
"""Ad-hoc exploration during RCA. Not part of the delivered pipeline."""
import pandas as pd, numpy as np, os

pd.set_option("display.width", 250)
CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "render_tasks.csv")

raw = pd.read_csv(CSV, dtype=str)
df = raw.copy()
df["c"] = pd.to_datetime(df.created_at, utc=True)
df["s"] = pd.to_datetime(df.cluster_synced_at, utc=True)
df["lag"] = (df.c - df.s).dt.total_seconds() / 60
df["ok"] = pd.to_numeric(df.is_success, errors="coerce")
df["slots"] = pd.to_numeric(df.free_slots)
df["dur"] = pd.to_numeric(df.video_duration_sec)
df["tid"] = pd.to_numeric(df.task_id)

cols = ["task_id", "created_at", "cluster_synced_at", "cluster", "job_type",
        "free_slots", "video_duration_sec", "is_success"]
dd = df.drop_duplicates(subset=cols)
d = dd.dropna(subset=["ok"]).copy()
d["stale"] = d.lag >= 20

print("=== FREE SLOTS distribution: stale rows, per cluster ===")
st = d[d.stale]
print(st.groupby("cluster")["slots"].describe()[["count", "mean", "50%", "min", "max"]])
print()
print("=== stale rows: slots bucket x cluster, fail rate ===")
st2 = st.copy(); st2["sb"] = pd.cut(st2.slots, [-1, 0, 5, 10, 15, 20, 25, 30])
print(st2.pivot_table(index="sb", columns="cluster", values="ok", aggfunc=["count", "mean"], observed=True))
print()
print("=== stale rows: duration bucket x cluster ===")
st3 = st.copy(); st3["db"] = pd.cut(st3.dur, [-1, 35, 50, 65, 100])
print(st3.pivot_table(index="db", columns="cluster", values="ok", aggfunc=["count", "mean"], observed=True))
print()
print("=== west stale: what is special? compare with east/north/south stale ===")
for cl in ["cluster_west", "cluster_east", "cluster_north", "cluster_south"]:
    g = st[st.cluster == cl]
    print("%-14s n=%3d slots_mean=%.1f dur_mean=%.1f lag_mean=%.1f fail=%.3f" % (
        cl, len(g), g.slots.mean(), g.dur.mean(), g.lag.mean(), 1 - g.ok.mean()))
print()
print("=== whole cluster profile (fresh+stale) ===")
for cl, g in d.groupby("cluster"):
    print("%-14s n=%3d slots_mean=%.1f dur_mean=%.1f lag_mean=%.1f fail=%.3f stale%%=%.2f" % (
        cl, len(g), g.slots.mean(), g.dur.mean(), g.lag.mean(), 1 - g.ok.mean(), g.stale.mean()))
print()
print("=== residual failures among FRESH rows ===")
f = d[~d.stale]
print("fresh failures: %d / %d" % ((f.ok == 0).sum(), len(f)))
print(f[f.ok == 0][["task_id", "cluster", "job_type", "slots", "dur", "lag"]].to_string())
print()
print("=== task_id continuity ===")
print("unique ids:", d.tid.nunique(), "min", d.tid.min(), "max", d.tid.max(),
      "missing:", (d.tid.max() - d.tid.min() + 1) - d.tid.nunique())
print(d.groupby("cluster")["tid"].agg(["min", "max", "count"]))
print()
print("=== per-day stale share and fail rate ===")
d2 = d.copy(); d2["day"] = d2.c.dt.tz_convert("Asia/Shanghai").dt.date
print(d2.groupby("day").apply(
    lambda g: pd.Series({"n": len(g), "stale_share": g.stale.mean(),
                         "fail": 1 - g.ok.mean(),
                         "fail_fresh": 1 - g[~g.stale].ok.mean(),
                         "fail_stale": 1 - g[g.stale].ok.mean() if g.stale.any() else np.nan}),
    include_groups=False))
