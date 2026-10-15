# -*- coding: utf-8 -*-
"""Ad-hoc exploration #5: WHY is west immune? Load-volatility hypothesis."""
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
print("A. FORMAL INTERACTION TEST: does the lag effect differ by cluster?")
print("=" * 74)
w = d[d.cluster == "cluster_west"]; o = d[d.cluster != "cluster_west"]
tab = np.array([
    [(w.stale & (w.ok == 0)).sum(), (w.stale & (w.ok == 1)).sum(),
     (~w.stale & (w.ok == 0)).sum(), (~w.stale & (w.ok == 1)).sum()],
    [(o.stale & (o.ok == 0)).sum(), (o.stale & (o.ok == 1)).sum(),
     (~o.stale & (o.ok == 0)).sum(), (~o.stale & (o.ok == 1)).sum()],
])
print("           stale_fail stale_ok fresh_fail fresh_ok")
print("west      %10d %8d %10d %8d" % tuple(tab[0]))
print("others    %10d %8d %10d %8d" % tuple(tab[1]))
orr_w = (tab[0][0] * tab[0][3]) / (tab[0][1] * tab[0][2] + 1e-9)
orr_o = (tab[1][0] * tab[1][3]) / (tab[1][1] * tab[1][2] + 1e-9)
print("odds ratio west   = %.3f" % orr_w)
print("odds ratio others = %.3f" % orr_o)
print("Breslow-Day-ish ratio = %.2f  (heterogeneity)" % (orr_o / (orr_w + 1e-9)))
print("Fisher (west stale vs others stale) p = %.3g" % stats.fisher_exact(
    [[tab[0][0], tab[0][1]], [tab[1][0], tab[1][1]]]).pvalue)

print()
print("=" * 74)
print("B. LOAD VOLATILITY per cluster (from free_slots series)")
print("=" * 74)
print("Hypothesis: an old snapshot only hurts if the node's load CHANGES.")
print("If a cluster's free_slots is stable, a 45-min-old snapshot is still true.")
print()
for cl, g in d.groupby("cluster"):
    g = g.sort_values("c")
    s = g.slots.values
    # local volatility: mean absolute successive change
    mac = np.abs(np.diff(s)).mean()
    print("  %-14s n=%3d  slots std=%5.2f  mean|螖slots|=%5.2f  range=%2d-%2d" % (
        cl, len(g), s.std(), mac, s.min(), s.max()))

print()
print("=" * 74)
print("C. DIRECT TEST: does the snapshot LIE? (slots at sync vs reality)")
print("=" * 74)
print("We cannot observe true capacity. But we CAN test whether a task failed")
print("more often when its cluster was BUSY at routing time, using the number")
print("of other tasks routed to the same cluster in the surrounding window as")
print("a load proxy.")
print()
for win in [10, 15, 30]:
    d["load"] = 0
    for cl, g in d.groupby("cluster"):
        idx = g.index.values
        times = g.c.values
        cnt = np.zeros(len(g), dtype=int)
        for i in range(len(g)):
            lo = times[i] - np.timedelta64(win, "m")
            hi = times[i] + np.timedelta64(win, "m")
            cnt[i] = ((times >= lo) & (times <= hi)).sum() - 1
        d.loc[idx, "load"] = cnt
    print("  window=+-%d min" % win)
    for name, sub in [("FRESH", d[~d.stale]), ("STALE", d[d.stale])]:
        q = pd.qcut(sub.load, 4, duplicates="drop")
        r = sub.groupby(q, observed=True)["ok"].agg(["count", "mean"])
        print("    %-6s %s" % (name, " | ".join(
            "Q%d n=%d fail=%.3f" % (i + 1, row["count"], 1 - row["mean"])
            for i, (_, row) in enumerate(r.iterrows()))))
    print()

print("=" * 74)
print("D. WEST vs OTHERS: load volatility comparison")
print("=" * 74)
for win in [15, 30]:
    print("  window=+-%d min" % win)
    for cl, g in d.groupby("cluster"):
        g = g.sort_values("c")
        times = g.c.values
        cnt = np.array([((times >= t - np.timedelta64(win, "m")) &
                         (times <= t + np.timedelta64(win, "m"))).sum() - 1 for t in times])
        st = g.stale.values
        if st.sum() > 3:
            print("    %-14s load mean=%.2f std=%.2f | stale-row load mean=%.2f | fail=%.3f" % (
                cl, cnt.mean(), cnt.std(), cnt[st].mean(), 1 - g.ok.values[st].mean()))

print()
print("=" * 74)
print("E. ALTERNATIVE: is west simply a different job mix / duration mix?")
print("=" * 74)
print("west vs others, mean duration and job_type distribution:")
print(d.groupby("cluster")["dur"].agg(["mean", "median"]))
ct = pd.crosstab(d.job_type, d.cluster, normalize="columns").round(3)
print(ct)
chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(d.job_type, d.cluster).values)
print("job_type x cluster chi2 p = %.3f (mix differs?)" % p)
chi2, p2, dof, _ = stats.chi2_contingency(pd.crosstab(d.cluster, pd.cut(d.dur, 5)).values)
print("duration x cluster chi2 p = %.3f" % p2)
