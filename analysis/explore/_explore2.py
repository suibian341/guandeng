# -*- coding: utf-8 -*-
"""Ad-hoc exploration #2: west anomaly, free_slots signal, statistical tests."""
import pandas as pd, numpy as np, os
from scipy import stats

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

print("=" * 70)
print("A. IS free_slots INFORMATIVE AT ALL?")
print("=" * 70)
print("corr(slots, ok)      = %.4f" % d.slots.corr(d.ok))
print("corr(lag, ok)        = %.4f" % d.lag.corr(d.ok))
print("corr(dur, ok)        = %.4f" % d.dur.corr(d.ok))
print("corr(lag, slots)     = %.4f" % d.lag.corr(d.slots))
print()
print("fail rate by slots bucket, WITHIN fresh and WITHIN stale:")
for name, sub in [("FRESH", d[~d.stale]), ("STALE", d[d.stale])]:
    sub = sub.copy(); sub["sb"] = pd.cut(sub.slots, [-1, 0, 5, 10, 15, 20, 25, 30])
    print("--", name)
    print(sub.groupby("sb", observed=True)["ok"].agg(["count", "mean"]).assign(
        fail=lambda x: 1 - x["mean"])[["count", "fail"]])

print()
print("=" * 70)
print("B. WEST ANOMALY")
print("=" * 70)
st = d[d.stale]
for cl, g in st.groupby("cluster"):
    print("%-14s n=%3d fails=%2d rate=%.3f" % (cl, len(g), (g.ok == 0).sum(), 1 - g.ok.mean()))
w = st[st.cluster == "cluster_west"]; o = st[st.cluster != "cluster_west"]
tab = [[(w.ok == 0).sum(), (w.ok == 1).sum()], [(o.ok == 0).sum(), (o.ok == 1).sum()]]
print("Fisher exact (west vs others, stale only):", stats.fisher_exact(tab))
print()
print("Do west stale rows differ in observable features?")
feat = ["lag", "slots", "dur"]
for f in feat:
    a, b = w[f], o[f]
    print("  %-6s west mean=%.2f  others mean=%.2f  MWU p=%.4f" % (
        f, a.mean(), b.mean(), stats.mannwhitneyu(a, b).pvalue))
print()
print("  west stale job_type mix vs others:")
print(pd.crosstab(st.job_type, st.cluster == "cluster_west", normalize="columns").round(3))

print()
print("=" * 70)
print("C. STATISTICAL TESTS: lag>=20 vs ok")
print("=" * 70)
ct = pd.crosstab(d.stale, d.ok)
print(ct)
odds, p = stats.fisher_exact(ct.values)
print("Fisher exact p = %.3g, odds ratio = %.3f" % (p, odds))
chi2, pchi, dof, _ = stats.chi2_contingency(ct.values)
print("Chi2 = %.2f, p = %.3g" % (chi2, pchi))
print()
print("Mantel-Haenszel stratified by cluster (excludes west):")
sub = d[d.cluster != "cluster_west"]
tables = []
for cl, g in sub.groupby("cluster"):
    t = pd.crosstab(g.stale, g.ok)
    tables.append(t.values)
try:
    from statsmodels.stats.contingency_tables import StratifiedTable
    print(StratifiedTable(tables).test_null_odds())
except Exception as e:
    # manual MH
    num = den = 0.0
    for t in tables:
        a, b = t[1][0], t[1][1]; c, dd_ = t[0][0], t[0][1]
        n = a + b + c + dd_
        num += a * dd_ / n; den += b * c / n
    print("  manual MH odds ratio = %.3f" % (num / den))

print()
print("=" * 70)
print("D. ROC / discrimination: which single field best predicts failure?")
print("=" * 70)
def auc(score, y):
    # rank-based AUC (higher score => more likely y=1)
    r = stats.rankdata(score)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
y = (d.ok == 0).astype(int).values
for name, col, sign in [("lag (stale)", d.lag.values, 1), ("free_slots", d.slots.values, -1),
                        ("video_duration_sec", d.dur.values, 1)]:
    a = auc(sign * col, y)
    print("  %-20s AUC = %.3f" % (name, max(a, 1 - a)))
print("  %-20s AUC = %.3f" % ("lag>=20 (binary)", auc(d.stale.astype(int).values, y)))

print()
print("=" * 70)
print("E. LOGISTIC REGRESSION (manual IRLS)")
print("=" * 70)
X = np.column_stack([
    np.ones(len(d)),
    (d.lag >= 20).astype(float),
    (d.slots == 0).astype(float),
    d.slots.values / 30.0,
    d.dur.values / 100.0,
    (d.cluster == "cluster_west").astype(float),
])
names = ["intercept", "lag>=20", "slots==0", "slots/30", "dur/100", "is_west"]
yv = (d.ok == 0).astype(float).values
beta = np.zeros(X.shape[1])
for _ in range(200):
    eta = X @ beta
    mu = 1 / (1 + np.exp(-eta))
    W = mu * (1 - mu) + 1e-9
    z = eta + (yv - mu) / W
    beta = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * z))
se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * W[:, None]))))
for n_, b_, s_ in zip(names, beta, se):
    z_ = b_ / s_
    p_ = 2 * (1 - stats.norm.cdf(abs(z_)))
    print("  %-12s coef=%+7.3f  se=%.3f  z=%+6.2f  p=%.3g  OR=%.2f" % (n_, b_, s_, z_, p_, np.exp(b_)))

print()
print("=" * 70)
print("F. BUSINESS IMPACT")
print("=" * 70)
FAIL_MIN = 75.0
DEGRADE_MIN = 10.0
n_total = len(d)
n_stale = int(d.stale.sum())
stale_fail = 1 - d[d.stale].ok.mean()
print("reported rows: %d, stale rows: %d (%.1f%%)" % (n_total, n_stale, 100 * n_stale / n_total))
print("stale fail rate: %.1f%%  -> failures attributable to stale: %.0f" % (
    100 * stale_fail, n_stale * stale_fail))
print("timeout minutes on stale rows: %.0f" % (n_stale * stale_fail * FAIL_MIN))
print("cost if ALL stale rows degraded: %.0f min" % (n_stale * DEGRADE_MIN))
print("break-even P(fail) = %.4f (%.2f%%)" % (DEGRADE_MIN / FAIL_MIN, 100 * DEGRADE_MIN / FAIL_MIN))
