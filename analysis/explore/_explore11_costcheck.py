# -*- coding: utf-8 -*-
"""Check the Option A cost model carefully.

Option A refreshes the snapshot before routing. The 261 stale tasks do not
simply become "no failures" - they become FRESH tasks, and fresh tasks still
fail at the fresh rate. The earlier draft credited A with avoiding all 81
stale failures, which overstates it.
"""
import pandas as pd, numpy as np, os

CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "render_tasks.csv")
raw = pd.read_csv(CSV, dtype=str)
raw["c"] = pd.to_datetime(raw.created_at, utc=True)
raw["s"] = pd.to_datetime(raw.cluster_synced_at, utc=True)
raw["lag"] = (raw.c - raw.s).dt.total_seconds() / 60
raw["ok"] = pd.to_numeric(raw.is_success, errors="coerce")
key = ["task_id", "created_at", "cluster_synced_at", "cluster", "job_type",
       "free_slots", "video_duration_sec", "is_success"]
d = raw.drop_duplicates(subset=key).dropna(subset=["ok"]).copy()
d["stale"] = d.lag >= 20

FAIL, DEG = 75.0, 10.0
n_st = int(d.stale.sum())
n_fr = len(d) - n_st
f_st = int((d[d.stale].ok == 0).sum())
f_fr = int((d[~d.stale].ok == 0).sum())
r_fr = f_fr / n_fr
base = (f_st + f_fr) * FAIL

print("baseline: %d stale failures + %d fresh failures = %d failures = %.0f min"
      % (f_st, f_fr, f_st + f_fr, base))
print("fresh failure rate = %.4f" % r_fr)
print()

print("OPTION A - refresh snapshot, then route normally")
a_exp = n_st * r_fr
print("  the %d stale tasks become fresh -> expected failures %.1f" % (n_st, a_exp))
a_total = f_fr + a_exp
print("  total expected failures = %d fresh + %.1f = %.1f" % (f_fr, a_exp, a_total))
a_cost = a_total * FAIL
print("  cost = %.1f x %.0f = %.0f min   (refresh latency ~0)" % (a_total, FAIL, a_cost))
print("  saving = %.0f - %.0f = %.0f min (%.1f h)" % (base, a_cost, base - a_cost, (base - a_cost) / 60))
print("  naive (WRONG) version credited all 81: saving %.0f min" % (f_st * FAIL))
print()

print("OPTION B - send stale tasks to the degrade channel")
b_cost = f_fr * FAIL + n_st * DEG
print("  residual fresh failures %d x %.0f = %.0f min" % (f_fr, FAIL, f_fr * FAIL))
print("  degrade cost %d x %.0f = %.0f min" % (n_st, DEG, n_st * DEG))
print("  cost = %.0f min ; saving = %.0f - %.0f = %.0f min (%.1f h)"
      % (b_cost, base, b_cost, base - b_cost, (base - b_cost) / 60))
print("  (assumes the degrade channel avoids the stale failures entirely)")
print()

print("OPTION B' - as B, but exempt west (its stale rate is below break-even)")
w = d[(d.cluster == "cluster_west") & d.stale]
o = d[(d.cluster != "cluster_west") & d.stale]
n_w, f_w = len(w), int((w.ok == 0).sum())
n_o, f_o = len(o), int((o.ok == 0).sum())
bp_cost = f_fr * FAIL + f_w * FAIL + n_o * DEG
print("  exempt west: %d tasks stay normal -> %d failure = %.0f min" % (n_w, f_w, f_w * FAIL))
print("  degrade %d tasks -> %.0f min" % (n_o, n_o * DEG))
print("  cost = %.0f min ; saving = %.0f min (%.1f h)"
      % (bp_cost, base - bp_cost, (base - bp_cost) / 60))
print()
print("=> A is better than B by %.0f min/week." % ((base - a_cost) - (base - b_cost)))
print("   Corrected A saving: %.0f min (%.1f h), not %.0f min." % (
    base - a_cost, (base - a_cost) / 60, f_st * FAIL))
