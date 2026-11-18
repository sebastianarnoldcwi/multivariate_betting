"""
Build the case-study data from the OHIE public-use files.

Usage:  python3 extract_pairs.py <path to OHIE_Public_Use_Files/OHIE_Data>

Reads oregonhie_descriptive_vars.dta and oregonhie_survey12m_vars.dta, keeps respondents to the
12-month survey living in single-person households (numhh_list == 1), builds the eight oriented
endpoints, orders the respondents by survey return date, pairs the i-th selected with the i-th
non-selected respondent, and writes

  data/ohie_respondents.csv   one row per respondent: arm (1 = selected), return-order rank,
                              the eight oriented endpoint values (NaN where not answered)
  data/ohie_pairs.csv         one row per pair in return order: the eight oriented differences
                              X_i = orient_i * (Y_i^selected - Y_i^non-selected), complete cases only
  data/endpoints.csv          endpoint table: variable, orientation, range, arm means, difference,
                              standardised difference, numbers of respondents
  data/correlations.csv       correlation matrix of the paired differences

The raw files are not redistributed; they are available from the NBER
(https://www.nber.org/research/data/oregon-health-insurance-experiment-data).
"""
import sys, os
import numpy as np, pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "../../../casestudy/OHIE_Public_Use_Files/OHIE_Data"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data"); os.makedirs(OUT, exist_ok=True)

desc = pd.read_stata(os.path.join(RAW, "oregonhie_descriptive_vars.dta"), convert_categoricals=False)
s12 = pd.read_stata(os.path.join(RAW, "oregonhie_survey12m_vars.dta"), convert_categoricals=False)
d = desc.merge(s12, on="person_id")
r = d[(d.returned_12m == 1) & (d.numhh_list == 1)].copy()
r["phq2"] = (r.dep_interest_12m - 1) + (r.dep_sad_12m - 1)          # each item coded 1..4 -> 0..3

# name: (variable, orientation, lower bound, upper bound, description)
ENDPOINTS = {
    "visit":   ("doc_any_12m",        +1, 0, 1,  "any outpatient visit, last 6 months"),
    "oop":     ("cost_any_oop_12m",   -1, 0, 1,  "any out-of-pocket medical expense, last 6 months"),
    "borrow":  ("cost_borrow_12m",    -1, 0, 1,  "borrowed money or skipped bills to pay medical costs"),
    "phq2":    ("phq2",               -1, 0, 6,  "PHQ-2 depression score"),
    "baddays": ("baddays_ment_12m",   -1, 0, 30, "days of the last 30 with poor mental health"),
    "fairpoor":("health_gen_bin_12m", -1, 0, 1,  "self-rated health fair or poor"),
    "unhappy": ("happiness_12m",      -1, 1, 3,  "happiness (1 very, 2 pretty, 3 not too happy)"),
    "er":      ("er_any_12m",         -1, 0, 1,  "any emergency-department visit, last 6 months"),
}
names = list(ENDPOINTS)

# respondent-level table, oriented so that larger = selected arm doing better
r = r.sort_values(["dt_returned_12m", "person_id"]).reset_index(drop=True)
resp = pd.DataFrame({"arm": r.treatment.astype(int), "rank": np.arange(1, len(r) + 1)})
for k, (v, o, lo, hi, _) in ENDPOINTS.items():
    resp[k] = o * r[v].values
resp.to_csv(os.path.join(OUT, "ohie_respondents.csv"), index=False)

# endpoint table
rows = []
for k, (v, o, lo, hi, desc_) in ENDPOINTS.items():
    a = r.loc[r.treatment == 0, v]; b = r.loc[r.treatment == 1, v]
    diff = o * (b.mean() - a.mean()); sp = np.sqrt((a.var() + b.var()) / 2)
    rows.append(dict(name=k, variable=v, description=desc_, orientation=o, lower=lo, upper=hi,
                     bound_c=hi - lo, mean_nonselected=a.mean(), mean_selected=b.mean(),
                     oriented_difference=diff, standardised=diff / sp,
                     n_nonselected=int(a.notna().sum()), n_selected=int(b.notna().sum())))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "endpoints.csv"), index=False)

# pairs in return order (complete cases on all eight endpoints)
Q = resp.dropna(subset=names)
B = Q[Q.arm == 1]; A = Q[Q.arm == 0]; m = min(len(A), len(B))
X = pd.DataFrame({k: B[k].values[:m] - A[k].values[:m] for k in names})
X.insert(0, "round", np.arange(1, m + 1))
X.to_csv(os.path.join(OUT, "ohie_pairs.csv"), index=False)
X[names].corr().round(4).to_csv(os.path.join(OUT, "correlations.csv"))

print(f"respondents in single-person households: {len(r)} (selected {int(r.treatment.sum())})")
print(f"complete cases on all eight endpoints: {len(Q)} (selected {int(Q.arm.sum())}); pairs: {m}")
print(pd.DataFrame(rows)[["name", "orientation", "bound_c", "mean_nonselected", "mean_selected", "oriented_difference", "standardised"]].round(4).to_string(index=False))
print("\ncorrelations of paired differences:\n", X[names].corr().round(2).to_string())
