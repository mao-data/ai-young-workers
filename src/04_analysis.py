"""Young-worker employment in AI-exposed occupations, 2021-2026.

Design follows the spirit of Brynjolfsson, Chandar & Chen (2025) using public
CPS microdata instead of ADP payroll records.

  Figure 1: employment of 22-25 year-olds by GPT-exposure quintile,
            indexed to 2022Q3 (the last quarter before ChatGPT's release).
  Figure 2: event study at the occupation level.
      young_share_ot = a_o + l_t + sum_k b_k * exposure_o * 1[t = k] + e_ot
    where young_share is the 22-25 share of occupation o's employment in
    quarter t, exposure is standardized, weights are 2022 occupation
    employment, SEs are clustered by occupation, and k = 2022Q3 is omitted.
  Table 2: pooled post-period coefficient under alternative definitions.

Quarterly values average the available months, so the missing October 2025
release (household survey not collected during the Oct 1 - Nov 12, 2025
federal shutdown; BLS publishes no estimate for that month) does not bias Q4 2025.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

START = "2021Q1"   # after the 2020 pandemic collapse
BASE = "2022Q3"    # last full quarter before ChatGPT (released 30 Nov 2022)
YOUNG = (22, 25)


def load():
    d = pd.read_parquet(CLEAN / "cps_2020_2026.parquet")
    d = d[d.lfs.isin([1, 2]) & (d.occ > 0)].copy()  # employed with an occupation
    exp = pd.read_csv(CLEAN / "exposure_by_census_occ.csv")[["occ_code", "exp_gpt4", "exp_human"]]
    d = d.merge(exp, left_on="occ", right_on="occ_code", how="inner")
    d["q"] = pd.PeriodIndex(pd.to_datetime(dict(year=d.year, month=d.month, day=1)), freq="Q")
    return d[d.q >= pd.Period(START, "Q")]


def quarterly(cells, keys):
    """Weighted employment per month, then averaged over the months observed in each quarter."""
    m = cells.groupby(keys + ["q", "year", "month"]).cmpwgt.sum().reset_index()
    return m.groupby(keys + ["q"]).cmpwgt.mean().rename("emp").reset_index()


def assign_quintiles(d, col):
    """Employment-weighted quintiles: each holds ~20% of 2022 employment (all ages)."""
    base = d[d.year == 2022].groupby("occ").agg(emp=("cmpwgt", "sum"), x=(col, "first")).sort_values("x")
    base["cum"] = base.emp.cumsum() / base.emp.sum()
    base["quintile"] = np.minimum((base.cum * 5).apply(np.ceil), 5).astype(int)
    return base.quintile


def figure_index(d):
    q5 = d.occ.map(assign_quintiles(d, "exp_gpt4"))
    d = d.assign(quintile=q5.map({1: "Least exposed (Q1)", 5: "Most exposed (Q5)"}).fillna("Middle (Q2-Q4)"))
    colors = {"Least exposed (Q1)": "#2471a3", "Middle (Q2-Q4)": "0.6", "Most exposed (Q5)": "#c0392b"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (label, lo, hi) in zip(axes, [("Ages 22-25", *YOUNG), ("Ages 35-49", 35, 49)]):
        q = quarterly(d[d.age.between(lo, hi)], ["quintile"])
        q["index"] = q.groupby("quintile").emp.transform(lambda s: 100 * s / s[q.loc[s.index, "q"] == pd.Period(BASE, "Q")].iloc[0])
        for k, g in q.groupby("quintile"):
            ax.plot(g.q.dt.to_timestamp(), g["index"], label=k, color=colors[k],
                    lw=1.4 if k.startswith("Middle") else 2.4, marker="o", ms=3)
        ax.axvline(pd.Timestamp("2022-11-30"), color="k", ls="--", lw=0.8)
        ax.axhline(100, color="k", lw=0.5)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"Employment index ({BASE} = 100)")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "fig1_index_by_quintile.png", dpi=200)
    plt.close(fig)


def occ_panel(d, exp_col="exp_gpt4", young=YOUNG):
    q_all = quarterly(d, ["occ"])
    q_young = quarterly(d[d.age.between(*young)], ["occ"]).rename(columns={"emp": "emp_young"})
    p = q_all.merge(q_young, on=["occ", "q"], how="left").fillna({"emp_young": 0})
    p["young_share"] = 100 * p.emp_young / p.emp
    w = d[d.year == 2022].groupby("occ").cmpwgt.sum().rename("w")
    x = d.groupby("occ")[exp_col].first()
    p = p.merge(w, on="occ").merge(((x - x.mean()) / x.std()).rename("exposure"), on="occ")
    return p[p.w > 0]


def event_study(p):
    quarters = sorted(p.q.unique())
    terms = []
    for k in quarters:
        if k == pd.Period(BASE, "Q"):
            continue
        name = f"x_{k}"
        p[name] = p.exposure * (p.q == k)
        terms.append(name)
    p = p.assign(qs=p.q.astype(str))
    fit = smf.wls(f"young_share ~ {' + '.join(terms)} + C(occ) + C(qs)", data=p, weights=p.w).fit(
        cov_type="cluster", cov_kwds={"groups": p.occ})
    ci = fit.conf_int()
    out = pd.DataFrame({"q": [t[2:] for t in terms], "coef": fit.params[terms].values,
                        "lo": ci.loc[terms, 0].values, "hi": ci.loc[terms, 1].values})
    return pd.concat([out, pd.DataFrame([{"q": BASE, "coef": 0.0, "lo": 0.0, "hi": 0.0}])]).sort_values("q")


def pooled(p, event=BASE, end=None):
    if end is not None:
        p = p[p.q <= pd.Period(end, "Q")]
    p = p.assign(post=(p.q > pd.Period(event, "Q")).astype(int), qs=p.q.astype(str))
    fit = smf.wls("young_share ~ exposure:post + C(occ) + C(qs)", data=p, weights=p.w).fit(
        cov_type="cluster", cov_kwds={"groups": p.occ})
    b, se = fit.params["exposure:post"], fit.bse["exposure:post"]
    at = p.q == pd.Period(event, "Q")
    base_mean = np.average(p.loc[at, "young_share"], weights=p.loc[at, "w"])
    return {"coef": b, "se": se, "p": fit.pvalues["exposure:post"], "base_mean": base_mean,
            "n_occ": p.occ.nunique(), "n_obs": len(p)}


def main():
    d = load()
    figure_index(d)

    es = event_study(occ_panel(d))
    es.to_csv(RES / "event_study.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = pd.PeriodIndex(es.q, freq="Q").to_timestamp()
    ax.errorbar(x, es.coef, yerr=[es.coef - es.lo, es.hi - es.coef], fmt="o", color="#c0392b", ms=4, capsize=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(pd.Timestamp("2022-11-30"), color="k", ls="--", lw=0.8)
    ax.set_ylabel("Effect on 22-25 share (pp)\nper 1 s.d. exposure")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "fig2_event_study.png", dpi=200)
    plt.close(fig)

    base = occ_panel(d)
    computer = d.occ.between(1005, 1240)  # Census codes for computer & mathematical occupations
    specs = {
        "(1) Baseline: GPT-4 exposure, ages 22-25": pooled(base),
        "(2) Human-rated exposure": pooled(occ_panel(d, exp_col="exp_human")),
        "(3) Wage and salary workers only": pooled(occ_panel(d[d.cow.between(1, 5)])),
        "(4) Ages 22-29": pooled(occ_panel(d, young=(22, 29))),
        "(5) Excluding computer & math occupations": pooled(occ_panel(d[~computer])),
        "(6) Sample ends 2025Q4": pooled(base, end="2025Q4"),
        "(7) Placebo: fake event 2021Q4, pre-ChatGPT data only": pooled(base, event="2021Q4", end=BASE),
        "(8) Composition check: ages 35-49": pooled(occ_panel(d, young=(35, 49))),
    }
    tab = pd.DataFrame(specs).T
    tab.to_csv(RES / "table2_pooled.csv")
    print(tab.round(4).to_string())
    print("\nEvent study:\n" + es.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
