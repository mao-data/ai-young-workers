"""Map Eloundou et al. (2024) GPT exposure scores onto 2018 Census occupation
codes, the coding used in the CPS from January 2020 onward.

Chain: O*NET-SOC (8-digit) -> 2018 SOC (6-digit) -> Census occupation code.
The Census crosswalk expresses aggregated occupations three ways, each handled
explicitly:
  1. "combines-item" rows listing every SOC code inside a combined category;
  2. wildcard codes such as 15-124X (prefix match);
  3. broad-group codes ending in 0, such as 13-2070 (prefix match on the first
     six characters) when no exact detailed code exists.
Scores are the unweighted mean across matched O*NET occupations.

Output: data/clean/exposure_by_census_occ.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "clean" / "exposure_by_census_occ.csv"

# beta = E1 + 0.5 * E2: the paper's headline measure. dv_ = GPT-4 rated, human_ = annotators.
SCORES = {"dv_rating_beta": "exp_gpt4", "human_rating_beta": "exp_human"}


def census_to_soc_patterns(xwalk):
    """Return {census_code: (title, [soc patterns])} for every detailed occupation."""
    summary, items, titles, current = {}, {}, {}, None
    for rtype, code, soc, title in xwalk.itertuples(index=False):
        if rtype == "detail":
            current = code
            titles[code] = title
            summary[code] = [soc] if isinstance(soc, str) else []
            items[code] = []
        elif rtype == "combines-item" and current is not None:
            items[current].append(soc)
        else:
            current = None
    # When a combined category lists its member SOC codes, use those; else the summary code.
    return {c: (titles[c], items[c] or summary[c]) for c in titles}


def match(pattern, soc6):
    """Return (matched 6-digit SOC codes, method) for one Census SOC pattern."""
    if "X" in pattern:
        prefix = pattern[: pattern.index("X")]
        return [s for s in soc6 if s.startswith(prefix)], "wildcard"
    if pattern in soc6:
        return [pattern], "exact"
    if pattern.endswith("0"):
        # 13-2070 -> 13-207*, 25-1000 -> 25-1*: strip trailing zeros to the group prefix.
        prefix = pattern.rstrip("0")
        return [s for s in soc6 if s.startswith(prefix)], "broad"
    if pattern.endswith("9"):
        # "All other" residual codes (e.g. 21-1019) have no O*NET profile;
        # fall back to their broad occupation group (21-101*), then minor group (21-10*).
        for k in (6, 5):
            m = [s for s in soc6 if s.startswith(pattern[:k])]
            if m:
                return m, "residual"
    return [], "none"


def main():
    onet = pd.read_csv(RAW / "exposure" / "occ_level.csv", dtype={"O*NET-SOC Code": str})
    onet["soc6"] = onet["O*NET-SOC Code"].str[:7]
    soc_scores = onet.groupby("soc6")[list(SCORES)].mean().rename(columns=SCORES)
    soc6 = set(soc_scores.index)

    xwalk = pd.read_csv(RAW / "cps" / "cps_occupation_codes.csv", dtype=str, encoding="utf-8-sig")
    mapping = census_to_soc_patterns(xwalk)

    rows = []
    for code, (title, patterns) in mapping.items():
        matched, methods = [], []
        for p in patterns:
            m, how = match(p, soc6)
            matched += m
            methods.append(how)
        matched = sorted(set(matched))
        rec = {"occ_code": int(code), "title": title, "soc_patterns": ";".join(patterns),
               "n_soc_matched": len(matched), "match_method": ";".join(sorted(set(methods)))}
        rec.update(soc_scores.loc[matched].mean().to_dict() if matched else {v: float("nan") for v in SCORES.values()})
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("occ_code")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    unmatched = out[out.n_soc_matched == 0]
    print(f"Census occupations: {len(out)} | matched: {len(out) - len(unmatched)} "
          f"| unmatched: {len(unmatched)}")
    print(out.match_method.value_counts().to_string())
    if len(unmatched):
        print("Unmatched:\n" + unmatched[["occ_code", "title", "soc_patterns"]].to_string(index=False))


if __name__ == "__main__":
    main()
