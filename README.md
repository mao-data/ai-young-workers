# Young Workers in AI-Exposed Occupations: Evidence from Public CPS Microdata

Do early-career workers lose ground in occupations most exposed to generative AI after ChatGPT's release (30 Nov 2022)? Brynjolfsson, Chandar & Chen (2025) document such a decline using proprietary ADP payroll data. This project asks whether the pattern is visible in **fully public** data: the Census Bureau's Current Population Survey (CPS) basic monthly microdata, merged with the occupation-level GPT exposure scores of Eloundou et al. (2024).

**Paper:** [`paper/paper.pdf`](paper/paper.pdf) (5 pages; LaTeX source in `paper/paper.tex`).

## Reproduce

All code lives in **[`ai_young_workers.ipynb`](ai_young_workers.ipynb)**, saved with its outputs so it can be read without running. To rebuild everything from a cold start:

```bash
./run_all.sh   # creates .venv, downloads ~3.2 GB, executes the notebook (~3 min after download)
```

| Section | What it does | Output |
|---|---|---|
| 1 | Download | 79 monthly CPS files (Jan 2020 – Aug 2026), ACS PUMS 2021–2024, exposure scores (potential and observed), BLS series |
| 2 | Exposure crosswalk | `data/clean/exposure_by_census_occ.csv` (525 of 526 Census occupations matched; Armed Forces excluded) |
| 3 | Parse and clean CPS | `data/clean/cps_2020_2026.parquet`, `results/cleaning_waterfall.csv` |
| 4 | Validate against BLS | assertions: employment matches BLS in all 79 months; every worker has an exposure score |
| 5–7 | Figures, event study, robustness | `results/fig1_*.png`, `fig2_*.png`, `event_study.csv`, `table2_pooled.csv` |
| 8 | Precision, influence, randomization inference, unemployment | printed in notebook |
| 9 | Cross-validation: ACS and observed AI exposure | `data/clean/acs_2021_2024_employed.parquet`, `results/table3_cross_validation.csv` |

Python 3.14; versions pinned in `requirements.txt`.

## Data construction

- **CPS parsing.** Fixed-width records (1,000 characters each) sliced as a numpy byte matrix. Field positions verified against every year's Census record layout (2020–2026 and the May 2024 revision); all used fields are at identical positions. The occupation field is named `PEIO1OCD` in 2020 and `PTIO1OCD` afterwards, at the same position.
- **One occupation coding throughout.** CPS adopted 2018 Census occupation codes in January 2020, so the window needs no cross-vintage crosswalk.
- **Missing month.** October 2025 was never collected (federal shutdown, Oct 1 – Nov 12, 2025). Quarterly values average the months available.
- **Exposure crosswalk.** O\*NET-SOC (8-digit) → 2018 SOC (6-digit) → Census code. Aggregated Census categories handled explicitly: listed member codes, wildcard codes (`15-124X`), broad-group codes (`13-2070`), and "all other" residual codes (fallback to parent group).

**Cleaning waterfall (79 months pooled)**

| Step | Records |
|---|---|
| Raw person records | 10,037,598 |
| Adult civilian (PRPERTYP = 2) | 6,564,159 |
| Age 16+ | 6,464,072 |
| Employed | 3,722,523 |
| Employed with occupation matched to exposure | 3,722,523 |

**Validation.** Weighted employment (composite weight) reproduces the BLS published series LNU02000000 in all 79 months; maximum deviation 0.0003%.

## Results

Main specification: occupation × quarter panel, 2021Q1 – 2026Q3; outcome = 22–25 share of occupation employment (pp); exposure standardized; occupation and quarter fixed effects; weights = 2022 occupation employment; SEs clustered by occupation; omitted quarter 2022Q3.

| Specification | Coef. (pp per 1 s.d.) | SE | p |
|---|---|---|---|
| (1) Baseline: GPT-4 exposure, ages 22–25 | −0.294 | 0.089 | 0.001 |
| (2) Human-rated exposure | −0.290 | 0.090 | 0.001 |
| (3) Wage and salary workers only | −0.308 | 0.096 | 0.001 |
| (4) Ages 22–29 | −0.415 | 0.119 | 0.001 |
| (5) Excluding computer & math occupations | −0.308 | 0.092 | 0.001 |
| (6) Sample ends 2025Q4 | −0.231 | 0.089 | 0.009 |
| (7) Placebo: fake event 2021Q4, pre-ChatGPT data only | +0.044 | 0.124 | 0.721 |
| (8) Composition check: ages 35–49 | +0.531 | 0.133 | <0.001 |

Baseline mean young share in 2022Q3: 7.74%, so (1) is a ~3.8% relative decline per s.d. of exposure. Pre-period event-study coefficients are all near zero (`results/event_study.csv`).

### Further checks (Section 8)

- **Weighting.** Unweighted, the coefficient is −0.09 (p = 0.73), but this is imprecision rather than absence: the median occupation has ~124 young-worker observations over six years. Restricting to occupations with ≥300 young observations, the unweighted estimate is −0.46 (p = 0.007).
- **Influence.** Leave-one-out over the 30 largest occupations: −0.312 to −0.278, all p ≤ 0.002.
- **Randomization inference.** 0 of 1,000 permutations of exposure across occupations produce |b| ≥ 0.294 (95th percentile of |b|: 0.165).
- **Mechanism.** Unemployment by last-job exposure rose about equally for ages 22–25 (+0.58 pp) and 35–49 (+0.55 pp). A young-specific fall in employment share without a young-specific rise in unemployment points to **reduced entry** rather than displacement of young incumbents.

### Cross-validation (Section 9)

Independent survey (ACS 1-year PUMS, ~105,000 employed 22–25 year-olds per year) and an independent exposure measure (Anthropic Economic Index *observed* exposure; correlation with potential exposure 0.67 employment-weighted, Spearman 0.74).

| Specification | Coef. | SE | p |
|---|---|---|---|
| CPS quarterly 2021Q1–2026Q3, potential exposure (baseline) | −0.294 | 0.089 | 0.001 |
| CPS annual 2021–2024, potential exposure | −0.276 | 0.105 | 0.008 |
| **ACS annual 2021–2024, potential exposure** | **−0.149** | 0.056 | 0.007 |
| ACS, human-rated exposure | −0.164 | 0.063 | 0.009 |
| ACS, excluding computer & math | −0.137 | 0.058 | 0.018 |
| ACS, ages 22–29 | −0.060 | 0.076 | 0.430 |
| CPS quarterly, observed exposure (AEI) | −0.183 | 0.070 | 0.009 |
| ACS annual, observed exposure (AEI) | −0.122 | 0.049 | 0.013 |

ACS by year relative to 2022: 2021 +0.046 (p = 0.47), 2023 −0.042 (p = 0.54), 2024 −0.210 (p = 0.006). ACS employment runs 2.5–3.8% above CPS, as expected from its broader universe (group quarters) and rolling reference period.

## Limitations

- Potential and observed exposure point the same way, but neither identifies AI as the cause; the estimates are descriptive.
- CPS young-worker cells are small (~3,200 employed 22–25 year-olds per month); individual quarters are imprecise, which is why the ACS replication matters.
- The ACS ends in 2024; the 2025 file will show whether the larger 2026 CPS coefficients reflect a strengthening trend or noise.
- High-exposure occupations may face other post-2022 shocks (interest rates, hiring cycles). Specification (5) removes the most obvious one, the tech layoff wave, but cannot rule out all of them.
- (8) is mechanical, not a placebo: shares sum to 100, so a falling young share implies rising shares elsewhere.

## References

- Brynjolfsson, E., Chandar, B., & Chen, R. (2025). Canaries in the Coal Mine? Six Facts about the Recent Employment Effects of Artificial Intelligence. Stanford Digital Economy Lab working paper.
- Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2024). GPTs are GPTs: Labor market impact potential of LLMs. *Science*, 384(6702).
- Anthropic. Anthropic Economic Index, labor market impacts: occupation-level observed exposure (`labor_market_impacts/job_exposure.csv`), Hugging Face dataset `Anthropic/EconomicIndex`.
- U.S. Census Bureau. American Community Survey 1-year Public Use Microdata Sample, 2021–2024.
- U.S. Census Bureau. Current Population Survey basic monthly public-use files, 2020–2026.
