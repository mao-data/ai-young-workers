# Young Workers in AI-Exposed Occupations: Evidence from Public CPS Microdata

Do early-career workers lose ground in occupations most exposed to generative AI after ChatGPT's release (30 Nov 2022)? Brynjolfsson, Chandar & Chen (2025) document such a decline using proprietary ADP payroll data. This project asks whether the pattern is visible in **fully public** data: the Census Bureau's Current Population Survey (CPS) basic monthly microdata, merged with the occupation-level GPT exposure scores of Eloundou et al. (2024).

## Reproduce

All code lives in **[`ai_young_workers.ipynb`](ai_young_workers.ipynb)**, saved with its outputs so it can be read without running. To rebuild everything from a cold start:

```bash
./run_all.sh   # creates .venv, downloads ~790 MB, executes the notebook (~2 min after download)
```

| Section | What it does | Output |
|---|---|---|
| 1 | Download | 79 monthly CPS files (Jan 2020 – Aug 2026), exposure scores, BLS series |
| 2 | Exposure crosswalk | `data/clean/exposure_by_census_occ.csv` (525 of 526 Census occupations matched; Armed Forces excluded) |
| 3 | Parse and clean CPS | `data/clean/cps_2020_2026.parquet`, `results/cleaning_waterfall.csv` |
| 4 | Validate against BLS | assertions: employment matches BLS in all 79 months; every worker has an exposure score |
| 5–7 | Figures, event study, robustness | `results/fig1_*.png`, `fig2_*.png`, `event_study.csv`, `table2_pooled.csv` |

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

## Limitations

- Exposure measures *potential* task exposure, not actual AI use.
- Young-worker cells are small (~3,200 employed 22–25 year-olds per month); quarterly averaging reduces noise but individual quarters remain imprecise.
- High-exposure occupations may face other post-2022 shocks (interest rates, hiring cycles). Specification (5) removes the most obvious one, the tech layoff wave, but cannot rule out all of them.
- (8) is mechanical, not a placebo: shares sum to 100, so a falling young share implies rising shares elsewhere.

## References

- Brynjolfsson, E., Chandar, B., & Chen, R. (2025). Canaries in the Coal Mine? Six Facts about the Recent Employment Effects of Artificial Intelligence. Stanford Digital Economy Lab working paper.
- Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2024). GPTs are GPTs: Labor market impact potential of LLMs. *Science*, 384(6702).
- U.S. Census Bureau. Current Population Survey basic monthly public-use files, 2020–2026.
