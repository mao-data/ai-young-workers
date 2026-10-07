"""Parse CPS basic monthly fixed-width files into one typed person-month Parquet.

Each zip holds one 1,000-character fixed-width record per person. Field
positions were checked against every year's Census record layout (2020-2026,
plus the May 2024 revision) and are identical for all fields used here; the
primary-occupation field is named PEIO1OCD in 2020 and PTIO1OCD afterwards,
at the same position (860-863).

Records are sliced as a byte matrix with numpy rather than parsed line by
line, so a 120 MB month parses in about a second.

Kept universe: civilian adults (PRPERTYP == 2), age 16+. Every step's record
count is written to results/cleaning_waterfall.csv.

Output: data/clean/cps_2020_2026.parquet
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cps"
OUT = ROOT / "data" / "clean" / "cps_2020_2026.parquet"
WATERFALL = ROOT / "results" / "cleaning_waterfall.csv"

RECORD_LEN = 1000
# name: (start, end) as 1-indexed inclusive positions from the Census layout
FIELDS = {
    "year": (18, 21),      # HRYEAR4
    "month": (16, 17),     # HRMONTH
    "mis": (63, 64),       # HRMIS, month-in-sample
    "age": (122, 123),     # PRTAGE (topcoded at 85)
    "educ": (137, 138),    # PEEDUCA
    "pertype": (161, 162), # PRPERTYP: 2 = adult civilian
    "lfs": (180, 181),     # PEMLR: 1-2 employed, 3-4 unemployed, 5-7 NILF
    "cow": (432, 433),     # PEIO1COW: 1-5 wage/salary, 6-7 self-employed, 8 unpaid
    "earnwk": (527, 534),  # PTERNWA weekly earnings, 2 implied decimals (outgoing rotation)
    "orgwgt": (603, 612),  # PWORWGT, 4 implied decimals
    "cmpwgt": (846, 855),  # PWCMPWGT composite weight, 4 implied decimals
    "occ": (860, 863),     # PTIO1OCD / PEIO1OCD, 2018 Census occupation code
}


def parse_month(path):
    with zipfile.ZipFile(path) as z:
        (name,) = z.namelist()
        raw = z.read(name)
    stride = raw.index(b"\n") + 1  # 1001 for \n, 1002 for \r\n line endings
    if stride - RECORD_LEN not in (1, 2):
        raise ValueError(f"{path.name}: unexpected record length {stride}")
    if len(raw) % stride:  # final record without a trailing newline
        raw += b"\n" * (stride - len(raw) % stride)
    mat = np.frombuffer(raw, dtype=np.uint8).reshape(-1, stride)

    cols = {}
    for name, (a, b) in FIELDS.items():
        width = b - a + 1
        s = np.ascontiguousarray(mat[:, a - 1:b]).view(f"S{width}").ravel()
        cols[name] = pd.to_numeric(pd.Series(s).str.decode("ascii").str.strip(), errors="coerce")
    df = pd.DataFrame(cols)
    df["cmpwgt"] /= 1e4
    df["orgwgt"] /= 1e4
    df["earnwk"] = df["earnwk"].where(df["earnwk"] >= 0) / 100
    return df


def main():
    files = sorted(RAW.glob("*pub.zip"))
    if not files:
        sys.exit("No CPS files found; run 01_download.py first.")

    frames, steps = [], []
    for f in files:
        df = parse_month(f)
        n_raw = len(df)
        # Sanity checks: the file must be the month its name says.
        ym = df[["year", "month"]].drop_duplicates()
        assert len(ym) == 1, f"{f.name}: multiple year-months {ym.values.tolist()}"
        year, month = map(int, ym.iloc[0])

        df = df[df.pertype == 2]
        n_adult = len(df)
        df = df[df.age >= 16]
        n_16 = len(df)
        assert df.lfs.between(1, 7).all(), f"{f.name}: labor force status out of range"
        assert (df.cmpwgt >= 0).all(), f"{f.name}: negative weight"

        steps.append({"year": year, "month": month, "raw_records": n_raw,
                      "adult_civilian": n_adult, "age_16plus": n_16,
                      "employed": int(df.lfs.isin([1, 2]).sum()),
                      "employed_with_occ": int((df.lfs.isin([1, 2]) & (df.occ > 0)).sum())})
        frames.append(df.drop(columns="pertype"))
        print(f"{f.name}: {n_raw:,} records -> {n_16:,} adults 16+", flush=True)

    out = pd.concat(frames, ignore_index=True)
    out = out.astype({"year": "int16", "month": "int8", "mis": "int8", "age": "int8",
                      "educ": "int8", "lfs": "int8", "cow": "int8", "occ": "int16",
                      "cmpwgt": "float32", "orgwgt": "float32", "earnwk": "float32"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    wf = pd.DataFrame(steps).sort_values(["year", "month"])
    WATERFALL.parent.mkdir(parents=True, exist_ok=True)
    wf.to_csv(WATERFALL, index=False)
    print(f"\n{len(files)} months, {len(out):,} person-month rows -> {OUT.name} "
          f"({OUT.stat().st_size / 1e6:.0f} MB)")
    print(wf.drop(columns=["year", "month"]).sum().to_string())


if __name__ == "__main__":
    main()
