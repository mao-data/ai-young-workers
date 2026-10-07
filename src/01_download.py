"""Download CPS basic monthly public-use files (Jan 2020 - Aug 2026) and the
Eloundou et al. (2024) occupation-level GPT exposure scores.

CPS switched to 2018 Census occupation codes in January 2020, so this window
uses a single occupation coding throughout. Files are fetched once; existing
files are skipped.
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CPS = ROOT / "data" / "raw" / "cps"
RAW_EXP = ROOT / "data" / "raw" / "exposure"

CPS_URL = "https://www2.census.gov/programs-surveys/cps/datasets/{year}/basic/{mon}{yy}pub.zip"
EXP_URL = "https://raw.githubusercontent.com/openai/GPTs-are-GPTs/main/data/{f}"
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
FIRST, LAST = (2020, 1), (2026, 8)
# The October 2025 household survey was never collected (federal shutdown,
# Oct 1 - Nov 12, 2025), so Census publishes no file for that month.
KNOWN_MISSING = {(2025, 10)}


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return "ok"


def main():
    RAW_CPS.mkdir(parents=True, exist_ok=True)
    RAW_EXP.mkdir(parents=True, exist_ok=True)

    for f in ["occ_level.csv"]:
        print(f, fetch(EXP_URL.format(f=f), RAW_EXP / f))

    failed = []
    for year in range(FIRST[0], LAST[0] + 1):
        for m, mon in enumerate(MONTHS, start=1):
            if (year, m) < FIRST or (year, m) > LAST or (year, m) in KNOWN_MISSING:
                continue
            name = f"{mon}{year % 100:02d}pub.zip"
            try:
                status = fetch(CPS_URL.format(year=year, mon=mon, yy=f"{year % 100:02d}"), RAW_CPS / name)
            except Exception as e:  # fail loudly at the end, not silently
                status = f"FAILED ({e})"
                failed.append(name)
            print(name, status, flush=True)

    if failed:
        sys.exit(f"{len(failed)} downloads failed: {failed}")


if __name__ == "__main__":
    main()
