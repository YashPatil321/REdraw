"""Fetch Census ACS 5-year block group tables, TIGER/Line block groups, and LEHD LODES 8
(OD main, WAC, RAC, crosswalk) for California (spec 5.4). Cached in data/raw/.

Needs CENSUS_API_KEY in .env (free: https://api.census.gov/data/key_signup.html).

    python pipeline/fetch_census.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from pipeline.common import DataSourceUnavailable, download, load_env, log
from pipeline.config import raw_dir, region

STATE_FIPS = "06"  # California
STATE_ABBR = "ca"
COUNTY_FIPS = "073"  # San Diego County (the whole bbox lies inside it)


def years() -> tuple[str, str, str]:
    load_env()
    return os.environ.get("ACS_YEAR", "2022"), os.environ.get("LODES_VERSION", "LODES8"), os.environ.get("LODES_YEAR", "2021")


def acs_url(year: str, variables: list[str], key: str | None) -> str:
    get = ",".join(["NAME", *variables])
    url = f"https://api.census.gov/data/{year}/acs/acs5?get={get}&for=block%20group:*&in=state:{STATE_FIPS}&in=county:{COUNTY_FIPS}&in=tract:*"
    return url + (f"&key={key}" if key else "")


def tiger_bg_url(year: str) -> str:
    return f"https://www2.census.gov/geo/tiger/TIGER{year}/BG/tl_{year}_{STATE_FIPS}_bg.zip"


def lodes_urls(version: str, year: str) -> dict[str, str]:
    base = f"https://lehd.ces.census.gov/data/lodes/{version}/{STATE_ABBR}"
    return {
        "od_main": f"{base}/od/{STATE_ABBR}_od_main_JT00_{year}.csv.gz",
        "wac": f"{base}/wac/{STATE_ABBR}_wac_S000_JT00_{year}.csv.gz",
        "rac": f"{base}/rac/{STATE_ABBR}_rac_S000_JT00_{year}.csv.gz",
        "xwalk": f"{base}/{STATE_ABBR}_xwalk.csv.gz",
    }


def acs_rows_to_frame(rows: list[list[str]]) -> pd.DataFrame:
    """Census API JSON (header row + data rows) -> DataFrame with a 12-digit GEOID (unit tested)."""
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    for c in df.columns:
        if c.endswith("E") and c[0] == "B":
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] < 0, c] = float("nan")  # Census uses large negative sentinels for suppressed cells
    return df


def fetch_acs(raw: Path) -> Path:
    import httpx

    from pipeline.build_population import acs_variables

    year, _, _ = years()
    dest = raw / f"acs5_{year}_bg_{STATE_FIPS}{COUNTY_FIPS}.csv"
    if dest.exists():
        log(f"cached: {dest.name}")
        return dest
    key = os.environ.get("CENSUS_API_KEY") or None
    url = acs_url(year, acs_variables(), key)
    shown = acs_url(year, acs_variables(), "YOUR_KEY" if key else None)
    how = (
        "Set CENSUS_API_KEY in .env (free key: https://api.census.gov/data/key_signup.html) and retry, or open the URL "
        f"in a browser and save the JSON converted to CSV (with a GEOID column = state+county+tract+block group) as {dest.name}."
    )
    log(f"ACS: requesting {len(acs_variables())} variables for block groups in {STATE_FIPS}{COUNTY_FIPS}")
    try:
        r = httpx.get(url, timeout=120)
        r.raise_for_status()
        df = acs_rows_to_frame(r.json())
    except Exception as e:  # noqa: BLE001
        raise DataSourceUnavailable("US Census ACS 5-year (api.census.gov)", shown, dest, how, e) from e
    df.to_csv(dest, index=False)
    log(f"ACS: {len(df):,} block groups -> {dest.name}")
    return dest


def fetch_all(raw: Path | None = None) -> dict[str, Path]:
    raw = raw or raw_dir()
    raw.mkdir(parents=True, exist_ok=True)
    year, lver, lyear = years()
    out = {"acs": fetch_acs(raw)}
    tu = tiger_bg_url(year)
    out["tiger_bg"] = download(tu, raw / Path(tu).name, "US Census TIGER/Line block groups", f"Download {tu} in a browser and save it unchanged into data/raw/.")
    for k, u in lodes_urls(lver, lyear).items():
        out[k] = download(u, raw / Path(u).name, f"LEHD LODES ({k})", f"Download {u} from https://lehd.ces.census.gov/data/ and save it unchanged (still gzipped) into data/raw/.")
    return out


def load_block_groups(tiger_zip: Path) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    """Block groups intersecting the bbox (EPSG:32611) and the fraction of each BG area inside it."""
    b = region()["bbox"]
    bb = (b["west"], b["south"], b["east"], b["north"])
    bgs = gpd.read_file(f"zip://{tiger_zip}", bbox=bb)
    bgs = bgs[bgs["GEOID"].str.startswith(STATE_FIPS + COUNTY_FIPS)].to_crs("EPSG:32611")
    region_poly = gpd.GeoSeries([box(*bb)], crs="EPSG:4326").to_crs("EPSG:32611").iloc[0]
    frac = {}
    for g, geom in zip(bgs["GEOID"], bgs.geometry, strict=True):
        a = geom.area
        frac[str(g)] = float(geom.intersection(region_poly).area / a) if a > 0 else 0.0
    bgs = bgs[[f > 0 for f in frac.values()]]
    return bgs[["GEOID", "geometry"]].reset_index(drop=True), {k: v for k, v in frac.items() if v > 0}


def load_acs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"GEOID": str, "state": str, "county": str, "tract": str, "block group": str})
    df["GEOID"] = df["GEOID"].str.zfill(12)
    return df


def load_lodes(od_path: Path, xwalk_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OD rows whose home block is in the county (chunked; CA OD main is large) and the block crosswalk."""
    prefix = STATE_FIPS + COUNTY_FIPS
    parts = []
    for chunk in pd.read_csv(od_path, usecols=["w_geocode", "h_geocode", "S000"], dtype={"w_geocode": str, "h_geocode": str}, chunksize=2_000_000):
        parts.append(chunk[chunk["h_geocode"].str.zfill(15).str.startswith(prefix)])
    od = pd.concat(parts, ignore_index=True)
    xw = pd.read_csv(xwalk_path, usecols=["tabblk2020", "blklatdd", "blklondd"], dtype={"tabblk2020": str})
    return od, xw


if __name__ == "__main__":
    try:
        for k, v in fetch_all().items():
            print(k, v)
    except DataSourceUnavailable as err:
        print(err, file=sys.stderr)
        raise SystemExit(2) from None
