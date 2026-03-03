import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


HDX_API_BASE = "https://data.humdata.org/api/3/action"


@dataclass(frozen=True)
class EventSpec:
    label: str
    country: str
    disaster_type: str
    year: int
    month: Optional[int] = None
    day: Optional[int] = None
    magnitude: Optional[float] = None


# Region 1 – Indonesia
# Historical (A1): 2018 Sulawesi Earthquake
EVENT_A1 = EventSpec(
    label="Indonesia_2018_09_28_M7.5",
    country="Indonesia",
    disaster_type="Earthquake",
    year=2018,
    month=9,
    day=28,
    magnitude=7.5,
)
# Recent (A2): 2024 West Java / Bandung Earthquake
EVENT_A2 = EventSpec(
    label="Indonesia_2024_09_M5.0",
    country="Indonesia",
    disaster_type="Earthquake",
    year=2024,
    month=9,
    magnitude=5.0,
)

# Region 2 – Afghanistan
# Historical (B1): 2015 Hindu Kush Earthquake
EVENT_B1 = EventSpec(
    label="Afghanistan_2015_10_26_M7.5",
    country="Afghanistan",
    disaster_type="Earthquake",
    year=2015,
    month=10,
    day=26,
    magnitude=7.5,
)
# Recent (B2): 2023 Herat Earthquakes
EVENT_B2 = EventSpec(
    label="Afghanistan_2023_10_07_M6.3",
    country="Afghanistan",
    disaster_type="Earthquake",
    year=2023,
    month=10,
    day=7,
    magnitude=6.3,
)

EVENTS: List[EventSpec] = [EVENT_A1, EVENT_A2, EVENT_B1, EVENT_B2]

"""
print(
    "Defining hdx_package_search: query the HDX CKAN API for datasets matching a query string."
)
"""


def hdx_package_search(
    session: requests.Session, query: str, rows: int = 10
) -> Dict[str, Any]:
    resp = session.get(
        f"{HDX_API_BASE}/package_search",
        params={"q": query, "rows": rows},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


"""
print(
    "Defining find_emdat_resource: locate a downloadable EM-DAT resource (CSV/XLSX) "
    "from HDX search results."
)
"""


def find_emdat_resource(pkg_search_json: Dict[str, Any]) -> Tuple[str, str, str]:
    results = pkg_search_json.get("result", {}).get("results", []) or []
    candidates: List[Tuple[str, str, str]] = []

    for pkg in results:
        ds_name = pkg.get("name") or pkg.get("title") or "unknown-dataset"
        for res in pkg.get("resources", []) or []:
            url = res.get("url")
            fmt = (res.get("format") or "").strip().upper()
            name = (res.get("name") or res.get("description") or "").lower()
            if not url or fmt not in {"CSV", "XLSX", "XLS"}:
                continue
            score = 0
            if "em-dat" in name or "emdat" in name:
                score += 2
            if "public" in name or "table" in name:
                score += 2
            if "disaster" in name:
                score += 1
            if fmt == "CSV":
                score += 1
            candidates.append((ds_name, url, fmt, score))

    if not candidates:
        raise RuntimeError("No EM-DAT CSV/XLSX resources found in HDX search results.")

    candidates.sort(key=lambda x: x[3], reverse=True)
    ds_name, url, fmt, _score = candidates[0]
    return ds_name, url, fmt


"""
print(
    "Defining download_emdat_table: download the chosen EM-DAT resource URL and load it into pandas."
)
"""


def download_emdat_table(session: requests.Session, url: str, fmt: str) -> pd.DataFrame:
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    content = resp.content
    if fmt.upper() == "CSV":
        return pd.read_csv(BytesIO(content), low_memory=False)
    return pd.read_excel(BytesIO(content))


"""
print(
    "Defining normalize_cols: normalize dataframe column names for robust matching."
)
"""


def normalize_cols(df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for c in df.columns:
        norm = re.sub(r"[^a-z0-9]+", " ", str(c).strip().lower()).strip()
        mapping[norm] = c
    return mapping


"""
print(
    "Defining pick_col: choose the first matching column by normalized name patterns."
)
"""


def pick_col(colmap: Dict[str, str], patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        pat_norm = re.sub(r"[^a-z0-9]+", " ", pat.strip().lower()).strip()
        for norm_name, original in colmap.items():
            if pat_norm in norm_name:
                return original
    return None


"""
print(
    "Defining select_emdat_row_for_event: filter the EM-DAT public table down to the best-matching "
    "row for a given event spec (country/date/type/magnitude)."
)
"""


def select_emdat_row_for_event(df: pd.DataFrame, event: EventSpec) -> pd.Series:
    colmap = normalize_cols(df)

    country_col = pick_col(colmap, ["country"])
    dtype_col = pick_col(colmap, ["disaster type", "disaster_type"])
    year_col = pick_col(colmap, ["start year", "year", "start_year"])
    month_col = pick_col(colmap, ["start month", "start_month"])
    day_col = pick_col(colmap, ["start day", "start_day"])
    mag_col = pick_col(colmap, ["magnitude"])

    if not country_col or not year_col:
        raise KeyError(
            f"Missing required columns. Found country={country_col}, year={year_col}"
        )

    mask = df[country_col].astype(str).str.contains(event.country, case=False, na=False)
    mask &= pd.to_numeric(df[year_col], errors="coerce") == event.year

    if dtype_col:
        mask &= (
            df[dtype_col]
            .astype(str)
            .str.contains(event.disaster_type, case=False, na=False)
        )
    if event.month is not None and month_col:
        mask &= pd.to_numeric(df[month_col], errors="coerce") == event.month
    if event.day is not None and day_col:
        mask &= pd.to_numeric(df[day_col], errors="coerce") == event.day

    subset = df.loc[mask].copy()
    if subset.empty:
        raise ValueError(
            f"No EM-DAT rows matched for {event.label} with filters applied."
        )

    if mag_col and event.magnitude is not None:
        mags = pd.to_numeric(subset[mag_col], errors="coerce")
        subset = subset.assign(_mag_diff=(mags - event.magnitude).abs())
        subset = subset.sort_values("_mag_diff", ascending=True)
        return subset.iloc[0].drop(labels=["_mag_diff"])

    affected_col = pick_col(colmap, ["total affected", "no affected", "affected"])
    if affected_col:
        vals = pd.to_numeric(subset[affected_col], errors="coerce").fillna(-1)
        subset = subset.assign(_aff=vals).sort_values("_aff", ascending=False)
        return subset.iloc[0].drop(labels=["_aff"])

    return subset.iloc[0]


"""
print(
    "Defining extract_impact_metrics: from a selected EM-DAT row, extract deaths, affected, "
    "damages, and any available magnitude-like and per-capita indicators into a standardized dict."
)
"""


def extract_impact_metrics(
    row: pd.Series, event: EventSpec, df_columns: List[str]
) -> Dict[str, Any]:
    colmap = normalize_cols(pd.DataFrame(columns=df_columns))

    deaths_col = pick_col(colmap, ["total deaths", "deaths"])
    affected_col = pick_col(colmap, ["total affected", "no affected", "affected"])
    damage_col = pick_col(
        colmap,
        [
            "total damage usd original",
            "total damage usd",
            "total damage",
            "total damages",
            "total damage ( 000 us",
            "total damage ('000 us",
        ],
    )
    events_col = pick_col(colmap, ["total events", "total_events"])

    def num(val) -> float:
        try:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return 0.0
            return float(str(val).replace(",", "").strip())
        except Exception:
            return 0.0

    deaths = num(row.get(deaths_col)) if deaths_col else 0.0
    affected = num(row.get(affected_col)) if affected_col else 0.0
    damage = num(row.get(damage_col)) if damage_col else 0.0
    magnitude = event.magnitude or 0.0
    total_events = num(row.get(events_col)) if events_col else 0.0

    if (
        damage_col
        and isinstance(damage_col, str)
        and "000" in damage_col.lower()
        and damage
        and damage < 1e9
    ):
        damage *= 1000.0

    population_exposed = affected
    econ_per_capita = damage / affected if affected > 0 else 0.0

    return {
        "event_label": event.label,
        "country": event.country,
        "start_year": event.year,
        "start_month": event.month,
        "start_day": event.day,
        "disaster_type": event.disaster_type,
        "magnitude": magnitude,
        "total_events_in_year_country": total_events,
        "total_deaths": deaths,
        "total_affected": affected,
        "population_exposed": population_exposed,
        "total_damages_usd": damage,
        "economic_damage_per_capita": econ_per_capita,
    }


"""
print(
    "Defining run_stage2_emdat_impact: end-to-end Stage 2 runner that downloads EM-DAT data "
    "via HDX API, selects the events, and writes emdat_events_impact.csv."
)
"""


def run_stage2_emdat_impact() -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )

    print("Searching HDX for an EM-DAT public table resource...")
    pkg_json = hdx_package_search(session, query="EM-DAT public table", rows=10)
    ds_name, res_url, res_fmt = find_emdat_resource(pkg_json)
    print(f"Selected HDX dataset: {ds_name}")
    print(f"Selected resource format: {res_fmt}")

    print("Downloading EM-DAT table from HDX and loading into pandas...")
    df = download_emdat_table(session, res_url, res_fmt)
    print(f"Loaded EM-DAT table: shape={df.shape}")

    metrics: List[Dict[str, Any]] = []
    for ev in EVENTS:
        print(f"Selecting row for event: {ev.label}...")
        row = select_emdat_row_for_event(df, ev)
        metrics.append(extract_impact_metrics(row, ev, list(df.columns)))

    out_df = pd.DataFrame(metrics)
    print("\nStage 2 EM-DAT impact metrics (from HDX/EM-DAT data):")
    print(out_df)

    out_path = "emdat_events_impact.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out_df


if __name__ == "__main__":
    print(
        "About to run Stage 2 (EM-DAT impact extraction) using HDX API: "
        "download EM-DAT data programmatically, then extract deaths/affected/damages for the four events."
    )
    run_stage2_emdat_impact()
