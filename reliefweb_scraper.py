import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus


@dataclass(frozen=True)
class EventSpec:
    label: str
    name: str
    date: str  # ISO string
    location: str
    hazard_type: str


# Region 1 – Indonesia
EVENT_A1 = EventSpec(
    label="Indonesia_2018_09_28_M7.5",
    name="2018 Central Sulawesi Earthquake and Tsunami",
    date="2018-09-28",
    location="Indonesia",
    hazard_type="Earthquake",
)
EVENT_A2 = EventSpec(
    label="Indonesia_2024_09_M5.0",
    name="2024 West Java / Bandung Earthquake",
    date="2024-09-01",  # approximate date in Sept 2024
    location="Indonesia",
    hazard_type="Earthquake",
)

# Region 2 – Myanmar
EVENT_B1 = EventSpec(
    label="Myanmar_2016_08_24_M6.8",
    name="2016 Chauk / Bagan Earthquake",
    date="2016-08-24",
    location="Myanmar",
    hazard_type="Earthquake",
)
EVENT_B2 = EventSpec(
    label="Myanmar_2025_03_M7.7",
    name="2025 Mandalay Earthquake",
    date="2025-03-01",  # approximate month; exact day unknown
    location="Myanmar",
    hazard_type="Earthquake",
)

EVENTS: List[EventSpec] = [EVENT_A1, EVENT_A2, EVENT_B1, EVENT_B2]
EVENT_MAP: Dict[str, EventSpec] = {e.label: e for e in EVENTS}

TARGET_REPORTS_PER_EVENT = 500
RELIEFWEB_BASE = "https://reliefweb.int"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
)

"""
print(
    "Defining scrape_reliefweb_list: paginate ReliefWeb's web UI for an event-specific "
    "query, returning unique report URLs and basic metadata via pure HTML scraping."
)
"""


def scrape_reliefweb_list(event: EventSpec, target_count: int) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    query = quote_plus(f"{event.hazard_type} {event.location}")
    page = 0
    max_pages = 80

    while len(reports) < target_count and page < max_pages:
        list_url = f"{RELIEFWEB_BASE}/updates?view=reports&search={query}&page={page}"
        try:
            resp = SESSION.get(list_url, timeout=20)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.content, "html.parser")

            links = soup.find_all("a", href=True)
            page_links: List[Tuple[str, str]] = []
            for a in links:
                href = a["href"]
                if "/report/" not in href:
                    continue
                url = href if href.startswith("http") else RELIEFWEB_BASE + href
                title = (a.get_text(strip=True) or "").strip()
                page_links.append((url, title))

            if not page_links:
                break

            for url, list_title in page_links:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                reports.append(
                    {
                        "url": url,
                        "list_title": list_title,
                        "event_label": event.label,
                        "event_name": event.name,
                        "event_date": event.date,
                    }
                )
                if len(reports) >= target_count:
                    break

            page += 1
            time.sleep(0.3)
        except Exception as e:
            print(f"  Error scraping ReliefWeb list page {page} for {event.label}: {e}")
            break

    return reports[:target_count]


"""
print(
    "Defining scrape_reliefweb_report: fetch a single ReliefWeb report page and "
    "extract headline, publication date, and a short text summary."
)
"""


def scrape_reliefweb_report(
    url: str, event: EventSpec, list_title: str | None = None
) -> Dict[str, Any]:
    try:
        resp = requests.get(url, headers=SESSION.headers, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")
        doc = BeautifulSoup(resp.content, "html.parser")

        title_el = doc.find("h1") or doc.find("title")
        headline = (
            title_el.get_text(strip=True) if title_el else list_title or ""
        ).strip()
        if not headline:
            headline = "N/A"

        pub_date = ""
        time_el = doc.find("time")
        if time_el:
            pub_date = time_el.get("datetime") or time_el.get_text(strip=True) or ""

        body = ""
        candidates = doc.find_all(
            lambda tag: tag.name in {"article", "section", "div"}
            and tag.get("class")
            and any(
                key in " ".join(tag.get("class", [])).lower()
                for key in ("body", "content", "article", "summary")
            )
        )
        if candidates:
            body = max(
                candidates, key=lambda el: len(el.get_text(strip=True))
            ).get_text(separator=" ", strip=True)
        else:
            body = doc.get_text(separator=" ", strip=True)

        summary = (body or "N/A")[:500]

        return {
            "report_id": f"rw_{hash(url) & 0xFFFFFFFF}",
            "headline": headline,
            "publication_date": pub_date,
            "url": url,
            "summary": summary,
            "event_label": event.label,
            "event_name": event.name,
            "event_date": event.date,
        }
    except Exception:
        # On failure (e.g. HTTP 429), return a stub row without printing noisy errors.
        return {
            "report_id": f"rw_{hash(url) & 0xFFFFFFFF}",
            "headline": list_title or "N/A",
            "publication_date": "",
            "url": url,
            "summary": "N/A",
            "event_label": event.label,
            "event_name": event.name,
            "event_date": event.date,
        }


"""
print(
    "Defining collect_reliefweb_media_for_event: orchestrate list and per-report "
    "scraping to collect up to N real reports for a single event."
)
"""


def collect_reliefweb_media_for_event(
    event: EventSpec, target_count: int
) -> List[Dict[str, Any]]:
    print(f"\n{'=' * 60}")
    print(f"Collecting ReliefWeb media for event: {event.name}")
    print(f"Target reports: {target_count}")
    print(f"{'=' * 60}")

    list_entries = scrape_reliefweb_list(event, target_count * 2)
    print(f"  Found {len(list_entries)} candidate report links in list pages.")

    reports: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    max_workers = 24
    tasks = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for entry in list_entries:
            url = entry["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            fut = executor.submit(
                scrape_reliefweb_report, url, event, entry.get("list_title")
            )
            tasks[fut] = url

        for fut in as_completed(tasks):
            try:
                report = fut.result()
                reports.append(report)
                if len(reports) % 50 == 0:
                    print(f"  Scraped {len(reports)} full reports for {event.label}...")
                if len(reports) >= target_count:
                    break
            except Exception:
                # Any unexpected worker error is swallowed; failures already return stub rows.
                continue

    print(f"Total ReliefWeb reports collected for {event.label}: {len(reports)}")
    return reports


"""
print(
    "Defining build_media_dataframe: merge per-event report lists, clean duplicates, "
    "parse dates, and save reliefweb_media_events.csv."
)
"""


def build_media_dataframe(
    reports_per_event: List[List[Dict[str, Any]]],
) -> pd.DataFrame:
    all_reports: List[Dict[str, Any]] = []
    for chunk in reports_per_event:
        all_reports.extend(chunk)
    if not all_reports:
        cols = [
            "report_id",
            "headline",
            "publication_date",
            "url",
            "summary",
            "event_label",
            "event_name",
            "event_date",
        ]
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(all_reports)
        df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
        df = df.drop_duplicates(subset=["headline", "url"], keep="first")
        df = df.sort_values("publication_date").reset_index(drop=True)

    # Merge with any existing file instead of always overwriting,
    # so that partial re-runs for individual events accumulate.
    out_path = Path("reliefweb_media_events.csv")
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path)
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(
                subset=["event_label", "headline", "url"], keep="first"
            ).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"\nSaved merged ReliefWeb media CSV to {out_path} with shape={df.shape}")
    return df


"""
print(
    "Defining run_stage3_reliefweb_media: full Stage 3 runner that scrapes ReliefWeb "
    "for all configured events and writes the merged CSV."
)
"""


def run_stage3_reliefweb_media(
    selected_events: List[EventSpec] | None = None,
) -> pd.DataFrame:
    events = selected_events or EVENTS
    reports_per_event: List[List[Dict[str, Any]]] = []
    for ev in events:
        reports = collect_reliefweb_media_for_event(ev, TARGET_REPORTS_PER_EVENT)
        reports_per_event.append(reports)
    df = build_media_dataframe(reports_per_event)

    print("\nReliefWeb media DataFrame preview:")
    print(df.head())
    print(f"\nCounts by event_label:\n{df['event_label'].value_counts()}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3 – ReliefWeb media scraping. "
            "By default scrapes all configured events; use --events to restrict."
        )
    )
    parser.add_argument(
        "--events",
        type=str,
        help=(
            "Comma-separated list of event_label values to scrape "
            "(e.g. 'Indonesia_2018_09_28_M7.5,Myanmar_2016_08_24_M6.8'). "
            "Default: all four events."
        ),
    )
    args = parser.parse_args()

    if args.events:
        labels = [s.strip() for s in args.events.split(",") if s.strip()]
        selected: List[EventSpec] = []
        for lab in labels:
            ev = EVENT_MAP.get(lab)
            if ev is None:
                print(
                    f"Warning: unknown event_label '{lab}' – skipping.", file=sys.stderr
                )
            else:
                selected.append(ev)
        if not selected:
            print(
                "No valid event_labels provided; defaulting to all events.",
                file=sys.stderr,
            )
            selected = EVENTS
    else:
        selected = EVENTS

    print(
        "\nSTAGE 3: Web Scraping – Media & Response Data (ReliefWeb, no API; web HTML only)"
    )
    print("Events to scrape:")
    for ev in selected:
        print(f"  - {ev.label} ({ev.name})")

    run_stage3_reliefweb_media(selected)
