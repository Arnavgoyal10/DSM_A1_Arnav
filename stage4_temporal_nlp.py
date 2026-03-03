import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
MEDIA_CSV = DATA_DIR / "reliefweb_media_events.csv"
IMPACT_CSV = DATA_DIR / "emdat_events_impact.csv"

OUT_ENRICHED = DATA_DIR / "stage4_media_enriched.csv"
OUT_ENTITIES = DATA_DIR / "stage4_media_entities.csv"
OUT_EVENT_SUMMARY = DATA_DIR / "stage4_event_summary.json"

"""
print(
    "Defining load_inputs: read the Stage 2 impact CSV and Stage 3 ReliefWeb media CSV into pandas, "
    "and normalize key columns for downstream analysis."
)
"""


def load_inputs(
    media_path: Path, impact_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_media = pd.read_csv(media_path)
    df_impact = pd.read_csv(impact_path)

    if "publication_date" in df_media.columns:
        df_media["publication_date"] = pd.to_datetime(
            df_media["publication_date"], errors="coerce", utc=True
        )

    if "event_date" in df_media.columns:
        df_media["event_date"] = pd.to_datetime(
            df_media["event_date"], errors="coerce", utc=True
        )

    if "event_label" not in df_media.columns:
        raise KeyError("media CSV missing required column: event_label")
    if "event_label" not in df_impact.columns:
        raise KeyError("impact CSV missing required column: event_label")

    return df_media, df_impact


"""
print(
    "Defining compute_temporal_features: compute days_since_event for each report, build daily "
    "volume series per event, find media peak date, and compute ΔT (days) per event."
)
"""


def compute_temporal_features(
    df_media: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df_media.copy()
    df = df.dropna(subset=["publication_date", "event_date"])

    df["pub_date"] = df["publication_date"].dt.date
    df["event_date_only"] = df["event_date"].dt.date
    df["days_since_event"] = (
        df["publication_date"] - df["event_date"]
    ).dt.total_seconds() / 86400.0

    daily = (
        df.groupby(["event_label", "pub_date"], as_index=False)
        .size()
        .rename(columns={"size": "daily_report_count"})
    )

    peak_by_event: Dict[str, Dict[str, Any]] = {}
    for event_label, g in daily.groupby("event_label"):
        g_sorted = g.sort_values(
            ["daily_report_count", "pub_date"], ascending=[False, True]
        )
        peak_row = g_sorted.iloc[0]
        media_peak_date = peak_row["pub_date"]

        event_dates = (
            df.loc[df["event_label"] == event_label, "event_date_only"]
            .dropna()
            .unique()
        )
        event_date_only = min(event_dates) if len(event_dates) else None

        delta_t_days = None
        if event_date_only is not None:
            delta_t_days = (
                pd.to_datetime(media_peak_date) - pd.to_datetime(event_date_only)
            ).days

        peak_by_event[event_label] = {
            "event_date": str(event_date_only) if event_date_only is not None else None,
            "media_peak_date": str(media_peak_date),
            "media_peak_daily_volume": int(peak_row["daily_report_count"]),
            "delta_t_days": int(delta_t_days) if delta_t_days is not None else None,
        }

    df = df.merge(daily, on=["event_label", "pub_date"], how="left")

    return df, peak_by_event


"""
print(
    "Defining ensure_spacy_model: load a spaCy English model for NER; if missing, download it "
    "programmatically and then load it."
)
"""


def ensure_spacy_model(model_name: str = "en_core_web_sm"):
    import spacy

    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli import download

        print(f"spaCy model '{model_name}' not found. Downloading now...")
        download(model_name)
        return spacy.load(model_name)


"""
print(
    "Defining extract_entities: run spaCy NER over headline+summary and return per-report entities "
    "(ORG/GPE + numeric-like entities) for later aggregation."
)
"""


def extract_entities(nlp, text: str) -> List[Tuple[str, str]]:
    doc = nlp(text)
    ents: List[Tuple[str, str]] = []
    for ent in doc.ents:
        if ent.label_ in {
            "ORG",
            "GPE",
            "NORP",
            "MONEY",
            "CARDINAL",
            "QUANTITY",
            "PERCENT",
        }:
            ents.append((ent.text.strip(), ent.label_))
    return ents


NGO_KEYWORDS = {
    "red cross",
    "ifrc",
    "icrc",
    "unicef",
    "unhcr",
    "wfp",
    "ocha",
    "who",
    "care",
    "save the children",
    "mercy corps",
    "world vision",
    "oxfam",
    "actionaid",
    "aha centre",
    "aha",
    "undp",
    "unfpa",
    "iom",
    "gfdrr",
    "cerf",
    "echo",
    "direct relief",
    "mapaction",
    "act alliance",
    "icrisat",
    "cws",
    "pwj",
    "peace winds",
    "taiwanicdf",
    "yappika",
    "islamic relief",
    "british red cross",
    "american red cross",
    "norcap",
    "insarag",
    "idmc",
    "nrc",
    "tufts",
    "feinstein",
    "nts centre",
    "ecw",
    "uNAIDS",
    "unesco",
    "undrr",
}
GOVT_KEYWORDS = {
    "government",
    "bnpb",
    "bmkg",
    "usgs",
    "gdacs",
    "copernicus",
    "pdc",
    "asean",
    "national board",
    "ministry",
    "bappenas",
    "pusdalops",
    "local government",
    "provincial",
    "regency",
    "district",
}
PRIVATE_KEYWORDS = {
    "private",
    "company",
    "corporation",
    "ltd",
    "inc",
    "llc",
}


def classify_entity_type(ent_text: str) -> Optional[str]:
    if not ent_text or not isinstance(ent_text, str):
        return None
    t = ent_text.lower().strip()
    if any(k in t for k in NGO_KEYWORDS):
        return "ngo"
    if any(k in t for k in GOVT_KEYWORDS):
        return "government"
    if any(k in t for k in PRIVATE_KEYWORDS):
        return "private"
    if len(t) > 3 and t not in {
        "pdf",
        "mb",
        "utm",
        "url",
        "format news",
        "press release",
    }:
        pass
    return None


DEATHS_PATTERN = re.compile(
    r"\b(\d[\d,\.]*)\s*(people|persons?|dead|killed|fatalities?)\b"
    r"|"
    r"\b(deaths?|killed|fatalities?|casualties?)\b[^\d]{0,20}(\d[\d,\.]*)",
    re.IGNORECASE,
)

LOSS_CONTEXT = re.compile(
    r"damage|destroyed|loss|affected|displaced|injured|damaged|housing|homes|"
    r"economic\s+loss|total\s+damage|usd|dollars?\$",
    re.IGNORECASE,
)
FUND_CONTEXT = re.compile(
    r"fund|donation|appeal|relief\s+(?:fund|package)|pledge|allocat|humanitarian\s+aid|"
    r"emergency\s+fund|cerf|grant",
    re.IGNORECASE,
)


def extract_impact_from_text(text: str) -> Dict[str, List[str]]:

    out: Dict[str, List[str]] = {"deaths": [], "losses": [], "relief_funds": []}
    if not text or not isinstance(text, str):
        return out
    text = " " + text + " "

    for m in DEATHS_PATTERN.finditer(text):
        g = m.group(0)
        num = re.search(r"\d+(?:\.\d+)?(?:\s*,\d{3})*", g)
        if num:
            out["deaths"].append(num.group(0).replace(",", "").strip())

    for m in re.finditer(
        r"(\d+(?:\.\d+)?(?:\s*,\d{3})*(?:\s*(?:million|billion|m|bn|k))?\s*(?:usd|\$)?|\$\s*\d+(?:\.\d+)?(?:\s*,\d{3})*)",
        text,
        re.IGNORECASE,
    ):
        start = max(0, m.start() - 120)
        end = min(len(text), m.end() + 120)
        window = text[start:end]
        num_raw = re.search(r"\d+(?:\.\d+)?(?:\s*,\d{3})*", m.group(0))
        if not num_raw:
            continue
        num_str = num_raw.group(0).replace(",", "").replace(" ", "").strip()
        try:
            is_big = float(num_str) > 100
        except ValueError:
            is_big = False
        if (
            re.search(r"million|billion|m\b|bn\b|\$|usd", m.group(0), re.IGNORECASE)
            or is_big
        ):
            if FUND_CONTEXT.search(window):
                out["relief_funds"].append(num_str)
            elif LOSS_CONTEXT.search(window):
                out["losses"].append(num_str)
    return out


"""
print(
    "Defining ensure_vader: initialize NLTK VADER sentiment analyzer, downloading the lexicon "
    "programmatically if needed."
)
"""


def ensure_vader():
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        print("NLTK VADER lexicon not found. Downloading vader_lexicon now...")
        nltk.download("vader_lexicon")
        return SentimentIntensityAnalyzer()


"""
print(
    "Defining compute_sentiment: compute VADER sentiment scores for a headline string "
    "and return the compound score."
)
"""


def compute_sentiment(vader, headline: str) -> float:
    if not isinstance(headline, str) or not headline.strip():
        return 0.0
    return float(vader.polarity_scores(headline)["compound"])


"""
print(
    "Defining run_stage4: orchestrate Stage 4 by computing temporal features, NER entities, "
    "and sentiment volatility, then write enriched per-report CSVs and an event-level summary."
)
"""


def run_stage4() -> Dict[str, Any]:
    if not MEDIA_CSV.exists():
        raise FileNotFoundError(f"Missing media CSV: {MEDIA_CSV}")
    if not IMPACT_CSV.exists():
        raise FileNotFoundError(f"Missing impact CSV: {IMPACT_CSV}")

    df_media, df_impact = load_inputs(MEDIA_CSV, IMPACT_CSV)

    # NEW LOGGING: Dynamically detect and announce the events found
    unique_events = df_media["event_label"].dropna().unique()
    print(f"\nFound {len(unique_events)} unique events in {MEDIA_CSV.name}:")
    for ev in unique_events:
        print(f"  - {ev}")
    print("-" * 60)

    # Temporal
    df_enriched, peak_by_event = compute_temporal_features(df_media)

    # Sentiment
    vader = ensure_vader()
    df_enriched["headline_sentiment_compound"] = df_enriched["headline"].apply(
        lambda s: compute_sentiment(vader, s)
    )

    # NER
    nlp = ensure_spacy_model()

    entity_rows: List[Dict[str, Any]] = []
    org_counts: Dict[str, Counter] = defaultdict(Counter)
    num_counts: Dict[str, Counter] = defaultdict(Counter)
    ngo_counts: Dict[str, Counter] = defaultdict(Counter)
    govt_counts: Dict[str, Counter] = defaultdict(Counter)
    private_counts: Dict[str, Counter] = defaultdict(Counter)
    deaths_by_event: Dict[str, List[str]] = defaultdict(list)
    losses_by_event: Dict[str, List[str]] = defaultdict(list)
    relief_funds_by_event: Dict[str, List[str]] = defaultdict(list)

    print("\nRunning NLP extraction (NER & Sentiment) on all reports...")
    for idx, row in df_enriched.iterrows():
        headline = str(row.get("headline", "") or "")
        summary = str(row.get("summary", "") or "")
        text = (headline + " " + summary).strip()

        impact = extract_impact_from_text(text)
        el = row["event_label"]
        deaths_by_event[el].extend(impact["deaths"])
        losses_by_event[el].extend(impact["losses"])
        relief_funds_by_event[el].extend(impact["relief_funds"])

        ents = extract_entities(nlp, text)
        for ent_text, ent_label in ents:
            entity_rows.append(
                {
                    "event_label": row["event_label"],
                    "url": row.get("url"),
                    "headline": headline,
                    "entity_text": ent_text,
                    "entity_label": ent_label,
                }
            )

            if ent_label in {"ORG", "GPE", "NORP"}:
                org_counts[row["event_label"]][ent_text] += 1
                etype = classify_entity_type(ent_text)
                if etype == "ngo":
                    ngo_counts[row["event_label"]][ent_text] += 1
                elif etype == "government":
                    govt_counts[row["event_label"]][ent_text] += 1
                elif etype == "private":
                    private_counts[row["event_label"]][ent_text] += 1
            if ent_label in {"MONEY", "CARDINAL", "QUANTITY", "PERCENT"}:
                if re.search(r"\d", ent_text) or ent_label == "MONEY":
                    num_counts[row["event_label"]][ent_text] += 1

    df_entities = pd.DataFrame(entity_rows)

    sentiment_summary: Dict[str, Any] = {}
    for event_label, g in df_enriched.groupby("event_label"):
        scores = g["headline_sentiment_compound"].dropna()
        sentiment_summary[event_label] = {
            "headline_sentiment_mean": float(scores.mean()) if len(scores) else None,
            "headline_sentiment_std": float(scores.std()) if len(scores) else None,
            "headline_sentiment_n": int(len(scores)),
        }

    impact_by_event = (
        df_impact.set_index("event_label").to_dict(orient="index")
        if not df_impact.empty
        else {}
    )

    def _summarize_extracted(values: List[str], top_n: int = 15) -> Dict[str, Any]:
        c = Counter(values)
        return {
            "total_mentions": len(values),
            "unique_values": len(c),
            "top_values": c.most_common(top_n),
        }

    event_summary: Dict[str, Any] = {}
    for event_label in sorted(set(df_enriched["event_label"].unique())):
        event_summary[event_label] = {
            "temporal": peak_by_event.get(event_label),
            "sentiment": sentiment_summary.get(event_label),
            "top_entities": {
                "org_like_top20": org_counts[event_label].most_common(20),
                "numeric_like_top20": num_counts[event_label].most_common(20),
            },
            "entity_classification": {
                "ngo": ngo_counts[event_label].most_common(15),
                "government": govt_counts[event_label].most_common(15),
                "private": private_counts[event_label].most_common(15),
            },
            "extracted_impact": {
                "deaths": _summarize_extracted(deaths_by_event[event_label]),
                "losses": _summarize_extracted(losses_by_event[event_label]),
                "relief_funds": _summarize_extracted(
                    relief_funds_by_event[event_label]
                ),
            },
            "impact": impact_by_event.get(event_label),
        }

    df_enriched.to_csv(OUT_ENRICHED, index=False)
    df_entities.to_csv(OUT_ENTITIES, index=False)
    OUT_EVENT_SUMMARY.write_text(json.dumps(event_summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Wrote enriched media CSV: {OUT_ENRICHED}")
    print(f"Wrote entities CSV: {OUT_ENTITIES}")
    print(f"Wrote event summary JSON: {OUT_EVENT_SUMMARY}")
    print("=" * 60)

    return event_summary


if __name__ == "__main__":
    print("\nSTAGE 4: Temporal Analysis & NLP (ΔT, NER, Sentiment)")
    run_stage4()
