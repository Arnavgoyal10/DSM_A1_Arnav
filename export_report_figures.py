"""
Export dashboard figures to report_figures/ for inclusion in the final report.
Run from project root: python export_report_figures.py
Requires: pandas, plotly. For PNG export: pip install kaleido (optional; falls back to HTML).
"""
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
REPORT_FIGURES_DIR = BASE_DIR / "report_figures"
MEDIA_ENRICHED_CSV = BASE_DIR / "stage4_media_enriched.csv"
IMPACT_CSV = BASE_DIR / "emdat_events_impact.csv"
MEDIA_EVENTS_CSV = BASE_DIR / "reliefweb_media_events.csv"
EVENT_SUMMARY_JSON = BASE_DIR / "stage4_event_summary.json"


def load_data() -> Dict[str, Any]:
    """Load the same CSVs and JSON the dashboard uses (no Streamlit)."""
    media_df = pd.read_csv(
        MEDIA_ENRICHED_CSV, parse_dates=["publication_date", "event_date"]
    )
    impact_df = pd.read_csv(IMPACT_CSV)
    media_events_df = pd.read_csv(MEDIA_EVENTS_CSV, parse_dates=["publication_date"])
    if EVENT_SUMMARY_JSON.exists():
        summary = json.loads(EVENT_SUMMARY_JSON.read_text(encoding="utf-8"))
    else:
        summary = {}
    return {
        "media_enriched": media_df,
        "impact": impact_df,
        "media_events": media_events_df,
        "summary": summary,
    }


def build_dual_timeline_figure(
    media_enriched: pd.DataFrame, rolling_window: int = 3
) -> go.Figure:
    df = media_enriched.dropna(subset=["publication_date"]).copy()
    df["pub_date"] = df["publication_date"].dt.date
    daily = (
        df.groupby(["event_label", "pub_date"], as_index=False)
        .size()
        .rename(columns={"size": "daily_report_count"})
    )
    daily = daily.sort_values(["event_label", "pub_date"])
    if rolling_window and rolling_window > 1:
        daily["daily_report_count"] = daily.groupby("event_label")[
            "daily_report_count"
        ].transform(lambda s: s.rolling(rolling_window, min_periods=1).mean())
    fig = px.line(
        daily,
        x="pub_date",
        y="daily_report_count",
        color="event_label",
        markers=True,
        title="Dual Timeline – Daily News Volume by Event (3-day rolling average)",
        labels={"pub_date": "Date", "daily_report_count": "Number of reports"},
    )
    fig.update_layout(
        legend_title_text="Event",
        hovermode="x unified",
        margin=dict(l=40, r=30, t=80, b=40),
    )
    return fig


def build_resilience_radar_figure(
    impact_df: pd.DataFrame, media_events_df: pd.DataFrame
) -> go.Figure:
    coverage_counts = media_events_df["event_label"].value_counts().to_dict()
    metrics: List[Dict[str, Any]] = []
    for _, row in impact_df.iterrows():
        label = row["event_label"]
        metrics.append({
            "event_label": label,
            "Magnitude": float(row.get("magnitude") or 0),
            "Population exposure": float(row.get("total_affected") or 0),
            "Media coverage": float(coverage_counts.get(label, 0)),
            "Vulnerability proxy": float(row.get("economic_damage_per_capita") or 0),
        })
    radar_df = pd.DataFrame(metrics)
    if radar_df.empty:
        return go.Figure()
    axes = ["Magnitude", "Population exposure", "Media coverage", "Vulnerability proxy"]
    norm_df = radar_df.copy()
    for col in axes:
        col_max = norm_df[col].max()
        norm_df[col] = (norm_df[col] / col_max) if col_max > 0 else 0.0
    fig = go.Figure()
    for _, row in norm_df.iterrows():
        values = [row[c] for c in axes]
        values.append(values[0])
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=axes + [axes[0]],
                fill="toself",
                name=row["event_label"],
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Resilience Radar – Magnitude, Exposure, Coverage, Vulnerability",
    )
    return fig


def build_sentiment_over_time_figure(media_enriched: pd.DataFrame) -> go.Figure:
    if "headline_sentiment_compound" not in media_enriched.columns:
        return go.Figure()
    df = media_enriched.dropna(
        subset=["publication_date", "headline_sentiment_compound"]
    ).copy()
    if df.empty:
        return go.Figure()
    df["pub_date"] = df["publication_date"].dt.date
    daily = (
        df.groupby(["event_label", "pub_date"], as_index=False)["headline_sentiment_compound"]
        .mean()
        .rename(columns={"headline_sentiment_compound": "avg_sentiment"})
    )
    fig = px.line(
        daily,
        x="pub_date",
        y="avg_sentiment",
        color="event_label",
        markers=True,
        title="Average Headline Sentiment Over Time",
        labels={"pub_date": "Date", "avg_sentiment": "Avg sentiment (compound)"},
    )
    fig.update_layout(legend_title_text="Event", hovermode="x unified")
    return fig


# ---------- Report-only figures (not in dashboard) ----------


def build_fci_and_coverage_comparison_figure(
    impact_df: pd.DataFrame, media_events_df: pd.DataFrame
) -> go.Figure:
    """Bar chart: Forgotten Crisis Index and reports-per-USD damage by event."""
    counts = media_events_df["event_label"].value_counts()
    rows = []
    for _, row in impact_df.iterrows():
        label = row["event_label"]
        n_reports = int(counts.get(label, 0))
        affected = float(row.get("total_affected") or 0)
        damage_usd = float(row.get("total_damages_usd") or 1)
        fci = (n_reports / affected) if affected > 0 else 0
        reports_per_usd = (n_reports / damage_usd) if damage_usd > 0 else 0
        rows.append({
            "event_label": label,
            "FCI": fci,
            "Reports_per_USD_1e8": reports_per_usd * 1e8,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(name="FCI (reports per affected person)", x=df["event_label"], y=df["FCI"]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(name="Reports per USD damage (×1e-8)", x=df["event_label"], y=df["Reports_per_USD_1e8"]),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Event", tickangle=-25)
    fig.update_yaxes(title_text="FCI", secondary_y=False)
    fig.update_yaxes(title_text="Reports per USD (×1e-8)", secondary_y=True)
    fig.update_layout(
        title="Forgotten Crisis Index and Coverage per Dollar of Damage by Event",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(b=100),
    )
    return fig


def build_sentiment_distribution_figure(media_enriched: pd.DataFrame) -> go.Figure:
    """Box plot of headline sentiment compound by event (spread and center)."""
    if "headline_sentiment_compound" not in media_enriched.columns:
        return go.Figure()
    df = media_enriched.dropna(subset=["event_label", "headline_sentiment_compound"])[["event_label", "headline_sentiment_compound"]]
    if df.empty:
        return go.Figure()
    fig = px.box(
        df,
        x="event_label",
        y="headline_sentiment_compound",
        title="Headline Sentiment Distribution by Event (VADER compound)",
        labels={"event_label": "Event", "headline_sentiment_compound": "Sentiment (compound)"},
    )
    fig.update_layout(xaxis_tickangle=-25, margin=dict(b=100))
    return fig


def build_entity_breakdown_figure(summary: Dict[str, Any]) -> go.Figure:
    """Grouped bar: total NGO / government / private entity mention counts per event."""
    events = []
    ngo_totals = []
    gov_totals = []
    private_totals = []
    for ev_label, ev_data in (summary or {}).items():
        ec = ev_data.get("entity_classification") or {}
        ngo = sum(c for _, c in (ec.get("ngo") or []))
        gov = sum(c for _, c in (ec.get("government") or []))
        priv = sum(c for _, c in (ec.get("private") or []))
        events.append(ev_label)
        ngo_totals.append(ngo)
        gov_totals.append(gov)
        private_totals.append(priv)
    if not events:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="NGO", x=events, y=ngo_totals))
    fig.add_trace(go.Bar(name="Government", x=events, y=gov_totals))
    fig.add_trace(go.Bar(name="Private", x=events, y=private_totals))
    fig.update_layout(
        title="Entity Mention Counts by Type (NER + keyword classification)",
        barmode="group",
        xaxis_title="Event",
        yaxis_title="Total mentions",
        xaxis_tickangle=-25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(b=100),
    )
    return fig


def build_delta_t_comparison_figure(summary: Dict[str, Any]) -> go.Figure:
    """Bar chart: days from event date to media peak (ΔT) by event."""
    events = []
    delta_days = []
    for ev_label, ev_data in (summary or {}).items():
        temporal = ev_data.get("temporal") or {}
        dt = temporal.get("delta_t_days")
        if dt is not None:
            events.append(ev_label)
            delta_days.append(int(dt))
    if not events:
        return go.Figure()
    fig = go.Figure(go.Bar(x=events, y=delta_days, text=delta_days, textposition="outside"))
    fig.update_layout(
        title="Days from Event Date to Media Peak (ΔT)",
        xaxis_title="Event",
        yaxis_title="Days",
        xaxis_tickangle=-25,
        margin=dict(b=100),
    )
    return fig


def main():
    REPORT_FIGURES_DIR.mkdir(exist_ok=True)
    data = load_data()
    media_enriched = data["media_enriched"]
    if pd.api.types.is_datetime64tz_dtype(media_enriched["publication_date"]):
        media_enriched = media_enriched.copy()
        media_enriched["publication_date"] = media_enriched["publication_date"].dt.tz_localize(None)
    impact_df = data["impact"]
    media_events_df = data["media_events"]
    summary = data["summary"]

    figures = [
        ("dual_timeline", build_dual_timeline_figure(media_enriched)),
        ("resilience_radar", build_resilience_radar_figure(impact_df, media_events_df)),
        ("sentiment_over_time", build_sentiment_over_time_figure(media_enriched)),
        # Report-only (not in dashboard):
        ("fci_and_coverage_comparison", build_fci_and_coverage_comparison_figure(impact_df, media_events_df)),
        ("sentiment_distribution", build_sentiment_distribution_figure(media_enriched)),
        ("entity_breakdown", build_entity_breakdown_figure(summary)),
        ("delta_t_comparison", build_delta_t_comparison_figure(summary)),
    ]
    for name, fig in figures:
        if fig is None or len(fig.data) == 0:
            continue
        path_png = REPORT_FIGURES_DIR / f"{name}.png"
        path_html = REPORT_FIGURES_DIR / f"{name}.html"
        try:
            fig.write_image(str(path_png), scale=2)
            print(f"Wrote {path_png}")
        except Exception as e:
            print(f"PNG export failed ({e}); writing HTML only.")
        fig.write_html(str(path_html))
        print(f"Wrote {path_html}")
    print(f"Done. Figures in {REPORT_FIGURES_DIR}")


if __name__ == "__main__":
    main()
