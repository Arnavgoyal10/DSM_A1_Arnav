import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

MEDIA_ENRICHED_CSV = BASE_DIR / "stage4_media_enriched.csv"
IMPACT_CSV = BASE_DIR / "emdat_events_impact.csv"
MEDIA_EVENTS_CSV = BASE_DIR / "reliefweb_media_events.csv"
EVENT_SUMMARY_JSON = BASE_DIR / "stage4_event_summary.json"


@st.cache_data
def load_data() -> Dict[str, Any]:
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


def build_multi_event_timeline_figure(
    media_enriched: pd.DataFrame,
    events: List[str] | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    rolling_window: int = 3,
) -> go.Figure:
    df = media_enriched.copy()
    df = df.dropna(subset=["publication_date"])
    if events:
        df = df[df["event_label"].isin(events)]
    if date_range:
        start_date, end_date = date_range
        df = df[
            (df["publication_date"] >= start_date)
            & (df["publication_date"] <= end_date)
        ]

    df["pub_date"] = df["publication_date"].dt.date

    daily = (
        df.groupby(["event_label", "pub_date"], as_index=False)
        .size()
        .rename(columns={"size": "daily_report_count"})
    )

    daily = daily.sort_values(["event_label", "pub_date"])
    subtitle = ""
    if rolling_window and rolling_window > 1:
        daily["daily_report_count"] = daily.groupby("event_label")[
            "daily_report_count"
        ].transform(lambda s: s.rolling(rolling_window, min_periods=1).mean())
        subtitle = f" ({rolling_window}-day rolling average)"

    fig = px.line(
        daily,
        x="pub_date",
        y="daily_report_count",
        color="event_label",
        markers=True,
        title=f"Multi-Event Timeline – Daily News Volume{subtitle}",
        labels={
            "pub_date": "Date",
            "daily_report_count": "Number of reports",
            "event_label": "Event",
        },
    )
    fig.update_layout(
        legend_title_text="Event",
        hovermode="x unified",
        margin=dict(l=40, r=30, t=80, b=40),
        xaxis_title="Date",
        yaxis_title="Daily reports",
    )
    return fig


def build_resilience_radar_figure(
    impact_df: pd.DataFrame,
    media_events_df: pd.DataFrame,
    events: List[str] | None = None,
) -> go.Figure:
    if events:
        impact_df = impact_df[impact_df["event_label"].isin(events)]
        media_events_df = media_events_df[media_events_df["event_label"].isin(events)]

    coverage_counts = media_events_df["event_label"].value_counts().to_dict()

    metrics: List[Dict[str, Any]] = []
    for _, row in impact_df.iterrows():
        label = row["event_label"]
        magnitude = row.get("magnitude", None)
        population_exposed = row.get("total_affected", None)
        total_reports = coverage_counts.get(label, 0)
        econ_per_capita = row.get("economic_damage_per_capita", None)

        metrics.append(
            {
                "event_label": label,
                "Magnitude": float(magnitude) if pd.notna(magnitude) else 0.0,
                "Population exposure": (
                    float(population_exposed) if pd.notna(population_exposed) else 0.0
                ),
                "Media coverage": float(total_reports),
                "Vulnerability proxy": (
                    float(econ_per_capita) if pd.notna(econ_per_capita) else 0.0
                ),
            }
        )

    radar_df = pd.DataFrame(metrics)
    if radar_df.empty:
        return go.Figure()

    axes = ["Magnitude", "Population exposure", "Media coverage", "Vulnerability proxy"]
    norm_df = radar_df.copy()
    for col in axes:
        col_max = norm_df[col].max()
        if col_max > 0:
            norm_df[col] = norm_df[col] / col_max
        else:
            norm_df[col] = 0.0

    categories = axes
    fig = go.Figure()
    for _, row in norm_df.iterrows():
        values = [row[c] for c in categories]
        values.append(values[0])
        cats_closed = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=cats_closed,
                fill="toself",
                name=row["event_label"],
                opacity=0.6,  # Lowered opacity so overlapping shapes are visible
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Resilience Radar – Cross-Regional Comparison",
    )
    return fig


def build_event_volume_bar_figure(media_enriched: pd.DataFrame) -> go.Figure:
    """Bar chart of media volume per event for the current filter window."""
    if media_enriched.empty:
        return go.Figure()

    counts = (
        media_enriched.groupby("event_label", as_index=False)
        .size()
        .rename(columns={"size": "num_reports"})
    )
    counts = counts.sort_values("num_reports", ascending=False)

    fig = px.bar(
        counts,
        x="event_label",
        y="num_reports",
        color="event_label",
        title="Media Volume by Event (filtered window)",
        labels={"event_label": "Event", "num_reports": "Number of reports"},
    )
    fig.update_layout(
        xaxis_tickangle=-25, margin=dict(l=40, r=10, t=60, b=120), showlegend=False
    )
    return fig


def build_sentiment_over_time_figure(media_enriched: pd.DataFrame) -> go.Figure:
    """Line chart of average headline sentiment over time by event."""
    if (
        media_enriched.empty
        or "headline_sentiment_compound" not in media_enriched.columns
    ):
        return go.Figure()

    df = media_enriched.dropna(
        subset=["publication_date", "headline_sentiment_compound"]
    ).copy()
    if df.empty:
        return go.Figure()

    df["pub_date"] = df["publication_date"].dt.date
    daily = (
        df.groupby(["event_label", "pub_date"], as_index=False)[
            "headline_sentiment_compound"
        ]
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
        labels={
            "pub_date": "Date",
            "avg_sentiment": "Avg sentiment (compound)",
            "event_label": "Event",
        },
    )
    fig.update_layout(
        legend_title_text="Event",
        hovermode="x unified",
        margin=dict(l=40, r=30, t=80, b=40),
        xaxis_title="Date",
        yaxis_title="Sentiment (compound)",
    )
    return fig


def _inject_custom_css() -> None:
    """Light-touch theming to make the dashboard feel more polished."""
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f5f7;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        section[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }
        /* Make selectbox text legible against white background in the sidebar */
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
            color: #000000 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox svg {
            color: #000000 !important;
            fill: #000000 !important;
        }
        h1, h2, h3 {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Disaster Pulse Dashboard", layout="wide")
    _inject_custom_css()

    st.title("Disaster Pulse Dashboard")
    st.markdown(
        """
This dashboard compares **four major seismic events** across Indonesia and Myanmar 
to analyze how media response aligns with actual human and economic impact.
- **Historical:** Indonesia 2018 (M7.5) & Myanmar 2016 (M6.8)
- **Recent:** Indonesia 2024 (M5.0) & Myanmar 2025 (M7.7)
"""
    )

    data = load_data()
    media_enriched = data["media_enriched"]
    if pd.api.types.is_datetime64tz_dtype(media_enriched["publication_date"]):
        media_enriched["publication_date"] = media_enriched[
            "publication_date"
        ].dt.tz_localize(None)
    impact_df = data["impact"]
    media_events_df = data["media_events"]
    summary = data["summary"]

    # Sidebar – primary event selection and rich textual details.
    st.sidebar.title("Event Details")
    event_labels = impact_df["event_label"].tolist()
    selected = st.sidebar.selectbox(
        "Select event to view deep dive details", event_labels
    )

    st.sidebar.markdown("**Impact metrics (EM-DAT/HDX):**")
    impact_row = impact_df.set_index("event_label").loc[selected].to_dict()
    for k, v in impact_row.items():
        st.sidebar.write(f"- {k}: {v}")

    if summary and selected in summary:
        st.sidebar.markdown("**Temporal & Sentiment Summary:**")
        ev = summary[selected]
        temporal = ev.get("temporal") or {}
        sentiment = ev.get("sentiment") or {}
        st.sidebar.write(f"- Event date: {temporal.get('event_date')}")
        st.sidebar.write(f"- Media peak date: {temporal.get('media_peak_date')}")
        st.sidebar.write(f"- ΔT (days): {temporal.get('delta_t_days')}")
        st.sidebar.write(
            f"- Mean headline sentiment: {sentiment.get('headline_sentiment_mean')}"
        )
        st.sidebar.write(
            f"- Sentiment volatility (std): {sentiment.get('headline_sentiment_std')}"
        )

        st.sidebar.markdown("**Top organizations/entities:**")
        top_orgs = (ev.get("top_entities") or {}).get("org_like_top20") or []
        for text, count in top_orgs[:10]:
            st.sidebar.write(f"- {text} ({count} mentions)")

        ec = ev.get("entity_classification") or {}
        if ec.get("ngo") or ec.get("government") or ec.get("private"):
            st.sidebar.markdown("**Entities by type (NGO / Govt / Private):**")
            for label, key in [
                ("NGOs", "ngo"),
                ("Government", "government"),
                ("Private", "private"),
            ]:
                items = ec.get(key) or []
                if items:
                    st.sidebar.caption(label)
                    for text, count in items[:5]:
                        st.sidebar.write(f"  - {text} ({count})")

        ex = ev.get("extracted_impact") or {}
        if ex.get("deaths") or ex.get("losses") or ex.get("relief_funds"):
            st.sidebar.markdown("**Extracted from text (deaths / losses / funds):**")
            for key, title in [
                ("deaths", "Deaths"),
                ("losses", "Losses"),
                ("relief_funds", "Relief funds"),
            ]:
                blob = ex.get(key) or {}
                total = blob.get("total_mentions", 0)
                top = (blob.get("top_values") or [])[:3]
                if total or top:
                    st.sidebar.caption(f"{title}: {total} mentions")
                    for val, cnt in top:
                        st.sidebar.write(f"  - {val} ({cnt})")

    # Global filters driving both charts.
    st.markdown("### Comparison controls")
    events_available = sorted(media_enriched["event_label"].unique())
    default_events = events_available
    selected_events = st.multiselect(
        "Events to show in charts",
        events_available,
        default=default_events,
        help="Use this to focus the timeline and radar on a subset of events.",
    )

    min_date = media_enriched["publication_date"].min()
    max_date = media_enriched["publication_date"].max()
    if pd.isna(min_date) or pd.isna(max_date):
        date_range = None
    else:
        start_date_default = min_date.date()
        end_date_default = max_date.date()
        start_date, end_date = st.slider(
            "Publication date range",
            min_value=start_date_default,
            max_value=end_date_default,
            value=(start_date_default, end_date_default),
        )
        date_range = (
            pd.Timestamp(start_date),
            pd.Timestamp(end_date),
        )

    rolling_window = st.select_slider(
        "Smoothing window (days)",
        options=[1, 3, 7, 14],
        value=3,
        help="Apply a rolling mean to the daily media counts to smooth noise.",
    )

    st.markdown("### Comparative views")

    if not selected_events:
        st.info("Select at least one event to render the charts.")
        return

    # Multi-Event timeline – media volume over time.
    st.subheader("Multi-Event Timeline – Media Volume Over Time")
    filtered_media = media_enriched.copy()
    if date_range:
        start_ts, end_ts = date_range
        filtered_media = filtered_media[
            (filtered_media["publication_date"] >= start_ts)
            & (filtered_media["publication_date"] <= end_ts)
        ]
    filtered_media = filtered_media[filtered_media["event_label"].isin(selected_events)]

    if filtered_media.empty:
        st.info("No media reports for the selected filters.")
    else:
        fig_timeline = build_multi_event_timeline_figure(
            filtered_media,
            events=selected_events,
            date_range=None,
            rolling_window=rolling_window,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    # Resilience radar – impact and media coverage, normalised per axis.
    st.subheader("Resilience Radar – Magnitude, Exposure, Coverage, Vulnerability")
    filtered_impact = impact_df[impact_df["event_label"].isin(selected_events)]
    filtered_media_events = media_events_df[
        media_events_df["event_label"].isin(selected_events)
    ]

    if filtered_impact.empty:
        st.info("No impact data available for the selected events.")
    else:
        fig_radar = build_resilience_radar_figure(
            filtered_impact,
            filtered_media_events,
            events=selected_events,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Additional perspectives using the same filters.
    if not filtered_media.empty:
        st.markdown("### Additional perspectives")
        st.subheader("Average Headline Sentiment Over Time")
        if "headline_sentiment_compound" not in filtered_media.columns:
            st.info("No sentiment data available for the selected filters.")
        else:
            fig_sentiment = build_sentiment_over_time_figure(filtered_media)
            st.plotly_chart(fig_sentiment, use_container_width=True)

    with st.expander("How to read this dashboard"):
        st.markdown(
            """
            - **Multi-Event Timeline** shows comparative media trajectories across all events, with optional
              smoothing to reveal structural patterns instead of daily noise.
            - **Resilience Radar** normalises magnitude, exposure, coverage, and a vulnerability
              proxy so you can compare profile shapes rather than raw scales.
            - **Sentiment Over Time** tracks how the tone of media coverage evolves for each event.
            """
        )


if __name__ == "__main__":
    print(
        "Launching Stage 6 Disaster Pulse dashboard with Streamlit.\n"
        "Run this script via: streamlit run stage6_dashboard.py"
    )
    main()
