"""
SmogAlert PK - Interactive Dashboard (SmogNet Datathon Edition)

Streamlit dashboard for the 3-stage pipeline:
  anomaly detection → source classification → alert generation

Data files consumed:
  - data/cleaned_data.csv             (all 127,551 cleaned readings, 5 cities)
  - outputs/anomalies.csv             (5,765 detected anomalies)
  - outputs/anomalies_classified.csv  (anomalies + source label)
  - outputs/alerts_log.csv            (372 structured bilingual alerts)

WHAT IS STREAMLIT?
==================
Streamlit turns Python scripts into web apps with no HTML/CSS/JS.
Key concepts used below:
  st.tabs()           - tabbed layout
  st.columns()        - side-by-side columns
  st.plotly_chart()   - interactive Plotly charts
  st.dataframe()      - interactive table
  st.expander()       - collapsible section
  st.multiselect()    - filter widget
  st.selectbox()      - dropdown widget
  @st.cache_data      - cache expensive loads so reruns stay fast
"""

# ============================================================================
# IMPORTS
# ============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import os
from datetime import datetime

# ============================================================================
# PAGE CONFIG  (must be the very first Streamlit call)
# ============================================================================

st.set_page_config(
    page_title="SmogAlert PK",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

CLEANED_DATA_FILE  = "data/cleaned_data.csv"
ANOMALIES_FILE     = "outputs/anomalies.csv"
CLASSIFIED_FILE    = "outputs/anomalies_classified.csv"
ALERTS_LOG_FILE    = "outputs/alerts_log.csv"

CONFUSION_MATRIX_IMG    = "outputs/confusion_matrix.png"
FEATURE_IMPORTANCE_IMG  = "outputs/feature_importance.png"
FORECAST_FILE           = "outputs/forecast_24h.csv"
FORECAST_ALERTS_FILE    = "outputs/forecast_alerts.csv"
FORECAST_PLOT_IMG       = "outputs/forecast_plot.png"

# Real coordinates for the 5 monitored cities
CITY_COORDS = {
    "Islamabad": {"lat": 33.6844, "lon": 73.0479},
    "Karachi":   {"lat": 24.8607, "lon": 67.0011},
    "Lahore":    {"lat": 31.5497, "lon": 74.3436},
    "Peshawar":  {"lat": 34.0151, "lon": 71.5249},
    "Quetta":    {"lat": 30.1798, "lon": 66.9750},
}

WHO_SAFE_LIMIT = 15   # µg/m³

# Emoji badges and colours for each emission source type
SOURCE_META = {
    "crop_burning": {"badge": "🌾 Crop Burning",  "color": "#e07b39"},
    "vehicular":    {"badge": "🚗 Vehicular",      "color": "#5a8fc2"},
    "industrial":   {"badge": "🏭 Industrial",     "color": "#9b59b6"},
    "dust_storm":   {"badge": "🌪 Dust Storm",     "color": "#c8a951"},
    "mixed":        {"badge": "🔀 Mixed",           "color": "#27ae60"},
    "unclassified": {"badge": "❓ Unclassified",   "color": "#95a5a6"},
}

# AQI category → map marker colour
AQI_COLORS = {
    "Good":        "green",
    "Moderate":    "orange",   # yellow markers are hard to see on light tiles
    "Unhealthy":   "orange",
    "Hazardous":   "red",
}

# Season background colours for trend chart shading
SEASON_COLORS = {
    "Winter": "rgba(173, 216, 230, 0.15)",   # light blue
    "Spring": "rgba(144, 238, 144, 0.15)",   # light green
    "Summer": "rgba(255, 200, 100, 0.15)",   # light orange
    "Autumn": "rgba(205, 133,  63, 0.12)",   # light brown
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def aqi_to_color(aqi_category: str) -> str:
    """
    Map an AQI category string to a Folium-compatible colour name.

    Parameters:
        aqi_category (str): One of Good / Moderate / Unhealthy / Hazardous

    Returns:
        str: Colour name understood by Folium CircleMarker
    """
    return AQI_COLORS.get(aqi_category, "gray")


def source_badge(label: str) -> str:
    """
    Return the display badge string for a source label.

    Parameters:
        label (str): Raw source label, e.g. 'crop_burning'

    Returns:
        str: Human-readable badge, e.g. '🌾 Crop Burning'
    """
    return SOURCE_META.get(label, {}).get("badge", label)


@st.cache_data
def load_data():
    """
    Load all four pipeline output files and parse timestamps.

    Uses @st.cache_data so files are read only once per session —
    subsequent widget interactions reuse the in-memory copies.

    Returns:
        tuple: (cleaned_df, anomalies_df, classified_df, alerts_df)
               Any file that does not exist is returned as None.
    """
    def read(path):
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        return None

    return (
        read(CLEANED_DATA_FILE),
        read(ANOMALIES_FILE),
        read(CLASSIFIED_FILE),
        read(ALERTS_LOG_FILE),
    )


@st.cache_data
def load_forecast():
    """
    Load the city-specific Prophet 24-hour PM2.5 forecast CSV.

    Returns:
        DataFrame with columns: city, timestamp, pm25_predicted, pm25_lower,
        pm25_upper, alert_level — or None if the file does not exist or uses
        the old single-city schema (no 'city' column).
    """
    if not os.path.exists(FORECAST_FILE):
        return None
    df = pd.read_csv(FORECAST_FILE)
    if "city" not in df.columns:
        # Old single-city schema — not compatible with city-specific dashboard.
        # User must re-run python src/model.py to get the updated output.
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def load_forecast_alerts():
    """
    Load the forward-looking forecast alert records.

    Returns:
        DataFrame with columns: alert_type, city, forecast_timestamp,
        hours_ahead, pm25_predicted, pm25_upper, aqi_level, alert_level,
        affected_groups, protective_actions, alert_text_en, alert_text_ur
        — or None if the file does not exist or is empty.
    """
    if not os.path.exists(FORECAST_ALERTS_FILE):
        return None
    df = pd.read_csv(FORECAST_ALERTS_FILE)
    return df if len(df) > 0 else None


def add_season_bands(fig, df):
    """
    Add coloured vertical background bands to a Plotly figure for each season.

    The function scans the DataFrame for contiguous blocks of the same season
    (sorted by timestamp) and draws a semi-transparent rectangle for each block.

    Parameters:
        fig (go.Figure): Plotly figure to annotate in-place
        df  (DataFrame): Must have 'timestamp' and 'season' columns
    """
    if df is None or df.empty or "season" not in df.columns:
        return

    # Build list of (start, end, season) contiguous blocks
    sorted_df = df.sort_values("timestamp")[["timestamp", "season"]].drop_duplicates()
    blocks = []
    current_season = None
    block_start = None

    for _, row in sorted_df.iterrows():
        if row["season"] != current_season:
            if current_season is not None:
                blocks.append((block_start, row["timestamp"], current_season))
            current_season = row["season"]
            block_start = row["timestamp"]

    # Close the last open block
    if current_season is not None:
        blocks.append((block_start, sorted_df["timestamp"].iloc[-1], current_season))

    for start, end, season in blocks:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=SEASON_COLORS.get(season, "rgba(200,200,200,0.1)"),
            layer="below",
            line_width=0,
            annotation_text=season,
            annotation_position="top left",
            annotation=dict(font_size=10, font_color="gray"),
        )

# ============================================================================
# LOAD DATA
# ============================================================================

cleaned_df, anomalies_df, classified_df, alerts_df = load_data()
forecast_df        = load_forecast()
forecast_alerts_df = load_forecast_alerts()

# ============================================================================
# PAGE HEADER
# ============================================================================

st.title("🌫️ SmogAlert PK — Air Quality Intelligence Dashboard")
st.markdown(
    "Anomaly detection → source classification → public health alerts → 24h PM2.5 forecasting "
    "across 5 Pakistani cities (Aug 2021 – Dec 2024)"
)
st.markdown("---")

if cleaned_df is None:
    st.error("⚠️ Cleaned data not found. Run the full pipeline first:")
    st.code(
        "python download_data.py\n"
        "python src/preprocess.py\n"
        "python src/model.py\n"
        "python src/source_classifier.py\n"
        "python src/alert_system.py"
    )
    st.stop()

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Live Map",
    "📈 Air Quality Trends",
    "🔬 Source Classification",
    "⚠️ Alerts Dashboard",
    "🔮 24h PM2.5 Forecast",
    "🎯 Model Performance",
])

# ============================================================================
# TAB 1 — LIVE MAP
# ============================================================================

with tab1:
    st.header("Air Quality Map of Pakistan")
    st.markdown("Latest PM2.5 reading per city, coloured by AQI category.")

    # --- Get latest reading per city -------------------------------------------
    # cleaned_df has a real 'city' column, so we group and take the last row.
    latest_per_city = (
        cleaned_df.sort_values("timestamp")
        .groupby("city")
        .last()
        .reset_index()
    )

    # Attach latest alert source label for map popup (if available)
    if alerts_df is not None:
        latest_alert = (
            alerts_df.sort_values("timestamp")
            .groupby("city")[["source_label"]]
            .last()
            .rename(columns={"source_label": "latest_source"})
        )
        latest_per_city = latest_per_city.merge(latest_alert, on="city", how="left")
    else:
        latest_per_city["latest_source"] = "N/A"

    # --- Build Folium map -------------------------------------------------------
    pakistan_map = folium.Map(
        location=[30.3753, 69.3451],
        zoom_start=5,
        tiles="OpenStreetMap"
    )

    for _, row in latest_per_city.iterrows():
        city = row["city"]
        if city not in CITY_COORDS:
            continue

        coords  = CITY_COORDS[city]
        pm25    = round(row["pm25"], 1)
        aqi_cat = row.get("aqi_category", "N/A")
        source  = source_badge(str(row.get("latest_source", "N/A")))
        color   = aqi_to_color(aqi_cat)

        popup_html = (
            f"<b>{city}</b><br>"
            f"PM2.5: {pm25} µg/m³<br>"
            f"AQI: <b>{aqi_cat}</b><br>"
            f"Latest source: {source}"
        )

        folium.CircleMarker(
            location=[coords["lat"], coords["lon"]],
            radius=18,
            popup=folium.Popup(popup_html, max_width=200),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2,
        ).add_to(pakistan_map)

        folium.Marker(
            location=[coords["lat"], coords["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11pt;color:black;font-weight:bold">{city}</div>'
            ),
        ).add_to(pakistan_map)

    folium_static(pakistan_map, width=1200, height=580)

    # --- Legend ----------------------------------------------------------------
    st.markdown("### Legend")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("🟢 **Good** — PM2.5 ≤ 50")
    c2.markdown("🟡 **Moderate** — 51–100")
    c3.markdown("🟠 **Unhealthy** — 101–150")
    c4.markdown("🔴 **Hazardous** — > 150")

    # --- Per-city summary cards ------------------------------------------------
    st.markdown("### Latest Readings")
    cols = st.columns(5)
    for idx, (_, row) in enumerate(latest_per_city.iterrows()):
        with cols[idx % 5]:
            st.metric(
                label=f"{row['city']} — {row.get('aqi_category', '')}",
                value=f"{round(row['pm25'], 1)} µg/m³",
            )

# ============================================================================
# TAB 2 — AIR QUALITY TRENDS
# ============================================================================

with tab2:
    st.header("Air Quality Trends Over Time")

    # City filter
    city_list = sorted(cleaned_df["city"].unique().tolist())
    selected_city = st.selectbox("Select city:", options=city_list, index=city_list.index("Lahore"))

    city_data = cleaned_df[cleaned_df["city"] == selected_city].copy()

    # Build line chart
    fig = go.Figure()

    # Season background bands (drawn first so they sit behind the data lines)
    add_season_bands(fig, city_data)

    # PM2.5 line
    fig.add_trace(go.Scatter(
        x=city_data["timestamp"],
        y=city_data["pm25"],
        mode="lines",
        name="PM2.5",
        line=dict(color="#2980b9", width=1),
        hovertemplate="<b>%{x}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
    ))

    # WHO safe limit
    fig.add_hline(
        y=WHO_SAFE_LIMIT,
        line_dash="dash",
        line_color="green",
        annotation_text=f"WHO limit ({WHO_SAFE_LIMIT} µg/m³)",
        annotation_position="bottom right",
    )

    # Anomaly markers for this city
    if anomalies_df is not None:
        city_anomalies = anomalies_df[anomalies_df["city"] == selected_city]
        if len(city_anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=city_anomalies["timestamp"],
                y=city_anomalies["pm25"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="red", size=6, symbol="x", line=dict(width=2)),
                hovertemplate="<b>ANOMALY</b> %{x}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
            ))

    fig.update_layout(
        title=f"PM2.5 — {selected_city} (shaded bands = seasons)",
        xaxis_title="Date",
        yaxis_title="PM2.5 (µg/m³)",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Stats row
    st.markdown("### Statistics")
    s1, s2, s3, s4 = st.columns(4)
    city_anom_count = len(anomalies_df[anomalies_df["city"] == selected_city]) if anomalies_df is not None else 0

    s1.metric("Avg PM2.5",  f"{city_data['pm25'].mean():.1f} µg/m³")
    s2.metric("Max PM2.5",  f"{city_data['pm25'].max():.1f} µg/m³")
    s3.metric("Min PM2.5",  f"{city_data['pm25'].min():.1f} µg/m³")
    s4.metric("Anomalies",  str(city_anom_count))

# ============================================================================
# TAB 3 — SOURCE CLASSIFICATION
# ============================================================================

with tab3:
    st.header("Pollution Source Classification")
    st.markdown(
        "Rule-based chemical fingerprint classifier applied to all "
        f"{len(classified_df):,} detected anomalies."
        if classified_df is not None else ""
    )

    if classified_df is None:
        st.warning("⚠️ Run `python src/source_classifier.py` first.")
    else:
        # ── A. Source Distribution Bar Chart ─────────────────────────────────
        st.markdown("### Source Label Distribution (all anomalies)")

        source_counts = (
            classified_df["source_label"]
            .value_counts()
            .reset_index()
        )
        source_counts.columns = ["source_label", "count"]
        source_counts["badge"]  = source_counts["source_label"].map(source_badge)
        source_counts["color"]  = source_counts["source_label"].map(
            lambda x: SOURCE_META.get(x, {}).get("color", "#888")
        )

        fig_dist = px.bar(
            source_counts,
            x="badge",
            y="count",
            color="badge",
            color_discrete_map={row["badge"]: row["color"] for _, row in source_counts.iterrows()},
            labels={"badge": "Source Type", "count": "Anomaly Count"},
            title="Anomaly Count by Emission Source",
            text="count",
        )
        fig_dist.update_traces(textposition="outside")
        fig_dist.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

        # ── B. Per-City Source Breakdown ──────────────────────────────────────
        st.markdown("### Per-City Source Breakdown")

        city_source_table = pd.crosstab(
            classified_df["city"],
            classified_df["source_label"]
        )
        # Rename columns with badge labels for readability
        city_source_table.columns = [source_badge(c) for c in city_source_table.columns]
        city_source_table.index.name = "City"

        st.dataframe(city_source_table, use_container_width=True)

        # Grouped bar chart version
        melt_df = (
            classified_df
            .groupby(["city", "source_label"])
            .size()
            .reset_index(name="count")
        )
        melt_df["badge"] = melt_df["source_label"].map(source_badge)
        melt_df["color"] = melt_df["source_label"].map(
            lambda x: SOURCE_META.get(x, {}).get("color", "#888")
        )

        fig_city = px.bar(
            melt_df,
            x="city",
            y="count",
            color="badge",
            barmode="group",
            color_discrete_map={row["badge"]: row["color"] for _, row in melt_df.iterrows()},
            labels={"city": "City", "count": "Anomaly Count", "badge": "Source"},
            title="Source Types per City",
            height=400,
        )
        st.plotly_chart(fig_city, use_container_width=True)

        # ── C. Top Severe Events ──────────────────────────────────────────────
        st.markdown("### Top 20 Severe Events (test window, sorted by PM2.5)")

        top_events = (
            classified_df[classified_df["split"] == "test"]
            .sort_values("pm25", ascending=False)
            .head(20)[["timestamp", "city", "pm25", "aqi_category", "source_label", "pollutant_signature"]]
            .copy()
        )
        top_events["source_label"] = top_events["source_label"].map(source_badge)
        top_events["timestamp"] = top_events["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        top_events["pm25"] = top_events["pm25"].round(1)
        top_events.columns = ["Timestamp", "City", "PM2.5", "AQI", "Source", "Signature"]

        st.dataframe(top_events, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 4 — ALERTS DASHBOARD
# ============================================================================

with tab4:
    st.header("Public Health Alerts")

    if alerts_df is None or len(alerts_df) == 0:
        st.warning("⚠️ Run `python src/alert_system.py` first.")
    else:
        # ── Filters ───────────────────────────────────────────────────────────
        st.markdown("### Filters")
        f1, f2, f3 = st.columns(3)

        with f1:
            city_opts = sorted(alerts_df["city"].unique().tolist())
            sel_cities = st.multiselect("City", options=city_opts, default=city_opts)

        with f2:
            src_opts = sorted(alerts_df["source_label"].unique().tolist())
            src_display = {source_badge(s): s for s in src_opts}
            sel_src_badges = st.multiselect(
                "Source type",
                options=list(src_display.keys()),
                default=list(src_display.keys()),
            )
            sel_sources = [src_display[b] for b in sel_src_badges]

        with f3:
            min_date = alerts_df["timestamp"].min().date()
            max_date = alerts_df["timestamp"].max().date()
            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

        # Apply filters
        filtered = alerts_df[
            alerts_df["city"].isin(sel_cities) &
            alerts_df["source_label"].isin(sel_sources)
        ].copy()

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_dt = pd.Timestamp(date_range[0])
            end_dt   = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
            filtered = filtered[
                (filtered["timestamp"] >= start_dt) &
                (filtered["timestamp"] <  end_dt)
            ]

        # ── Summary Metrics ───────────────────────────────────────────────────
        st.markdown("### Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Alerts",       len(filtered))
        m2.metric("🔴 Hazardous",       (filtered["aqi_level"] == "Hazardous").sum())
        m3.metric("🟠 Unhealthy",       (filtered["aqi_level"] == "Unhealthy").sum())
        m4.metric("Cities Affected",    filtered["city"].nunique())

        st.markdown("---")

        # ── Alert Table ───────────────────────────────────────────────────────
        st.markdown("### Alert Records")

        display = filtered[["timestamp", "city", "aqi_level", "source_label", "pollutant_signature"]].copy()
        display["timestamp"]    = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        display["source_label"] = display["source_label"].map(source_badge)
        display.columns = ["Timestamp", "City", "AQI Level", "Source", "Pollutant Signature"]

        st.dataframe(display, use_container_width=True, height=350, hide_index=True)

        # ── Expandable Detail Sections per Source ─────────────────────────────
        st.markdown("### Alert Details by Source Type")

        for raw_label, meta in SOURCE_META.items():
            subset = filtered[filtered["source_label"] == raw_label]
            if len(subset) == 0:
                continue

            sample = subset.iloc[0]  # representative row for shared fields

            with st.expander(f"{meta['badge']}  —  {len(subset)} alert(s)", expanded=False):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(f"**Affected groups**")
                    st.info(sample["affected_groups"])
                    st.markdown(f"**Protective actions**")
                    st.info(sample["protective_actions"])

                with col_b:
                    st.markdown("**Sample alert (English)**")
                    st.success(sample["alert_text_en"])
                    st.markdown("**Sample alert (Urdu)**")
                    st.success(sample["alert_text_ur"])

        # ── Download ──────────────────────────────────────────────────────────
        st.markdown("---")
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download filtered alerts as CSV",
            data=csv_bytes,
            file_name=f"alerts_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# ============================================================================
# TAB 5 — 24H PM2.5 FORECAST (PROPHET)
# ============================================================================

with tab5:
    st.header("24-Hour PM2.5 Forecast (City-Specific Prophet Models)")
    st.markdown(
        "One Prophet model is trained per city so each city's unique seasonal "
        "patterns (e.g. Lahore's winter smog vs. Karachi's sea-breeze effect) "
        "are captured independently.  Forward-looking alerts are generated when "
        "predicted PM2.5 crosses the Unhealthy (100 µg/m³) or Hazardous (150 µg/m³) threshold."
    )

    if forecast_df is None:
        st.warning(
            "⚠️ City-specific forecast not found or uses an outdated format.  "
            "Re-run `python src/model.py` to generate the updated city-specific "
            "Prophet forecasts and forecast alerts."
        )
    else:
        # ── City Selector ─────────────────────────────────────────────────────
        forecast_cities = sorted(forecast_df["city"].unique())
        selected_fc_city = st.selectbox(
            "Select city to display forecast:",
            options=forecast_cities,
            index=0,
            key="forecast_city_selector",
        )

        city_fc = forecast_df[forecast_df["city"] == selected_fc_city].copy()

        # ── Summary Metrics ───────────────────────────────────────────────────
        peak_val   = city_fc["pm25_predicted"].max()
        trough_val = city_fc["pm25_predicted"].min()
        peak_hour  = city_fc.loc[city_fc["pm25_predicted"].idxmax(), "timestamp"]
        alert_lvls = city_fc["alert_level"].value_counts()

        # Count forecast alerts for this city
        n_fc_alerts = 0
        if forecast_alerts_df is not None and "city" in forecast_alerts_df.columns:
            n_fc_alerts = len(forecast_alerts_df[forecast_alerts_df["city"] == selected_fc_city])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak Forecast",     f"{peak_val:.1f} µg/m³",
                  help=f"At {peak_hour.strftime('%H:%M')}")
        m2.metric("Min Forecast",      f"{trough_val:.1f} µg/m³")
        m3.metric("Forecast Hours",    str(len(city_fc)))
        m4.metric("Forecast Alerts",   str(n_fc_alerts),
                  help="Number of forward-looking alerts triggered for this city")

        st.markdown("---")

        # ── Forecast Chart ────────────────────────────────────────────────────
        st.markdown(f"### Hourly PM2.5 Forecast — {selected_fc_city}")

        fig_fc = go.Figure()

        # Shaded 95% confidence band (upper → lower filled as a closed polygon)
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([city_fc["timestamp"], city_fc["timestamp"][::-1]]),
            y=pd.concat([city_fc["pm25_upper"], city_fc["pm25_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(41, 128, 185, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence Interval",
            hoverinfo="skip",
        ))

        # Predicted line with alert-level colour coding via marker colour
        alert_color_map = {"GREEN": "#2ecc71", "YELLOW": "#f1c40f",
                           "ORANGE": "#e67e22", "RED": "#e74c3c"}
        marker_colors = city_fc["alert_level"].map(alert_color_map).fillna("#2980b9")

        fig_fc.add_trace(go.Scatter(
            x=city_fc["timestamp"],
            y=city_fc["pm25_predicted"],
            mode="lines+markers",
            name="PM2.5 Forecast",
            line=dict(color="#2980b9", width=2),
            marker=dict(size=7, color=marker_colors, line=dict(width=0.5, color="white")),
            hovertemplate=(
                "<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                "Predicted: %{y:.1f} µg/m³<extra></extra>"
            ),
        ))

        # Reference lines
        fig_fc.add_hline(
            y=WHO_SAFE_LIMIT, line_dash="dash", line_color="green",
            annotation_text=f"WHO limit ({WHO_SAFE_LIMIT} µg/m³)",
            annotation_position="bottom right",
        )
        fig_fc.add_hline(
            y=100, line_dash="dot", line_color="orange",
            annotation_text="Unhealthy (100 µg/m³)",
            annotation_position="top right",
        )
        fig_fc.add_hline(
            y=150, line_dash="dot", line_color="red",
            annotation_text="Hazardous (150 µg/m³)",
            annotation_position="top right",
        )

        fig_fc.update_layout(
            xaxis_title="Hour",
            yaxis_title="PM2.5 (µg/m³)",
            hovermode="x unified",
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        st.plotly_chart(fig_fc, use_container_width=True)

        # ── Forecast Alerts ───────────────────────────────────────────────────
        st.markdown("### Forward-Looking Forecast Alerts")

        if forecast_alerts_df is None:
            st.info(
                "No forecast alerts generated — either no threshold breach is predicted "
                "or `forecast_alerts.csv` has not been created yet.  "
                "Re-run `python src/model.py` to regenerate."
            )
        else:
            city_alerts = forecast_alerts_df[
                forecast_alerts_df["city"] == selected_fc_city
            ].copy()

            if len(city_alerts) == 0:
                st.success(
                    f"No PM2.5 threshold breaches predicted in the next 24 hours for "
                    f"{selected_fc_city}.  Air quality is expected to remain below "
                    f"Unhealthy levels."
                )
            else:
                st.warning(
                    f"**{len(city_alerts)} forecast alert(s)** triggered for "
                    f"{selected_fc_city} in the next 24 hours."
                )

                for _, alert_row in city_alerts.iterrows():
                    level_icon = "🚨" if alert_row["alert_level"] == "RED" else "⚠️"
                    with st.expander(
                        f"{level_icon} {alert_row['aqi_level']} — "
                        f"predicted {alert_row['pm25_predicted']:.0f} µg/m³ "
                        f"at {alert_row['forecast_timestamp']} "
                        f"(~{alert_row['hours_ahead']}h from now)",
                        expanded=True,
                    ):
                        col_en, col_ur = st.columns(2)
                        with col_en:
                            st.markdown("**English**")
                            st.info(alert_row["alert_text_en"])
                        with col_ur:
                            st.markdown("**اردو**")
                            st.info(alert_row["alert_text_ur"])

                        st.markdown(
                            f"**At risk:** {alert_row['affected_groups']}  \n"
                            f"**Actions:** {alert_row['protective_actions']}  \n"
                            f"**Worst-case PM2.5:** {alert_row['pm25_upper']:.1f} µg/m³ "
                            f"(95% confidence upper bound)"
                        )

            # Show all-cities forecast alert table
            with st.expander("View all-city forecast alerts table", expanded=False):
                if len(forecast_alerts_df) == 0:
                    st.write("No forecast alerts across any city.")
                else:
                    display_fa = forecast_alerts_df[[
                        "city", "forecast_timestamp", "hours_ahead",
                        "pm25_predicted", "pm25_upper", "aqi_level",
                        "alert_text_en",
                    ]].copy()
                    display_fa.columns = [
                        "City", "Forecast Time", "Hours Ahead",
                        "PM2.5 Predicted", "PM2.5 (Worst Case)", "AQI Level",
                        "Alert (English)",
                    ]
                    st.dataframe(display_fa, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Forecast Data Table ───────────────────────────────────────────────
        with st.expander("View forecast data table", expanded=False):
            display_fc = city_fc[["timestamp", "pm25_predicted",
                                   "pm25_lower", "pm25_upper", "alert_level"]].copy()
            display_fc["timestamp"] = display_fc["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            display_fc.columns = ["Hour", "PM2.5 Forecast", "Lower Bound",
                                   "Upper Bound", "Alert Level"]
            st.dataframe(display_fc, use_container_width=True, hide_index=True)

        # ── Static Multi-City Prophet Plot ────────────────────────────────────
        if os.path.exists(FORECAST_PLOT_IMG):
            st.markdown("### All-City Prophet Forecast Plot")
            with st.expander("Show all-city forecast chart", expanded=False):
                st.image(FORECAST_PLOT_IMG, use_container_width=True)

        # ── Explanation ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### How this forecast works")
        st.markdown("""
        **Facebook Prophet** is a time-series forecasting model designed for data
        with strong seasonal patterns and holiday effects.

        - One model is trained **per city** so Lahore's winter smog profile does
          not dilute Karachi's sea-breeze pattern (or vice versa)
        - Each model decomposes its city's signal into **trend + daily + weekly + yearly seasonality**
        - Outputs a point forecast (`pm25_predicted`) and a **95% confidence interval**
          (`pm25_lower` / `pm25_upper`) for each of the next 24 hours
        - **Marker colours** on the forecast line show the alert level per hour:
          🟢 GREEN ≤ 50 · 🟡 YELLOW ≤ 100 · 🟠 ORANGE ≤ 150 · 🔴 RED > 150
        - **Forward-looking alerts** fire when `pm25_predicted` crosses 100 µg/m³ (Unhealthy)
          or 150 µg/m³ (Hazardous); one alert per contiguous breach block to avoid alert fatigue
        - Models saved at `models/prophet_{city}.pkl` — re-run `python src/model.py` to retrain
        """)


# ============================================================================
# TAB 6 — MODEL PERFORMANCE
# ============================================================================

with tab6:
    st.header("Machine Learning Model Performance")
    st.markdown("""
    SmogAlert PK uses two ML models in tandem:
    1. **Isolation Forest** — per-city-season anomaly detector (20 models)
    2. **Random Forest Classifier** — AQI category predictor
    """)

    # ── Summary Metrics ───────────────────────────────────────────────────────
    st.markdown("### Pipeline Numbers")
    p1, p2, p3, p4 = st.columns(4)

    p1.metric("Cleaned Rows",       "127,551",     help="5 cities × Aug 2021 – Dec 2024")
    p2.metric("Anomalies Detected", "5,765 (5%)",  help="Isolation Forest, contamination = 0.05")
    p3.metric("IF Models Trained",  "20",           help="One per city × season group")
    p4.metric("RF AQI Accuracy",    "100%",         help="Random Forest on labelled AQI categories")

    # Per-city anomaly breakdown
    if anomalies_df is not None:
        st.markdown("### Anomalies per City")
        city_anom = (
            anomalies_df.groupby("city")
            .size()
            .reset_index(name="anomaly_count")
            .sort_values("anomaly_count", ascending=False)
        )
        fig_anom = px.bar(
            city_anom,
            x="city",
            y="anomaly_count",
            color="city",
            text="anomaly_count",
            labels={"city": "City", "anomaly_count": "Anomalies"},
            title="Detected Anomalies per City",
            height=350,
        )
        fig_anom.update_traces(textposition="outside")
        fig_anom.update_layout(showlegend=False)
        st.plotly_chart(fig_anom, use_container_width=True)

    st.markdown("---")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    st.markdown("### Random Forest — Confusion Matrix")

    with st.expander("ℹ️ What is a Confusion Matrix?", expanded=False):
        st.markdown("""
        Shows how well the model predicts each AQI category.
        - **Rows**: true labels  · **Columns**: predicted labels
        - Numbers on the diagonal = correct predictions
        """)

    if os.path.exists(CONFUSION_MATRIX_IMG):
        st.image(CONFUSION_MATRIX_IMG, use_container_width=True)
    else:
        st.warning("⚠️ Confusion matrix not found — run `python src/model.py`.")

    st.markdown("---")

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown("### Random Forest — Feature Importance")

    with st.expander("ℹ️ What is Feature Importance?", expanded=False):
        st.markdown("""
        Shows which input variables the Random Forest relies on most
        when predicting the AQI danger category.
        Higher bar = more influential feature.
        """)

    if os.path.exists(FEATURE_IMPORTANCE_IMG):
        st.image(FEATURE_IMPORTANCE_IMG, use_container_width=True)
    else:
        st.warning("⚠️ Feature importance chart not found — run `python src/model.py`.")

    st.markdown("---")

    # ── How the Models Work ───────────────────────────────────────────────────
    st.markdown("### How the Models Work")
    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("""
        **🔍 Isolation Forest (Anomaly Detection)**

        - One model trained per `city_season` group (e.g., `Lahore_Winter`)
        - Trained on training split only (pre-Jul 2024)
        - Features: `pm25`, `pm10`, `pm25_24h_avg`, `hour`
        - Contamination = 5% → flags the most anomalous 5% of readings
        - Produces `is_anomaly` flag applied to both train and test windows
        """)

    with mc2:
        st.markdown("""
        **🌳 Random Forest Classifier (AQI Prediction)**

        - 100 decision trees, majority vote
        - Predicts AQI category: Good / Moderate / Unhealthy / Hazardous
        - Learns patterns such as: "PM2.5 > 150 at hour 8 → Hazardous"
        - Provides feature importance scores
        - Retained alongside anomaly pipeline for dashboard visualisation
        """)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("About SmogAlert PK")
    st.markdown("""
    **SmogAlert PK** is an AI-powered air quality intelligence system
    for Pakistan, built for the **SmogNet Datathon (UET Mardan)**.

    **Pipeline:**
    1. 🔍 Anomaly detection (Isolation Forest)
    2. 🏷️ Source classification (rule-based fingerprints)
    3. 📢 Bilingual alert generation

    **Coverage:**
    - 5 cities: Islamabad, Karachi, Lahore, Peshawar, Quetta
    - 8 pollutants: PM2.5, PM10, NO, NO₂, SO₂, NH₃, CO, O₃
    - Aug 2021 – Dec 2024
    """)

    st.markdown("---")
    st.markdown("### Data Source")
    st.markdown(
        "Kaggle: `hajramohsin/`  \n"
        "`pakistan-air-quality-`  \n"
        "`pollutant-concentrations`"
    )

    st.markdown("---")
    st.markdown("### System Status")

    checks = {
        "Cleaned data":      CLEANED_DATA_FILE,
        "Anomalies":         ANOMALIES_FILE,
        "Classified":        CLASSIFIED_FILE,
        "Alerts":            ALERTS_LOG_FILE,
        "RF model":          "models/random_forest_model.pkl",
    }
    for label, path in checks.items():
        icon = "✅" if os.path.exists(path) else "❌"
        st.markdown(f"- {label}: {icon}")

    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center'>"
    "<p>SmogAlert PK © 2026 · "
    "Built with Streamlit · Maps by Folium · Charts by Plotly</p>"
    "</div>",
    unsafe_allow_html=True,
)
