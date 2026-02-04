import os
import random
import hashlib
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

st.set_page_config(
    page_title="Global Air Quality Dashboard",
    page_icon="AQ",
    layout="wide",
)

CITY_DEFAULTS = [
    "New York",
    "Los Angeles",
    "London",
    "Paris",
    "Berlin",
    "Madrid",
    "Rome",
    "Istanbul",
    "Moscow",
    "Cairo",
    "Lagos",
    "Johannesburg",
    "Dubai",
    "Delhi",
    "Mumbai",
    "Beijing",
    "Seoul",
    "Tokyo",
    "Singapore",
    "Jakarta",
    "Sydney",
    "Mexico City",
    "Sao Paulo",
]

CITY_COORDS = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050),
    "Madrid": (40.4168, -3.7038),
    "Rome": (41.9028, 12.4964),
    "Istanbul": (41.0082, 28.9784),
    "Moscow": (55.7558, 37.6173),
    "Cairo": (30.0444, 31.2357),
    "Lagos": (6.5244, 3.3792),
    "Johannesburg": (-26.2041, 28.0473),
    "Dubai": (25.2048, 55.2708),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Beijing": (39.9042, 116.4074),
    "Seoul": (37.5665, 126.9780),
    "Tokyo": (35.6762, 139.6503),
    "Singapore": (1.3521, 103.8198),
    "Jakarta": (-6.2088, 106.8456),
    "Sydney": (-33.8688, 151.2093),
    "Mexico City": (19.4326, -99.1332),
    "Sao Paulo": (-23.5505, -46.6333),
}

AQI_LABELS = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor",
}

COMPONENT_UNITS = "ug/m3"
DEMO_SEED = 42
AQI_COLORS = {
    1: "#2ecc71",
    2: "#f1c40f",
    3: "#f39c12",
    4: "#e67e22",
    5: "#e74c3c",
}


def get_api_key() -> Optional[str]:
    key = st.secrets.get("OPENWEATHER_API_KEY") if hasattr(st, "secrets") else None
    if not key:
        key = os.getenv("OPENWEATHER_API_KEY")
    return key


@st.cache_data(ttl=60 * 60)
def geocode_city(city: str, api_key: str) -> Optional[Dict[str, float]]:
    url = "https://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city, "limit": 1, "appid": api_key}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    return {"lat": data[0]["lat"], "lon": data[0]["lon"], "name": data[0]["name"]}


@st.cache_data(ttl=15 * 60)
def get_air_quality(lat: float, lon: float, api_key: str) -> Dict:
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60 * 60)
def get_air_quality_forecast(lat: float, lon: float, api_key: str) -> Dict:
    url = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60 * 60)
def get_air_quality_history(
    lat: float, lon: float, start_ts: int, end_ts: int, api_key: str
) -> Dict:
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {"lat": lat, "lon": lon, "start": start_ts, "end": end_ts, "appid": api_key}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def parse_current_aq(data: Dict) -> Tuple[int, Dict[str, float], int]:
    entry = data["list"][0]
    aqi = entry["main"]["aqi"]
    components = entry["components"]
    ts = entry["dt"]
    return aqi, components, ts


def forecast_to_df(data: Dict) -> pd.DataFrame:
    rows = []
    for entry in data.get("list", []):
        row = {"dt": datetime.utcfromtimestamp(entry["dt"])}
        row["aqi"] = entry["main"]["aqi"]
        row.update(entry["components"])
        rows.append(row)
    return pd.DataFrame(rows)


def history_to_df(data: Dict) -> pd.DataFrame:
    rows = []
    for entry in data.get("list", []):
        row = {"dt": datetime.utcfromtimestamp(entry["dt"])}
        row["aqi"] = entry["main"]["aqi"]
        row.update(entry["components"])
        rows.append(row)
    return pd.DataFrame(rows)


def components_summary(components: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(
        {"component": list(components.keys()), "value": list(components.values())}
    )
    df = df.sort_values("value", ascending=False, ignore_index=True)
    return df


def compute_view_state(points: pd.DataFrame) -> pdk.ViewState:
    if points.empty:
        return pdk.ViewState(latitude=15, longitude=10, zoom=0.9)
    lat_min, lat_max = points["lat"].min(), points["lat"].max()
    lon_min, lon_max = points["lon"].min(), points["lon"].max()
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    max_range = max(abs(lat_max - lat_min), abs(lon_max - lon_min))
    if max_range < 5:
        zoom = 4
    elif max_range < 10:
        zoom = 3
    elif max_range < 20:
        zoom = 2.5
    elif max_range < 40:
        zoom = 2
    elif max_range < 80:
        zoom = 1.5
    else:
        zoom = 1.0
    return pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=zoom)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
  font-family: 'Source Sans 3', system-ui, -apple-system, sans-serif;
}

.hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0f766e 100%);
  color: #f8fafc;
  padding: 28px 28px 22px 28px;
  border-radius: 18px;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.35);
}

.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.2rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 0 0 6px 0;
}

.hero-subtitle {
  font-size: 1.05rem;
  color: rgba(248, 250, 252, 0.9);
  margin: 0;
}

.pill {
  display: inline-block;
  margin-top: 10px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  background: rgba(248, 250, 252, 0.18);
}

.metric-card {
  background: #ffffff;
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
  border: 1px solid rgba(148, 163, 184, 0.25);
}

.metric-label {
  color: #64748b;
  font-size: 0.85rem;
  margin-bottom: 6px;
}

.metric-value {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.6rem;
  font-weight: 700;
  color: #0f172a;
}

.section-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 600;
  font-size: 1.35rem;
  color: #0f172a;
  margin-top: 10px;
}

.legend-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 10px;
  background: #f8fafc;
  border: 1px solid rgba(148, 163, 184, 0.25);
  margin-right: 8px;
  color: #0f172a;
}
</style>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Configuration")
    use_demo = st.toggle("Use demo data", value=False)
    api_key_input = st.text_input("OpenWeather API Key", type="password")
    select_all = st.toggle("Select all cities", value=True)
    default_cities = CITY_DEFAULTS if select_all else CITY_DEFAULTS[:5]
    city_selection = st.multiselect(
        "Cities",
        options=CITY_DEFAULTS,
        default=default_cities,
    )
    show_history = st.checkbox("Show historical trends", value=True)
    history_days = st.slider("History window (days)", min_value=1, max_value=5, value=3)
    rolling_hours = st.selectbox("Rolling average (hours)", [6, 12, 24, 168], index=2)
    st.caption("Tip: Provide the API key via `OPENWEATHER_API_KEY` or Streamlit secrets.")

hero_col = st.container()
with hero_col:
    mode_label = "Demo Mode" if use_demo else "Live Data"
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-title">Global Air Quality Dashboard</div>
  <p class="hero-subtitle">Live and demo-ready insights across the world's most watched cities.</p>
  <div class="pill">{mode_label} • Updated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</div>
</div>
""",
        unsafe_allow_html=True,
    )

api_key = api_key_input or get_api_key()

if not api_key and not use_demo:
    st.warning("Please provide an OpenWeather API key to load data or enable demo mode.")
    st.stop()

if not city_selection:
    st.info("Select at least one city.")
    st.stop()

@st.cache_data(ttl=60 * 60)
def get_city_payload(city: str, api_key: str) -> Optional[Dict]:
    loc = geocode_city(city, api_key)
    if not loc:
        return None
    current = get_air_quality(loc["lat"], loc["lon"], api_key)
    forecast = get_air_quality_forecast(loc["lat"], loc["lon"], api_key)
    return {"city": city, "loc": loc, "current": current, "forecast": forecast}


def demo_components(aqi: int, rng: random.Random) -> Dict[str, float]:
    base = {1: 6, 2: 12, 3: 20, 4: 35, 5: 55}.get(aqi, 12)
    jitter = lambda scale: max(0.1, rng.uniform(-scale, scale))
    return {
        "co": base * 8 + jitter(5),
        "no": base * 0.2 + jitter(0.2),
        "no2": base * 1.1 + jitter(1.5),
        "o3": base * 1.6 + jitter(2.0),
        "so2": base * 0.6 + jitter(1.0),
        "pm2_5": base * 1.4 + jitter(2.0),
        "pm10": base * 1.9 + jitter(3.0),
        "nh3": base * 0.5 + jitter(0.8),
    }


def demo_series(start_ts: int, hours: int, rng: random.Random) -> List[Dict]:
    series = []
    for i in range(hours):
        dt = start_ts + i * 3600
        aqi = max(1, min(5, int(3 + rng.uniform(-1.2, 1.2))))
        series.append(
            {
                "dt": dt,
                "main": {"aqi": aqi},
                "components": demo_components(aqi, rng),
            }
        )
    return series


def demo_payload(city: str) -> Dict:
    seed_bytes = hashlib.sha256(city.encode("utf-8")).digest()
    city_seed = int.from_bytes(seed_bytes[:4], "big") + DEMO_SEED
    rng = random.Random(city_seed)
    if city in CITY_COORDS:
        lat, lon = CITY_COORDS[city]
    else:
        lat = rng.uniform(-45, 55)
        lon = rng.uniform(-120, 140)
    now_ts = int(datetime.utcnow().timestamp())
    current_aqi = max(1, min(5, int(3 + rng.uniform(-1.2, 1.2))))
    current = {
        "list": [
            {
                "dt": now_ts,
                "main": {"aqi": current_aqi},
                "components": demo_components(current_aqi, rng),
            }
        ]
    }
    forecast = {"list": demo_series(now_ts, 48, rng=rng)}
    history = {"list": demo_series(now_ts - 72 * 3600, 72, rng=rng)}
    return {
        "city": city,
        "loc": {"lat": lat, "lon": lon, "name": city},
        "current": current,
        "forecast": forecast,
        "history": history,
    }


city_payloads = []
failed = []
for city in city_selection:
    try:
        if use_demo:
            payload = demo_payload(city)
        else:
            payload = get_city_payload(city, api_key)
        if payload is None:
            failed.append(city)
        else:
            city_payloads.append(payload)
    except requests.HTTPError:
        failed.append(city)

if failed:
    st.warning(f"Failed to load: {', '.join(failed)}")

if not city_payloads:
    st.stop()

# Build summary table
summary_rows = []
map_rows = []
for payload in city_payloads:
    aqi, components, ts = parse_current_aq(payload["current"])
    top_pollutant = max(components.items(), key=lambda x: x[1])[0]
    summary_rows.append(
        {
            "city": payload["loc"]["name"],
            "aqi": aqi,
            "aqi_label": AQI_LABELS.get(aqi, "Unknown"),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "top_pollutant": top_pollutant,
            "timestamp_utc": datetime.utcfromtimestamp(ts),
            "lat": payload["loc"]["lat"],
            "lon": payload["loc"]["lon"],
        }
    )
    map_rows.append(
        {
            "lat": payload["loc"]["lat"],
            "lon": payload["loc"]["lon"],
            "city": payload["loc"]["name"],
            "aqi": aqi,
        }
    )

summary_df = pd.DataFrame(summary_rows).sort_values("aqi", ascending=True)
avg_aqi = summary_df["aqi"].mean()
median_aqi = summary_df["aqi"].median()
poor_share = (summary_df["aqi"] >= 4).mean() * 100
dominant_pollutant = summary_df["top_pollutant"].mode().iloc[0]

hero_insights = st.columns(4)
hero_insights_data = [
    ("Global Avg AQI", f"{avg_aqi:.2f}"),
    ("Cities AQI >= 4", f"{poor_share:.0f}%"),
    ("Cleanest City", summary_df.sort_values("aqi").iloc[0]["city"]),
    ("Dominant Pollutant", dominant_pollutant),
]
for idx, (label, value) in enumerate(hero_insights_data):
    with hero_insights[idx]:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{value}</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown(
    f"""
<div style="margin-top:6px;color:#64748b;font-size:0.95rem;">
Median AQI {median_aqi:.1f} across selected cities.
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("Global Snapshot")
snapshot_left, snapshot_right = st.columns([2, 1])
with snapshot_left:
    buckets = []
    for aqi in [1, 2, 3, 4, 5]:
        cities = summary_df.loc[summary_df["aqi"] == aqi, "city"].tolist()
        buckets.append({"aqi": aqi, "label": AQI_LABELS[aqi], "cities": cities})

    bucket_cols = st.columns(2)
    for idx, bucket in enumerate(buckets):
        col = bucket_cols[idx % 2]
        with col:
            st.markdown(
                f"""
<div class="metric-card" style="border-top:6px solid {AQI_COLORS[bucket['aqi']]}; margin-bottom:12px;">
  <div class="metric-label">AQI {bucket['aqi']}</div>
  <div class="metric-value">{bucket['label']}</div>
  <div style="margin-top:10px;font-size:0.95rem;color:#0f172a;">
    {"<br/>".join(bucket["cities"]) if bucket["cities"] else "<span style='color:#94a3b8;'>No cities</span>"}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

with snapshot_right:
    st.caption("Snapshot highlights shown on the left.")

st.subheader("City Table")
table_df = summary_df[["city", "aqi", "aqi_label", "pm2_5", "pm10", "top_pollutant"]].copy()
table_df = table_df.rename(
    columns={
        "city": "City",
        "aqi": "AQI",
        "aqi_label": "AQI Level",
        "pm2_5": "PM2.5",
        "pm10": "PM10",
        "top_pollutant": "Top Pollutant",
    }
)
styled_table = (
    table_df.style
    .format({"PM2.5": "{:.1f}", "PM10": "{:.1f}"})
    .background_gradient(subset=["AQI"], cmap="RdYlGn_r")
    .set_properties(**{"border": "1px solid #e2e8f0"})
)
st.dataframe(styled_table, use_container_width=True, hide_index=True)

st.subheader("Air Quality Map")
map_df = pd.DataFrame(map_rows)
map_df["color"] = map_df["aqi"].map(
    lambda v: [
        int(AQI_COLORS[v][1:3], 16),
        int(AQI_COLORS[v][3:5], 16),
        int(AQI_COLORS[v][5:7], 16),
        170,
    ]
)
map_df["radius"] = map_df["aqi"].map(lambda v: 80000 + (v - 1) * 25000)
layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position="[lon, lat]",
    get_radius="radius",
    get_fill_color="color",
    pickable=True,
)
view_state = compute_view_state(map_df)
tooltip = {"text": "{city}\nAQI: {aqi}"}
deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
st.pydeck_chart(deck, use_container_width=True)

legend_items = []
for aqi in [1, 2, 3, 4, 5]:
    legend_items.append(
        f"""
<span class="legend-chip">
  <span style="width:12px;height:12px;border-radius:4px;background:{AQI_COLORS[aqi]};display:inline-block;"></span>
  <span>{aqi} - {AQI_LABELS[aqi]}</span>
</span>
"""
    )
st.markdown("".join(legend_items), unsafe_allow_html=True)

st.subheader("City Detail")
detail_options = [p["loc"]["name"] for p in city_payloads]
selected_city = st.selectbox("Select a city", detail_options, index=0, key="detail_city")
selected_payload = next(p for p in city_payloads if p["loc"]["name"] == selected_city)

current_aqi, current_components, current_ts = parse_current_aq(selected_payload["current"])

metric_cols = st.columns(4)
metric_cols[0].metric("AQI", current_aqi, AQI_LABELS.get(current_aqi, ""))
metric_cols[1].metric("PM2.5", f"{current_components.get('pm2_5', 0):.1f} {COMPONENT_UNITS}")
metric_cols[2].metric("PM10", f"{current_components.get('pm10', 0):.1f} {COMPONENT_UNITS}")
metric_cols[3].metric("O3", f"{current_components.get('o3', 0):.1f} {COMPONENT_UNITS}")

comp_df = components_summary(current_components)
detail_left, detail_right = st.columns([1, 1])
with detail_left:
    comp_chart = (
        alt.Chart(comp_df)
        .mark_bar(color="#34495e")
        .encode(
            x=alt.X("value:Q", title=f"Concentration ({COMPONENT_UNITS})"),
            y=alt.Y("component:N", sort="-x", title=None),
            tooltip=["component", "value"],
        )
        .properties(height=260)
    )
    st.altair_chart(comp_chart, use_container_width=True)

with detail_right:
    st.write("Latest observation (UTC):")
    st.write(datetime.utcfromtimestamp(current_ts).strftime("%Y-%m-%d %H:%M"))
    st.write(
        "Top pollutant:",
        summary_df.loc[summary_df["city"] == selected_city, "top_pollutant"].iloc[0],
    )
    st.write("AQI description:", AQI_LABELS.get(current_aqi, "Unknown"))

forecast_df = forecast_to_df(selected_payload["forecast"])
if not forecast_df.empty:
    forecast_melt = forecast_df[["dt", "aqi", "pm2_5", "pm10", "o3", "no2"]]
    forecast_line = (
        alt.Chart(forecast_melt)
        .mark_line()
        .encode(
            x=alt.X("dt:T", title="UTC"),
            y=alt.Y("aqi:Q", scale=alt.Scale(domain=[1, 5])),
            tooltip=["dt", "aqi", "pm2_5", "pm10", "o3", "no2"],
        )
        .properties(height=260)
    )

    st.altair_chart(forecast_line, use_container_width=True)
else:
    st.info("Forecast data not available for this city.")

if show_history:
    st.subheader("Historical Trends")
    if use_demo:
        history_df = history_to_df(selected_payload.get("history", {}))
    else:
        end_ts = int(datetime.utcnow().timestamp())
        start_ts = int((datetime.utcnow() - timedelta(days=history_days)).timestamp())
        try:
            history = get_air_quality_history(
                selected_payload["loc"]["lat"],
                selected_payload["loc"]["lon"],
                start_ts,
                end_ts,
                api_key,
            )
            history_df = history_to_df(history)
        except requests.HTTPError:
            history_df = pd.DataFrame()

    if not history_df.empty:
        if rolling_hours > history_days * 24:
            st.info("Rolling window exceeds available history; line reflects available data.")
        history_df = history_df.sort_values("dt")
        history_df = history_df.set_index("dt")
        history_df["aqi_roll"] = history_df["aqi"].rolling(
            f"{rolling_hours}H", min_periods=1
        ).mean()
        latest_24h = history_df.last("24H")["aqi"].mean()
        prev_24h = history_df.iloc[:-24].last("24H")["aqi"].mean() if len(history_df) >= 48 else None
        history_plot = history_df.reset_index()[["dt", "aqi", "aqi_roll"]]
        history_melt = history_plot.melt(
            id_vars="dt", value_vars=["aqi", "aqi_roll"], var_name="series", value_name="value"
        )
        history_line = (
            alt.Chart(history_melt)
            .mark_line()
            .encode(
                x=alt.X("dt:T", title="UTC"),
                y=alt.Y("value:Q", scale=alt.Scale(domain=[1, 5])),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(domain=["aqi", "aqi_roll"], range=["#1f77b4", "#ff7f0e"]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["dt", "series", "value"],
            )
            .properties(height=260)
        )
        if prev_24h is not None and not math.isnan(prev_24h):
            delta = latest_24h - prev_24h
            trend_label = "Improving" if delta < 0 else "Worsening"
            st.markdown(
                f"""
<div class="metric-card">
  <div class="metric-label">Last 24h vs previous 24h</div>
  <div class="metric-value">{trend_label}</div>
  <div class="metric-label">Δ AQI {delta:+.2f}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        st.altair_chart(history_line, use_container_width=True)
    else:
        st.info("Historical data not available for this city or time window.")

st.caption("Data: OpenWeather Air Pollution API")
