import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import joblib
import requests
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import streamlit_js_eval
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# MICROCLIMATE AI — REFERENCE-STYLE CAMPUS INTELLIGENCE UI
# ============================================================
# Current data is synthetic prototype data. Scenario outputs are
# illustrative decision-support models, not measured engineering effects.
# ============================================================

st.set_page_config(
    page_title="MicroClimate AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "environmental_anomaly_results.csv"
MODEL_PATH = ROOT / "models" / "isolation_forest_model.pkl"

LOCATIONS = {
    "Classroom": "🏫",
    "Computer Lab": "💻",
    "Library": "📚",
    "Canteen": "🍽️",
    "Parking Area": "🚗",
    "Garden": "🌳",
}

# ============================================================
# LIVE CAMPUS MAP CONFIGURATION
# ============================================================
# Default campus center is the publicly mapped Bangalore Technological
# Institute point. The six facility coordinates below are STARTER
# coordinates for the prototype. Replace them with the exact GPS
# coordinates of your real campus rooms/areas before using navigation.
CAMPUS_CENTER = (12.88549, 77.69963)

CAMPUS_LOCATIONS = {
    "Classroom":    {"lat": 12.88540, "lon": 77.69963, "icon": "🏫", "code": "B-204"},
    "Computer Lab": {"lat": 12.88552, "lon": 77.69978, "icon": "💻", "code": "C-102"},
    "Library":     {"lat": 12.88567, "lon": 77.69955, "icon": "📚", "code": "L-101"},
    "Canteen":     {"lat": 12.88529, "lon": 77.69984, "icon": "🍽️", "code": "C-001"},
    "Parking Area":{"lat": 12.88508, "lon": 77.69953, "icon": "🚗", "code": "P-1"},
    "Garden":      {"lat": 12.88578, "lon": 77.69982, "icon": "🌳", "code": "G-01"},
}

ROUTING_ENDPOINT = "https://router.project-osrm.org"
LIVE_MAP_REFRESH_MS = 5000
FEATURES = ["temperature", "humidity", "co2", "pm25", "light_intensity"]
LABELS = {
    "temperature": "Temperature",
    "humidity": "Humidity",
    "co2": "CO₂",
    "pm25": "PM2.5",
    "light_intensity": "Light intensity",
}
UNITS = {
    "temperature": "°C",
    "humidity": "%",
    "co2": "ppm",
    "pm25": "µg/m³",
    "light_intensity": "lux",
}
COLORS = {
    "temperature": "#ffb52e",
    "humidity": "#2f9cff",
    "co2": "#24d8a0",
    "pm25": "#b26cff",
    "light_intensity": "#ffdf69",
}


def inject_css():
    st.markdown(
        """
        <style>
        :root{
            --bg:#04111e; --panel:#071b2d; --panel2:#0a2237; --line:#173e5e;
            --text:#eef7ff; --muted:#8ea6bc; --cyan:#18bfff; --green:#2be4a0;
        }
        .stApp{
            background:
              radial-gradient(circle at 75% -5%, rgba(20,164,194,.16), transparent 30%),
              radial-gradient(circle at 15% 18%, rgba(26,103,219,.13), transparent 32%),
              linear-gradient(135deg,#04101c,#061525 58%,#061a22);
            color:var(--text);
        }
        /* Hide Streamlit's native toolbar so the application owns the header. */
        header[data-testid="stHeader"]{display:none}
        #MainMenu, footer{visibility:hidden}
        .block-container{max-width:1720px;padding:0.35rem 1.15rem 2.4rem}
        [data-testid="stSidebar"]{background:linear-gradient(180deg,#03101c,#061726);border-right:1px solid #123550}
        [data-testid="stSidebar"]>div:first-child{padding:1rem .75rem}
        .brand{padding:.1rem .2rem 1rem;border-bottom:1px solid #15364f}
        .brand-row{display:flex;gap:9px;align-items:center}
        .brand-mark{font-size:31px;filter:drop-shadow(0 0 10px rgba(24,223,192,.35))}
        .brand-title{font-size:23px;font-weight:900;letter-spacing:-.5px}.brand-title span{color:#25aef7}
        .brand-sub{color:#8199af;font-size:9.5px;line-height:1.35;margin-left:40px;margin-top:-2px}
        .side-section{color:#617c95;font-size:10px;text-transform:uppercase;letter-spacing:.14em;margin:17px 0 6px}
        .side-status{margin-top:12px;padding:11px 12px;border:1px solid #135b54;background:linear-gradient(135deg,rgba(17,75,71,.5),rgba(5,29,37,.8));border-radius:12px}
        .dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#2de6a0;box-shadow:0 0 11px #2de6a0;margin-right:7px}
        .topbar{height:72px;border-bottom:1px solid #12334d;display:grid;grid-template-columns:minmax(320px,1fr) auto auto;align-items:center;gap:18px;margin-bottom:4px;padding:0 2px}
        .search{height:42px;box-sizing:border-box;border:1px solid #20577c;background:linear-gradient(180deg,#071f33,#06192a);border-radius:12px;padding:0 15px;color:#91abc0;width:100%;max-width:720px;display:flex;align-items:center;box-shadow:inset 0 1px 0 rgba(255,255,255,.025),0 8px 20px rgba(0,0,0,.12);font-size:11px}
        .search-icon{font-size:15px;color:#4faed8;margin-right:9px}.search-placeholder{flex:1}.search-shortcut{font-size:9px;color:#557188;border:1px solid #183b54;border-radius:6px;padding:3px 6px}
        .topbar-right{display:flex;align-items:center;gap:14px;justify-content:flex-end}
        .live-pill{display:flex;align-items:center;gap:7px;padding:8px 11px;border:1px solid #16574e;background:rgba(16,74,66,.28);border-radius:10px;white-space:nowrap}
        .live-pill .live-dot{color:#2ee6a0;font-size:11px}.live-copy{line-height:1.15}.live-copy b{display:block;color:#e9f8f4;font-size:10px}.live-copy span{display:block;color:#6f8b9f;font-size:8px;margin-top:2px}
        .topbar-icon{width:36px;height:36px;border:1px solid #173f5a;background:#071b2d;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:15px;color:#cce0ee}
        .profile{display:flex;align-items:center;gap:8px;padding-left:2px}.avatar{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#0f75aa,#123e6c);border:1px solid #2b7ca6;color:#fff;font-size:10px;font-weight:900}.profile-copy{line-height:1.1}.profile-copy b{display:block;color:#eaf5fb;font-size:10px}.profile-copy span{display:block;color:#68839a;font-size:8px;margin-top:3px}
        .hero{margin-top:13px;padding:15px 3px 11px;position:relative;border-radius:14px}
        .hero::before{content:"";position:absolute;inset:0;border-radius:14px;background:linear-gradient(90deg,rgba(13,46,76,.18),rgba(12,64,65,.13));pointer-events:none}
        .hero-title{font-size:27px;font-weight:900;letter-spacing:-.6px}.hero-sub{color:#8da7bd;font-size:12px;margin-top:4px}
        .filter-row{margin-top:3px}
        .card{background:linear-gradient(145deg,rgba(8,34,55,.96),rgba(5,22,38,.97));border:1px solid #174463;border-radius:12px;padding:14px;box-shadow:0 10px 26px rgba(0,0,0,.18)}
        .kpi-title{color:#93abc0;font-size:10px}.kpi-value{font-size:26px;font-weight:900;color:#f4faff;margin:3px 0}.kpi-foot{color:#718ba3;font-size:9px}.kpi-spark{font-size:18px;text-align:right;color:#20d8c0}
        .section-head{display:flex;justify-content:space-between;align-items:center;margin:17px 0 8px;padding:0 2px}.section-head + .map-card,.section-head + .alert-panel{margin-top:0}.section-title{font-size:15px;font-weight:850}.section-sub{font-size:10px;color:#7e97ae}
        .map-card{height:325px;border-radius:12px;border:1px solid #174463;position:relative;overflow:hidden;background:
          radial-gradient(circle at 18% 22%,#1c5c43 0 5%,transparent 5.5%),
          radial-gradient(circle at 70% 28%,#245b45 0 6%,transparent 6.5%),
          radial-gradient(circle at 38% 72%,#1b5842 0 7%,transparent 7.5%),
          linear-gradient(135deg,#0b2a30,#103d32 45%,#112a35);}
        .map-card:before{content:"";position:absolute;inset:0;opacity:.55;background:repeating-linear-gradient(18deg,transparent 0 18px,rgba(110,176,151,.16) 19px 20px),repeating-linear-gradient(108deg,transparent 0 34px,rgba(91,157,191,.14) 35px 36px)}
        .map-road{position:absolute;background:rgba(190,205,184,.22);height:7px;border-radius:10px;transform-origin:left center}.r1{width:95%;top:36%;left:2%;transform:rotate(11deg)}.r2{width:82%;top:67%;left:8%;transform:rotate(-13deg)}.r3{width:62%;top:18%;left:30%;transform:rotate(70deg)}
        .pin{position:absolute;transform:translate(-50%,-50%);min-width:88px;padding:6px 8px;border-radius:9px;background:rgba(3,15,25,.9);border:1px solid #31566e;box-shadow:0 8px 18px rgba(0,0,0,.28);font-size:9px;text-align:center;z-index:2}.pin b{display:block;color:#edf7ff;font-size:9px}.pin span{font-weight:800}.pin-dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
        .alert-panel{background:linear-gradient(145deg,#071c2f,#061626);border:1px solid #174463;border-radius:12px;padding:8px 13px;height:325px;overflow:auto}.alert-item{padding:9px 0;border-bottom:1px solid #17364f}.alert-item:last-child{border-bottom:0}.alert-title{font-size:10.5px;color:#e6f0f8;font-weight:750}.alert-meta{font-size:9px;color:#718ca4;margin-top:3px}.alert-action{font-size:9px;color:#99bfd7;margin-top:4px;line-height:1.35}.sev{display:inline-block;padding:2px 6px;border-radius:99px;font-size:8px;font-weight:800}.sev-high{color:#ff7c91;background:rgba(255,80,108,.13)}.sev-medium{color:#ffc66b;background:rgba(255,183,71,.12)}.sev-low{color:#66b8ff;background:rgba(60,150,255,.12)}
        .metric{background:#071b2d;border:1px solid #174463;border-radius:10px;padding:11px;min-height:105px}.metric-name{font-size:9px;color:#8ea7bd}.metric-value{font-size:19px;font-weight:900;margin-top:5px}.metric-range{font-size:8px;color:#718aa1;margin-top:5px}
        .quick{height:74px;border:1px solid #194765;border-radius:10px;background:#071b2d;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#cce0ee;font-size:9px}.quick-icon{font-size:18px}.quick:hover{border-color:#27bfff}
        .comfort{display:flex;justify-content:space-between;align-items:center}.comfort-score{font-size:30px;font-weight:900;color:#2ce3a0}.note{font-size:9px;color:#7892a8;line-height:1.45}.insight{background:linear-gradient(135deg,rgba(24,156,255,.1),rgba(34,226,166,.07));border:1px solid #1c5a78;border-radius:12px;padding:13px;color:#c7dce9;line-height:1.55;font-size:11px}
        .stTabs [data-baseweb="tab-list"]{gap:3px;background:transparent}.stTabs [data-baseweb="tab"]{color:#8ea6bc;font-size:11px}.stTabs [aria-selected="true"]{color:#2bc4ff}
        div[data-testid="stDataFrame"]{border:1px solid #173e5e;border-radius:10px;overflow:hidden}
        .foot{color:#607b93;font-size:9px;margin-top:16px;text-align:center}

        /* ========================================================
           LIVE NAVIGATION / MAP UI
           ======================================================== */
        .nav-card{
            background:linear-gradient(145deg,rgba(8,34,55,.98),rgba(5,22,38,.98));
            border:1px solid #174463;border-radius:14px;padding:12px 14px;
            box-shadow:0 12px 32px rgba(0,0,0,.22)
        }
        .nav-card-title{font-size:14px;font-weight:900}
        .nav-card-sub{font-size:9px;color:#7892a8;margin-top:3px}
        .nav-status{
            display:inline-flex;align-items:center;gap:7px;padding:6px 9px;
            border-radius:999px;border:1px solid #16574e;background:rgba(16,74,66,.28);
            color:#c9fff0;font-size:8px;font-weight:800
        }
        .nav-status-dot{
            width:7px;height:7px;border-radius:50%;background:#2de6a0;
            box-shadow:0 0 11px #2de6a0
        }
        .route-summary{
            margin-top:8px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;
            font-size:10px;color:#9cb3c5
        }
        .route-summary b{color:#eaf7ff}
        .next-turn{
            margin-top:8px;padding:10px;border:1px solid #1c5a78;border-radius:11px;
            background:linear-gradient(135deg,rgba(24,156,255,.10),rgba(34,226,166,.07))
        }
        .next-turn .label{font-size:8px;color:#6f8ba0;text-transform:uppercase;letter-spacing:.12em}
        .next-turn .turn{font-size:11px;font-weight:800;color:#e9f8ff;margin-top:3px}
        .map-shell{
            border:1px solid #174463;border-radius:14px;overflow:hidden;
            box-shadow:0 12px 34px rgba(0,0,0,.25)
        }
        .map-caption{
            padding:7px 10px;font-size:8px;color:#6f8aa1;text-align:right;background:#051827;
            border-top:1px solid #12344f
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data(path: str):
    d = pd.read_csv(path)
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    return d.dropna(subset=["timestamp"]).copy()


@st.cache_resource
def load_model(path: str):
    return joblib.load(path) if Path(path).exists() else None


def fmt_change(value):
    if not np.isfinite(value):
        return "—"
    arrow = "↑" if value > 0 else "↓" if value < 0 else "→"
    return f"{arrow} {abs(value):.1f}% vs previous period"


def period_change(data, feature):
    if len(data) < 2:
        return 0.0
    days = max((data.timestamp.dt.date.max() - data.timestamp.dt.date.min()).days + 1, 1)
    current_start = data.timestamp.max() - pd.Timedelta(days=days - 1)
    previous_end = current_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=days - 1)
    current = data.loc[data.timestamp >= current_start, feature].mean()
    previous = data.loc[(data.timestamp >= previous_start) & (data.timestamp <= previous_end), feature].mean()
    if pd.isna(previous) or previous == 0:
        return 0.0
    return (current - previous) / previous * 100


def simulated_environment(data, canopy, cool_roof):
    out = data.sort_values("timestamp").copy()
    canopy_effect = 1.5 * canopy / 50.0
    roof_effect = 0.8 if cool_roof else 0.0
    pm_effect = 0.12 * canopy / 50.0
    out["sim_temperature"] = out["temperature"] - canopy_effect - roof_effect
    out["sim_pm25"] = out["pm25"] * (1 - pm_effect)
    return out


def comfort_score(data):
    if data.empty:
        return 0.0, "No data"
    t = float(data.temperature.mean())
    rh = float(data.humidity.mean())
    wind = 1.5  # temporary prototype assumption until a real wind sensor is added
    score = np.clip(100 - abs(t - 24) * 3 - max(0, abs(rh - 55) - 10) * .55 + min(wind, 3) * 1.2, 0, 100)
    band = "Comfortable" if score >= 80 else "Warm / acceptable" if score >= 60 else "Higher thermal discomfort"
    return float(score), band


def action_for(row):
    actions = []
    if row.co2 >= 1200:
        actions.append("increase fresh-air ventilation")
    if row.temperature >= 32:
        actions.append("reduce heat load or increase cooling")
    if row.pm25 >= 75:
        actions.append("inspect particulate sources and increase filtration")
    if row.humidity >= 80:
        actions.append("increase air circulation and moisture control")
    return "🤖 AI Suggestion: " + ("; ".join(actions).capitalize() if actions else "continue monitoring and verify the reading") + "."


def insight_text(data, anomalies):
    if data.empty:
        return "No observations match the active filters."
    rate = len(anomalies) / len(data) * 100
    if anomalies.empty:
        first = f"The selected view contains {len(data):,} observations and no model-flagged anomalies."
        second = "The current observations fall inside the Isolation Forest model's learned normality region."
    else:
        delta = anomalies[FEATURES].mean() - data[FEATURES].mean()
        key = delta.abs().sort_values(ascending=False).index[0]
        direction = "higher" if delta[key] > 0 else "lower"
        first = f"The selected view contains {len(data):,} observations, with {len(anomalies):,} flagged as anomalies ({rate:.2f}%)."
        second = f"{LABELS[key]} is the largest statistical difference between the anomaly subset and the selected baseline ({direction} in anomalous observations); this is a model pattern, not proof of causation."
    third = f"Average conditions are {data.temperature.mean():.1f} °C, {data.humidity.mean():.1f}% humidity, {data.co2.mean():.0f} ppm CO₂ and {data.pm25.mean():.1f} µg/m³ PM2.5."
    return f"{first} {second} {third}"


def chart_layout(fig, height=255, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#93a9bd", size=9),
        legend=dict(orientation="h", y=-.18, x=0) if legend else dict(visible=False),
        xaxis=dict(showgrid=False, linecolor="#173750", zeroline=False),
        yaxis=dict(gridcolor="#173750", zeroline=False),
        hovermode="x unified",
    )
    return fig


# ============================================================
# LIVE LOCATION + ROUTING HELPERS
# ============================================================
def _distance_m(a, b):
    """Approximate straight-line distance between two (lat, lon) points."""
    from math import radians, sin, cos, sqrt, atan2
    r = 6371000.0
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * r * atan2(sqrt(h), sqrt(1 - h))


def browser_geolocation(refresh_tick=0):
    """
    Ask the browser for the current GPS position.

    The promise is supported by streamlit-js-eval; using a changing key
    lets the component request a fresh browser position after each
    Streamlit auto-refresh.
    """
    js = """
    new Promise((resolve) => {
        if (!navigator.geolocation) {
            resolve({
                error: {code: 0, message: "Geolocation is not supported by this browser."}
            });
            return;
        }

        navigator.geolocation.getCurrentPosition(
            (position) => resolve({
                coords: {
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude,
                    accuracy: position.coords.accuracy,
                    altitude: position.coords.altitude,
                    heading: position.coords.heading,
                    speed: position.coords.speed
                },
                timestamp: position.timestamp
            }),
            (error) => resolve({
                error: {code: error.code, message: error.message}
            }),
            {
                enableHighAccuracy: true,
                maximumAge: 3000,
                timeout: 5000
            }
        );
    })
    """
    return streamlit_js_eval(
        js_expressions=js,
        want_output=True,
        key=f"LIVE_GPS_{refresh_tick}",
    )


@st.cache_data(ttl=20, show_spinner=False)
def get_osrm_route(origin_lat, origin_lon, dest_lat, dest_lon):
    """
    Request a walking route from OSRM.

    The public OSRM HTTP API accepts coordinates in lon,lat order and can
    return full GeoJSON geometry and turn-by-turn steps.
    """
    def request_profile(profile):
        url = (
            f"{ROUTING_ENDPOINT}/route/v1/{profile}/"
            f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        )
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "true",
            "alternatives": "false",
        }
        response = requests.get(url, params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "Ok" or not payload.get("routes"):
            return None
        return payload

    # Walking is the primary mode for an indoor/campus navigation concept.
    for profile in ("foot", "driving"):
        try:
            payload = request_profile(profile)
            if payload:
                route = payload["routes"][0]
                geometry = route["geometry"]["coordinates"]
                points = [(lat, lon) for lon, lat in geometry]
                steps = []
                for step in route.get("legs", [{}])[0].get("steps", []):
                    maneuver = step.get("maneuver", {})
                    street = step.get("name") or "Campus path"
                    modifier = maneuver.get("modifier")
                    mtype = maneuver.get("type", "")
                    distance_m = float(step.get("distance", 0.0))
                    if mtype == "depart":
                        text = f"Start on {street}"
                    elif mtype == "arrive":
                        text = "You have arrived at your destination."
                    else:
                        direction = f" {modifier}" if modifier else ""
                        text = f"{mtype.replace('_', ' ').title()}{direction} on {street}"
                    steps.append({
                        "text": text,
                        "distance_m": distance_m,
                    })
                return {
                    "points": points,
                    "distance_m": float(route.get("distance", 0.0)),
                    "duration_s": float(route.get("duration", 0.0)),
                    "steps": steps,
                    "profile": profile,
                }
        except Exception:
            continue
    return None


def _destination_marker_html(name, icon, code, selected=False):
    border = "#25d9e8" if selected else "#31566e"
    glow = "0 0 22px rgba(24,191,255,.35)" if selected else "0 8px 18px rgba(0,0,0,.28)"
    return f"""
    <div style="
        transform:translate(-50%,-100%);
        display:flex;align-items:center;gap:8px;
        padding:7px 9px;border-radius:10px;
        background:rgba(3,15,25,.94);
        border:1px solid {border};
        box-shadow:{glow};
        color:#eef7ff;white-space:nowrap;font-family:Arial,sans-serif;
        ">
        <div style="
            width:30px;height:30px;border-radius:50%;
            display:flex;align-items:center;justify-content:center;
            background:linear-gradient(135deg,#087fb6,#0b2b46);
            border:1px solid #35c9ff;box-shadow:0 0 18px rgba(24,191,255,.28);
            font-size:15px;">{icon}</div>
        <div>
            <div style="font-size:10px;font-weight:900;">{name}</div>
            <div style="font-size:8px;color:#6fe7c2;margin-top:2px;">
                <span style="color:#2de6a0;">●</span> Live location · {code}
            </div>
        </div>
    </div>
    """


def _user_marker_html():
    return """
    <div style="transform:translate(-50%,-50%);">
      <div style="
        width:18px;height:18px;border-radius:50%;
        background:#19c6ff;border:3px solid #eaffff;
        box-shadow:0 0 0 7px rgba(25,198,255,.17),0 0 26px rgba(25,198,255,.9);
      "></div>
    </div>
    """


def build_live_map(selected_destination=None, live_location=None, route=None):
    """Create the dark/neon live campus map used throughout the app."""
    center = list(CAMPUS_CENTER)
    if live_location:
        center = [live_location[0], live_location[1]]

    m = folium.Map(
        location=center,
        zoom_start=18,
        tiles="CartoDB dark_matter",
        control_scale=True,
        zoom_control=True,
        scrollWheelZoom=True,
        dragging=True,
        prefer_canvas=True,
    )

    # Same visual language as the generated reference: glowing live badge,
    # compact legend and a dark information card inside the map.
    map_overlay = """
    <style>
      .mc-map-card{
        position:fixed;left:16px;top:16px;z-index:9999;
        min-width:195px;padding:11px 13px;border-radius:13px;
        background:rgba(3,15,25,.94);border:1px solid #1a5d78;
        box-shadow:0 10px 28px rgba(0,0,0,.38);
        color:#eef7ff;font-family:Arial,sans-serif;
      }
      .mc-live{font-size:11px;font-weight:900;}
      .mc-live span{color:#2de6a0;}
      .mc-sub{font-size:8px;color:#7590a7;margin-top:3px;}
      .mc-legend{margin-top:8px;font-size:8px;color:#7894ab;line-height:1.7;}
    </style>
    <div class="mc-map-card">
      <div class="mc-live"><span>●</span> Live campus navigation</div>
      <div class="mc-sub">Real GPS position + route guidance</div>
      <div class="mc-legend">
        <div>🔵 Your live location</div>
        <div>📍 Campus destination</div>
        <div>〰️ Active walking route</div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(map_overlay))

    # Campus destinations.
    for name, item in CAMPUS_LOCATIONS.items():
        selected = name == selected_destination
        popup_html = f"""
        <div style="font-family:Arial,sans-serif;min-width:170px;">
          <div style="font-size:15px;font-weight:800;color:#0a2740;">
             {item["icon"]} {name}
          </div>
          <div style="font-size:11px;color:#547084;margin-top:4px;">{item["code"]}</div>
          <div style="font-size:10px;color:#708b9e;margin-top:6px;">
             Select this place in the Navigation panel to start a route.
          </div>
        </div>
        """
        folium.Marker(
            [item["lat"], item["lon"]],
            icon=folium.DivIcon(
                html=_destination_marker_html(
                    name, item["icon"], item["code"], selected=selected
                ),
                icon_size=(210, 55),
                icon_anchor=(0, 55),
            ),
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{name} · {item['code']}",
        ).add_to(m)

        # Small glow underneath each destination.
        folium.CircleMarker(
            [item["lat"], item["lon"]],
            radius=5 if selected else 4,
            color="#25d9e8" if selected else "#1c84aa",
            fill=True,
            fill_color="#25d9e8" if selected else "#1c84aa",
            fill_opacity=0.7,
            opacity=0.7,
            weight=1,
        ).add_to(m)

    # Live user position.
    if live_location:
        user_lat, user_lon = live_location
        accuracy = st.session_state.get("gps_accuracy")
        folium.Marker(
            [user_lat, user_lon],
            icon=folium.DivIcon(
                html=_user_marker_html(),
                icon_size=(30, 30),
                icon_anchor=(15, 15),
            ),
            tooltip=f"You are here · accuracy {accuracy:.0f} m" if accuracy else "You are here",
        ).add_to(m)
        if accuracy and accuracy < 100:
            folium.Circle(
                [user_lat, user_lon],
                radius=float(accuracy),
                color="#19c6ff",
                fill=True,
                fill_color="#19c6ff",
                fill_opacity=0.05,
                opacity=0.25,
                weight=1,
            ).add_to(m)

    # Active route with a dark under-stroke + cyan core to match the neon style.
    if route and route.get("points"):
        pts = route["points"]
        folium.PolyLine(
            pts,
            color="#06233a",
            weight=11,
            opacity=0.75,
        ).add_to(m)
        folium.PolyLine(
            pts,
            color="#19c6ff",
            weight=5,
            opacity=0.95,
        ).add_to(m)
        # End marker.
        end = pts[-1]
        folium.CircleMarker(
            end,
            radius=7,
            color="#2de6a0",
            fill=True,
            fill_color="#2de6a0",
            fill_opacity=0.95,
            opacity=1,
            weight=2,
        ).add_to(m)

        if live_location and selected_destination:
            bounds = [pts[0], pts[-1]]
            m.fit_bounds(bounds, padding=(32, 32))
    elif not live_location:
        # Keep the campus center visible until the browser grants location.
        m.location = CAMPUS_CENTER

    return m


def render_live_navigation(selected_destination=None, compact=False):
    """
    Render live GPS + campus destinations + route controls.
    Returns (destination_name, route_data, live_location).
    """
    # Refresh the app every 5 seconds while the map is visible.
    refresh_tick = st_autorefresh(
        interval=LIVE_MAP_REFRESH_MS,
        limit=None,
        key="MICROCLIMATE_LIVE_MAP_REFRESH",
    )

    raw_location = browser_geolocation(refresh_tick)
    if raw_location and "coords" in raw_location:
        st.session_state["live_location"] = (
            float(raw_location["coords"]["latitude"]),
            float(raw_location["coords"]["longitude"]),
        )
        st.session_state["gps_accuracy"] = float(
            raw_location["coords"].get("accuracy") or 0.0
        )
        st.session_state["gps_timestamp"] = raw_location.get("timestamp")
        st.session_state["gps_error"] = ""
    elif raw_location and "error" in raw_location:
        st.session_state["gps_error"] = raw_location["error"].get(
            "message", "Could not read browser location."
        )

    live_location = st.session_state.get("live_location")

    nav_names = list(CAMPUS_LOCATIONS.keys())
    default_index = nav_names.index(selected_destination) if selected_destination in nav_names else 1
    chosen = st.selectbox(
        "Navigate to",
        nav_names,
        index=default_index,
        key="live_destination_select",
    )

    if compact:
        a, b = st.columns([1.4, 1])
        with a:
            st.markdown(
                f"<div class='nav-card'><div class='nav-card-title'>🧭 {chosen}</div>"
                f"<div class='nav-card-sub'>Select a campus destination and follow the live route.</div></div>",
                unsafe_allow_html=True,
            )
        with b:
            start_nav = st.button("🚶 Start Navigation", use_container_width=True, key="dash_start_nav")
    else:
        a, b, c = st.columns([1.15, 1.15, .8])
        with a:
            st.markdown(
                "<div class='nav-card'><div class='nav-card-title'>🧭 Live Campus Navigator</div>"
                "<div class='nav-card-sub'>GPS → destination → walking route</div></div>",
                unsafe_allow_html=True,
            )
        with b:
            st.markdown(
                f"<div class='nav-card'><div class='nav-card-title'>{CAMPUS_LOCATIONS[chosen]['icon']} {chosen}</div>"
                f"<div class='nav-card-sub'>{CAMPUS_LOCATIONS[chosen]['code']} · selected destination</div></div>",
                unsafe_allow_html=True,
            )
        with c:
            start_nav = st.button("🚶 Start / Recalculate", use_container_width=True, key="zone_start_nav")

    if start_nav:
        if not live_location:
            st.warning("Allow browser location access first, then press Start Navigation.")
        else:
            dest = CAMPUS_LOCATIONS[chosen]
            route = get_osrm_route(
                live_location[0], live_location[1], dest["lat"], dest["lon"]
            )
            if route:
                st.session_state["nav_destination"] = chosen
                st.session_state["nav_route"] = route
                st.session_state["nav_route_origin"] = live_location
                st.rerun()
            else:
                st.error(
                    "A route could not be found from the current GPS point. "
                    "Campus paths may not yet be mapped in the routing data."
                )

    active_destination = st.session_state.get("nav_destination")
    route = st.session_state.get("nav_route")

    # Automatically refresh route after the user moves a meaningful distance.
    if (
        live_location
        and route
        and active_destination
        and active_destination in CAMPUS_LOCATIONS
    ):
        last_origin = st.session_state.get("nav_route_origin")
        if last_origin and _distance_m(live_location, last_origin) >= 20:
            dest = CAMPUS_LOCATIONS[active_destination]
            new_route = get_osrm_route(
                live_location[0], live_location[1], dest["lat"], dest["lon"]
            )
            if new_route:
                st.session_state["nav_route"] = new_route
                st.session_state["nav_route_origin"] = live_location
                route = new_route

    # Current status panel.
    status_text = (
        "GPS connected"
        if live_location
        else "Waiting for browser GPS permission"
    )
    dot = "#2de6a0" if live_location else "#ffbd4d"
    st.markdown(
        f"<div style='margin:8px 0 7px;display:flex;justify-content:space-between;align-items:center'>"
        f"<span class='nav-status'><span class='nav-status-dot' style='background:{dot};box-shadow:0 0 11px {dot}'></span>{status_text}</span>"
        f"<span style='font-size:8px;color:#68839a'>updates every 5 seconds</span></div>",
        unsafe_allow_html=True,
    )

    if st.session_state.get("gps_error") and not live_location:
        st.info(
            f"Location permission/error: {st.session_state['gps_error']}. "
            "Use HTTPS/localhost and allow location permission."
        )

    # Map.
    map_obj = build_live_map(active_destination, live_location, route)
    st.markdown("<div class='map-shell'>", unsafe_allow_html=True)
    st_folium(
        map_obj,
        use_container_width=True,
        height=390 if compact else 540,
        key="LIVE_CAMPUS_MAP",
    )
    st.markdown(
        "<div class='map-caption'>OpenStreetMap/CARTO basemap · OSRM route engine · campus destinations are configured in this file</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if route and active_destination:
        km = route["distance_m"] / 1000.0
        minutes = max(1, round(route["duration_s"] / 60.0))
        profile = route.get("profile", "foot")
        st.markdown(
            f"<div class='nav-card' style='margin-top:8px'>"
            f"<div class='nav-card-title'>➡️ To {LOCATIONS.get(active_destination, '')} {active_destination}</div>"
            f"<div class='route-summary'><span><b>{km:.2f} km</b> route</span>"
            f"<span><b>~{minutes} min</b></span><span>mode: <b>{profile}</b></span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        next_step = None
        for step in route.get("steps", []):
            if step["text"].lower().startswith(("arrive", "you have arrived")):
                continue
            next_step = step
            break
        if next_step:
            st.markdown(
                f"<div class='next-turn'><div class='label'>Next instruction</div>"
                f"<div class='turn'>{next_step['text']} · {next_step['distance_m']:.0f} m</div></div>",
                unsafe_allow_html=True,
            )

        if live_location:
            google_url = (
                "https://www.google.com/maps/dir/?api=1"
                f"&origin={live_location[0]},{live_location[1]}"
                f"&destination={CAMPUS_LOCATIONS[active_destination]['lat']},{CAMPUS_LOCATIONS[active_destination]['lon']}"
                "&travelmode=walking"
            )
            st.markdown(
                f"<div style='margin-top:8px;font-size:9px;color:#7892a8'>"
                f"If campus paths are not present in the routing dataset, use "
                f"<a href='{google_url}' target='_blank' style='color:#28c5ff'>Google Maps walking directions</a> as a backup.</div>",
                unsafe_allow_html=True,
            )

    if live_location:
        acc = st.session_state.get("gps_accuracy", 0)
        st.caption(
            f"Live GPS: {live_location[0]:.6f}, {live_location[1]:.6f} · accuracy ~{acc:.0f} m"
        )

    return chosen, route, live_location


inject_css()
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = load_data(str(DATA_PATH))
model = load_model(str(MODEL_PATH))



# Extra controls used by the functional reference-style build.
st.markdown("""<style>
div[data-testid="stButton"] > button {border:1px solid #194765;background:#071b2d;color:#d9e9f5;border-radius:10px;font-weight:700;}
div[data-testid="stButton"] > button:hover {border-color:#20c4ff;color:#ffffff;}
.reference-link button {background:transparent!important;border:0!important;color:#20bdf6!important;padding:0!important;min-height:0!important;font-size:10px!important;}
[data-testid="stSelectbox"] label,[data-testid="stDateInput"] label {color:#7f9ab0!important;font-size:9px!important;}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {background:#071b2d;border:1px solid #174463;border-radius:10px;color:#eaf6ff;}
[data-testid="stDateInput"] input {background:#071b2d;border:1px solid #174463;color:#eaf6ff;border-radius:10px;}
</style>""", unsafe_allow_html=True)

# ============================================================
# FUNCTIONAL REFERENCE-STYLE APPLICATION
# The Dashboard page intentionally follows the reference layout.
# Other features live in the sidebar pages so the dashboard stays clean.
# ============================================================

PAGES = [
    "Dashboard", "Campus Zone", "Detection Status", "Severity", "Time Window",
    "AI Simulation", "Anomaly Lab", "Data Explorer", "Model Center"
]

inject_css()
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = load_data(str(DATA_PATH))
model = load_model(str(MODEL_PATH))

# Ensure compatibility with the final dataset/model.
if "severity" not in df.columns:
    df["severity"] = np.where(df["anomaly_label"].eq("Normal"), "Normal", "Medium")
if "anomaly_score" not in df.columns:
    df["anomaly_score"] = np.nan

# ---------------- Session state ----------------
def init_state():
    defaults = {
        "page_nav": "Dashboard",
        "global_location": "All Locations",
        "global_status": "All",
        "global_severity": "All",
        "global_dates": (df.timestamp.min().date(), df.timestamp.max().date()),
        "top_search": "",
        "insight_text": "",
        "show_all_alerts": False,
        "show_all_data": False,
        "show_all_zones": False,
        "live_location": None,
        "gps_accuracy": 0.0,
        "gps_error": "",
        "gps_timestamp": None,
        "nav_destination": None,
        "nav_route": None,
        "nav_route_origin": None,
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ---------------- Sidebar: exact project navigation + all filters ----------------
st.sidebar.markdown(
    """<div class='brand'><div class='brand-row'><div class='brand-mark'>🌿</div><div class='brand-title'>MicroClimate <span>AI</span></div></div><div class='brand-sub'>Environmental Anomaly Detection & Campus Intelligence Platform</div></div>""",
    unsafe_allow_html=True,
)

page = st.sidebar.radio("Navigation", PAGES, key="page_nav", label_visibility="collapsed")
st.sidebar.markdown("<div class='side-section'>Analytics</div>", unsafe_allow_html=True)
for item in ["Overview", "Zone Intelligence", "Anomaly Lab", "Data Explorer", "Model Center"]:
    active = (page == {"Overview":"Dashboard","Zone Intelligence":"Campus Zone","Anomaly Lab":"Anomaly Lab","Data Explorer":"Data Explorer","Model Center":"Model Center"}.get(item))
    bg = "background:linear-gradient(90deg,#0e63aa,#0a2b4a);border:1px solid #1d77b9;color:#fff;" if active else "color:#9bb1c5;"
    st.sidebar.markdown(f"<div style='padding:7px 10px;border-radius:8px;font-size:11px;{bg}'>▰ {item}</div>", unsafe_allow_html=True)

st.sidebar.markdown("<div class='side-section'>Global Filters</div>", unsafe_allow_html=True)
location_options = ["All Locations"] + list(LOCATIONS.keys())
location = st.sidebar.selectbox("📍 Campus Zone", location_options, key="global_location")
status = st.sidebar.selectbox("🛡️ Detection Status", ["All", "Normal", "Anomaly"], key="global_status")
severity = st.sidebar.selectbox("⚠️ Severity", ["All", "High", "Medium", "Low", "Normal"], key="global_severity")
min_date, max_date = df.timestamp.min().date(), df.timestamp.max().date()
dates = st.sidebar.date_input("📅 Observation dates", value=st.session_state.get("global_dates", (min_date,max_date)), min_value=min_date, max_value=max_date, key="global_dates")
if isinstance(dates, (tuple,list)) and len(dates)==2:
    start_date,end_date=dates
else:
    start_date=end_date=dates

if st.sidebar.button("↻ Reset All Filters", use_container_width=True):
    for key in ["global_location","global_status","global_severity","global_dates","top_search","insight_text"]:
        st.session_state.pop(key,None)
    st.session_state["page_nav"]="Dashboard"
    st.rerun()

st.sidebar.markdown("<div class='side-status'><span class='dot'></span><b style='font-size:11px;color:#e5fbf4'>System Online</b><div style='font-size:9px;color:#718da5;margin-top:3px'>Software monitoring active · synthetic prototype</div></div>", unsafe_allow_html=True)

# ---------------- Apply global filters ----------------
filtered = df[(df.timestamp.dt.date >= start_date) & (df.timestamp.dt.date <= end_date)].copy()
if location != "All Locations": filtered = filtered[filtered.location == location]
if status != "All": filtered = filtered[filtered.anomaly_label == status]
if severity != "All": filtered = filtered[filtered.severity == severity]

# Top search is functional but does not change the visual layout.
if st.session_state.get("top_search"):
    q = st.session_state["top_search"].strip()
    if q:
        mask = filtered.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False, na=False)).any(axis=1)
        filtered = filtered[mask]

if filtered.empty:
    st.warning("No observations match the current filters/search. Use Reset All Filters to return to the full dataset.")
    st.stop()

anomalies = filtered[filtered.anomaly_label == "Anomaly"].copy()
anomaly_rate = len(anomalies) / len(filtered) * 100
comfort, comfort_band = comfort_score(filtered)

# ---------------- Shared top bar ----------------
tb1,tb2,tb3 = st.columns([1.55,.7,.35], gap="small")
with tb1:
    st.text_input("", placeholder="Search locations, zones, or anomalies...", key="top_search", label_visibility="collapsed")
with tb2:
    st.markdown("<div class='live-pill'><span class='live-dot'>●</span><div class='live-copy'><b>Live Monitoring</b><span>Software prototype · data-driven</span></div></div>", unsafe_allow_html=True)
with tb3:
    if st.button("🔔", key="top_notification", use_container_width=True):
        st.session_state["page_nav"] = "Anomaly Lab"
        st.rerun()

st.markdown(f"<div class='hero'><div class='hero-title'>MicroClimate AI🌿 </div><div class='hero-sub'>Environmental conditions are being analyzed across <b>{'all campus zones' if location == 'All Locations' else location}</b>. Model anomaly rate for the active view: <b>{anomaly_rate:.2f}%</b>.</div></div>", unsafe_allow_html=True)

# ---------------- Reference top controls ----------------
f1,f2,f3,f4 = st.columns([1.1,1.15,1.15,.85], gap="small")
with f1:
    # Keep the visual top selector synchronized with the sidebar selector.
    st.session_state["top_location"] = location
    top_location = st.selectbox("Location", location_options, key="top_location")
    if top_location != location:
        st.session_state["global_location"] = top_location
        st.session_state["top_location"] = top_location
        st.rerun()
with f2:
    st.selectbox("System", ["All Systems Online"], key="top_system")
with f3:
    preset_options=["Custom","Last 7 Days","Last 30 Days"]
    current_days=(end_date-start_date).days+1
    default_preset="Last 7 Days" if current_days==7 else ("Last 30 Days" if current_days==30 else "Custom")
    preset=st.selectbox("Time Window", preset_options, index=preset_options.index(default_preset), key="top_window")
    if preset != "Custom":
        new_end=max_date
        span=6 if preset=="Last 7 Days" else 29
        new_start=max(min_date,new_end-pd.Timedelta(days=span).to_pytimedelta())
        if (start_date,end_date)!=(new_start,new_end):
            st.session_state["global_dates"]=(new_start,new_end)
            st.rerun()
with f4:
    st.markdown(f"<div class='card' style='height:100%;box-sizing:border-box'><div class='kpi-title'>◷ Window</div><b>{(end_date-start_date).days+1} days</b></div>", unsafe_allow_html=True)

# ---------------- KPI row ----------------
kcols=st.columns(5)
kpi_data=[
    ("Total Observations",f"{len(filtered):,}","Selected records","🌿"),
    ("Anomalies Detected",f"{len(anomalies):,}",f"{anomaly_rate:.2f}% of selected","⚠️"),
    ("Anomaly Rate",f"{anomaly_rate:.2f}%","Isolation Forest result","💧"),
    ("Avg. Temperature",f"{filtered.temperature.mean():.1f} °C",fmt_change(period_change(filtered,"temperature")),"🌡️"),
    ("Avg. CO₂",f"{filtered.co2.mean():.0f} ppm",fmt_change(period_change(filtered,"co2")),"☁️"),
]
for col,(title,value,foot,icon) in zip(kcols,kpi_data):
    with col:
        st.markdown(f"<div class='card'><div class='kpi-title'>{icon} {title}</div><div class='kpi-value'>{value}</div><div class='kpi-foot'>{foot}</div></div>",unsafe_allow_html=True)

# ---------------- DASHBOARD PAGE: exact reference composition ----------------
def dashboard_page():
    left,center,right=st.columns([1.55,1.25,.82],gap="medium")
    with left:
        st.markdown("<div class='section-head'><div><div class='section-title'>⌁ Environmental Trends</div><div class='section-sub'>Selected environmental indicators · CO₂ uses a secondary scale</div></div><div class='section-sub'>7D · 30D · Custom</div></div>",unsafe_allow_html=True)
        trend=filtered.sort_values("timestamp")
        if len(trend)>420: trend=trend.iloc[np.linspace(0,len(trend)-1,420).astype(int)]
        fig=go.Figure()
        for f in ["temperature","humidity","pm25"]:
            fig.add_trace(go.Scatter(x=trend.timestamp,y=trend[f],mode="lines",name=LABELS[f],line=dict(color=COLORS[f],width=2),hovertemplate=f"%{{x|%d %b %H:%M}}<br>{LABELS[f]}: %{{y:.1f}} {UNITS[f]}<extra></extra>"))
        fig.add_trace(go.Scatter(x=trend.timestamp,y=trend.co2,mode="lines",name="CO₂",yaxis="y2",line=dict(color=COLORS["co2"],width=2),hovertemplate="%{x|%d %b %H:%M}<br>CO₂: %{y:.0f} ppm<extra></extra>"))
        chart_layout(fig,325)
        fig.update_layout(yaxis=dict(title="Temp / Humidity / PM2.5",gridcolor="rgba(78,126,157,.20)"),yaxis2=dict(title="CO₂ (ppm)",overlaying="y",side="right",showgrid=False),legend=dict(orientation="h",y=-.12,x=0,font=dict(size=9)))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with center:
        zh=st.columns([1,.2])
        with zh[0]:
            st.markdown(
                "<div class='section-head'><div><div class='section-title'>◉ Live Campus Navigator</div>"
                "<div class='section-sub'>Real GPS position + route to a campus destination</div></div></div>",
                unsafe_allow_html=True
            )
        with zh[1]:
            if st.button("View All",key="view_zones",help="Open the full live campus navigation page"):
                st.session_state["page_nav"]="Campus Zone"; st.rerun()
        render_live_navigation(compact=True)
    with right:
        a1,a2=st.columns([1,.35])
        with a1: st.markdown("<div class='section-head'><div class='section-title'>🔔 Recent Alerts</div></div>",unsafe_allow_html=True)
        with a2:
            if st.button("View All",key="view_alerts",help="Open the full anomaly lab"):
                st.session_state["page_nav"]="Anomaly Lab"; st.rerun()
        recent=filtered[filtered.anomaly_label=="Anomaly"].sort_values("timestamp",ascending=False).head(6)
        if recent.empty:
            st.markdown("<div class='alert-panel'><div class='section-sub'>No anomalies in this view.</div></div>",unsafe_allow_html=True)
        else:
            html="<div class='alert-panel'>"
            for _,r in recent.iterrows():
                sev=str(r.severity).lower()
                html+=f"<div class='alert-item'><div class='alert-title'>● {r.location} · {r.severity} anomaly</div><div class='alert-meta'>{r.timestamp:%d %b %H:%M} · {r.temperature:.1f}°C · {r.co2:.0f} ppm <span class='sev sev-{sev}'>{r.severity}</span></div><div class='alert-action'>{action_for(r)}</div></div>"
            html+="</div>"; st.markdown(html,unsafe_allow_html=True)

    m1,m2,m3=st.columns([1.0,1.58,.82],gap="medium")
    with m1:
        st.markdown("<div class='section-head'><div class='section-title'>⌁ Key Environmental Metrics</div></div>",unsafe_allow_html=True)
        grid=st.columns(2)
        for col,f in zip(grid*2,["temperature","humidity","co2","pm25"]):
            with col: st.markdown(f"<div class='metric'><div class='metric-name'>{LABELS[f]}</div><div class='metric-value'>{filtered[f].mean():.1f} {UNITS[f]}</div><div class='metric-range'>Observed range: {filtered[f].min():.1f} – {filtered[f].max():.1f}</div></div>",unsafe_allow_html=True)
    with m2:
        h1,h2=st.columns([1,.25])
        with h1: st.markdown("<div class='section-head'><div><div class='section-title'>▤ Recent Data</div><div class='section-sub'>Latest selected observations</div></div></div>",unsafe_allow_html=True)
        with h2:
            if st.button("View All",key="view_data",help="Open the full data explorer"):
                st.session_state["page_nav"]="Data Explorer"; st.rerun()
        table=filtered.sort_values("timestamp",ascending=False).head(6)[["timestamp","location","temperature","humidity","co2","pm25","severity"]].copy()
        table.columns=["Time","Location","Temperature","Humidity","CO₂","PM2.5","Status"]
        st.dataframe(table,use_container_width=True,hide_index=True,height=205)
    with m3:
        st.markdown("<div class='section-head'><div class='section-title'>⚡ Quick Actions</div></div>",unsafe_allow_html=True)
        q1,q2=st.columns(2)
        with q1:
            if st.button("📊\nExplore Data",use_container_width=True,key="qa_data"):
                st.session_state["page_nav"]="Data Explorer"; st.rerun()
        with q2:
            if st.button("🧠\nModel Center",use_container_width=True,key="qa_model"):
                st.session_state["page_nav"]="Model Center"; st.rerun()
        q3,q4=st.columns(2)
        with q3:
            st.download_button("📄\nExport Report",insight_text(filtered,anomalies).encode("utf-8"),"microclimate_brief.txt","text/plain",use_container_width=True,key="qa_export")
        with q4:
            if st.button("✨\nGenerate Insights",use_container_width=True,key="qa_insights"):
                st.session_state["insight_text"]=insight_text(filtered,anomalies)
                st.rerun()

    b1,b2=st.columns([1,1.45],gap="medium")
    with b1:
        st.markdown("<div class='section-head'><div class='section-title'>◒ CO₂ Distribution</div></div>",unsafe_allow_html=True)
        zc=filtered.groupby("location").co2.mean().sort_values(ascending=False)
        fig=go.Figure(go.Bar(x=zc.index,y=zc.values,text=[f"{v:.0f}" for v in zc.values],textposition="outside"))
        chart_layout(fig,230,legend=False); fig.update_layout(yaxis_title="ppm")
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with b2:
        st.markdown("<div class='section-head'><div><div class='section-title'>▦ Temperature & Humidity Heatmap</div><div class='section-sub'>Average profile by campus zone</div></div></div>",unsafe_allow_html=True)
        heat=filtered.groupby("location")[["temperature","humidity"]].mean().reindex(list(LOCATIONS)).dropna()
        fig=go.Figure(go.Heatmap(z=heat.values,x=["Temperature","Humidity"],y=heat.index,colorscale="Turbo",text=np.round(heat.values,1),texttemplate="%{text}",colorbar=dict(thickness=10)))
        chart_layout(fig,230,legend=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    if st.session_state.get("insight_text"):
        st.markdown(f"<div class='insight'><b>🧠 Automated AI Insights Board</b><br><br>{st.session_state['insight_text']}</div>",unsafe_allow_html=True)

    st.markdown("<div class='section-head'><div class='section-title'>🌡️ Human Thermal Comfort</div></div>",unsafe_allow_html=True)
    st.markdown(f"<div class='card comfort'><div><div class='kpi-title'>Outdoor Comfort Score · prototype</div><div style='font-size:12px;color:#c4d8e7;margin-top:5px'>Temperature + humidity + assumed 1.5 m/s wind input</div><div class='note' style='margin-top:6px'>Official UTCI requires measured wind speed and mean radiant temperature. This score is for demonstration only.</div></div><div class='comfort-score'>{comfort:.0f}%</div></div>",unsafe_allow_html=True)

# ---------------- Other functional pages ----------------
def campus_zone_page():
    st.markdown("## 🗺️ Live Campus Navigation")
    z=filtered.groupby("location").agg(
        Observations=("location","size"),
        Anomalies=("anomaly_label",lambda x:(x=="Anomaly").sum()),
        Temperature=("temperature","mean"),
        Humidity=("humidity","mean"),
        CO2=("co2","mean"),
        PM25=("pm25","mean"),
        Light=("light_intensity","mean")
    ).reset_index()
    z["Anomaly Rate %"]=z.Anomalies/z.Observations*100

    # Navigation is now real map/GPS based. The environmental table below
    # remains part of the MicroClimate AI zone intelligence view.
    render_live_navigation(compact=False)

    st.markdown("### 📊 Campus Zone Intelligence")
    st.dataframe(z.round(2),use_container_width=True,hide_index=True)

def detection_page():
    st.markdown("## 🛡️ Detection Status")
    n=len(filtered); normal=int((filtered.anomaly_label=="Normal").sum()); anom=int((filtered.anomaly_label=="Anomaly").sum())
    c=st.columns(3); c[0].metric("Total",f"{n:,}"); c[1].metric("Normal",f"{normal:,}"); c[2].metric("Anomaly",f"{anom:,}")
    by=filtered.groupby("location").agg(Normal=("anomaly_label",lambda x:(x=="Normal").sum()),Anomaly=("anomaly_label",lambda x:(x=="Anomaly").sum())).reset_index()
    st.dataframe(by,use_container_width=True,hide_index=True)
    st.download_button("Download current detection results",filtered.to_csv(index=False).encode(),"microclimate_detection_status.csv","text/csv")

def severity_page():
    st.markdown("## ⚠️ Severity Intelligence")
    order=["High","Medium","Low","Normal"]
    counts=filtered.severity.value_counts().reindex(order,fill_value=0)
    fig=go.Figure(go.Bar(x=counts.index,y=counts.values,text=counts.values,textposition="outside")); chart_layout(fig,300,False); fig.update_layout(yaxis_title="Observations")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.dataframe(filtered.groupby(["location","severity"]).size().reset_index(name="Count"),use_container_width=True,hide_index=True)

def time_window_page():
    st.markdown("## 🕒 Time Window Analytics")
    preset=st.radio("Choose analysis window",["Current selection","Last 7 days","Last 30 days","Full dataset"],horizontal=True)
    if preset=="Full dataset": d=df.copy()
    elif preset=="Last 30 days": d=df[df.timestamp>=df.timestamp.max()-pd.Timedelta(days=29)].copy()
    elif preset=="Last 7 days": d=df[df.timestamp>=df.timestamp.max()-pd.Timedelta(days=6)].copy()
    else: d=filtered.copy()
    st.write(f"Showing {len(d):,} observations from {d.timestamp.min():%d %b %Y} to {d.timestamp.max():%d %b %Y}.")
    tr=d.sort_values("timestamp")
    if len(tr)>500: tr=tr.iloc[np.linspace(0,len(tr)-1,500).astype(int)]
    fig=go.Figure()
    for f in ["temperature","humidity","co2","pm25"]: fig.add_trace(go.Scatter(x=tr.timestamp,y=tr[f],mode="lines",name=LABELS[f],line=dict(color=COLORS[f],width=2)))
    chart_layout(fig,330); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

def simulation_page():
    st.markdown("## 🔮 Predictive / What-If Simulation")
    st.caption("Scenario outputs are illustrative decision-support estimates, not measured physical effects.")
    c1,c2=st.columns([.7,1.3])
    with c1:
        canopy=st.slider("Virtual tree canopy",0,50,0,5,key="sim_canopy")
        roof=st.toggle("Apply cool-roof coating",False,key="sim_roof")
        st.info("Prototype coefficients: maximum 1.5°C canopy effect, 0.8°C cool-roof effect and up to 12% PM2.5 reduction at 50% canopy.")
    sim=simulated_environment(filtered,canopy,roof); d=sim.sort_values("timestamp")
    if len(d)>450: d=d.iloc[np.linspace(0,len(d)-1,450).astype(int)]
    with c2:
        fig=go.Figure(); fig.add_trace(go.Scatter(x=d.timestamp,y=d.temperature,name="Current",mode="lines",line=dict(color=COLORS["temperature"],width=2))); fig.add_trace(go.Scatter(x=d.timestamp,y=d.sim_temperature,name="Simulated",mode="lines",line=dict(color="#ffffff",width=2,dash="dash"))); chart_layout(fig,270); fig.update_layout(yaxis_title="°C"); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        fig=go.Figure(); fig.add_trace(go.Scatter(x=d.timestamp,y=d.pm25,name="Current",mode="lines",line=dict(color=COLORS["pm25"],width=2))); fig.add_trace(go.Scatter(x=d.timestamp,y=d.sim_pm25,name="Simulated",mode="lines",line=dict(color=COLORS["co2"],width=2,dash="dash"))); chart_layout(fig,270); fig.update_layout(yaxis_title="µg/m³"); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.success(f"Scenario average: temperature {sim.sim_temperature.mean():.2f} °C vs {sim.temperature.mean():.2f} °C current; PM2.5 {sim.sim_pm25.mean():.2f} vs {sim.pm25.mean():.2f} µg/m³.")

def anomaly_lab_page():
    st.markdown("## 🔬 Anomaly Investigation Lab")
    st.write(f"**{len(anomalies):,} anomalies** in the current filtered view.")
    if anomalies.empty: st.success("No anomalies match the current filters."); return
    idx=st.selectbox("Inspect anomaly",anomalies.sort_values("anomaly_score").index,format_func=lambda i:f"{anomalies.loc[i,'timestamp']:%d %b %H:%M} · {anomalies.loc[i,'location']} · {anomalies.loc[i,'severity']}")
    r=anomalies.loc[idx]
    st.markdown(f"<div class='card'><div class='section-title'>{r.location} · {r.severity} anomaly</div><div class='section-sub'>Score {r.anomaly_score:.4f} · {r.temperature:.1f}°C · {r.humidity:.1f}% · {r.co2:.0f} ppm · {r.pm25:.1f} µg/m³</div><div class='alert-action'>{action_for(r)}</div></div>",unsafe_allow_html=True)
    st.dataframe(anomalies.sort_values("anomaly_score").head(500),use_container_width=True,hide_index=True)
    st.download_button("Download anomaly CSV",anomalies.to_csv(index=False).encode(),"microclimate_anomalies.csv","text/csv")

def data_explorer_page():
    st.markdown("## 🗃️ Data Explorer")
    q=st.text_input("Search records",placeholder="Computer Lab, High, 1200, 2026-01-15...",key="record_search")
    view=filtered.copy()
    if q: view=view[view.astype(str).apply(lambda col:col.str.contains(q,case=False,regex=False,na=False)).any(axis=1)]
    st.write(f"Showing **{len(view):,}** records")
    st.dataframe(view.sort_values("timestamp",ascending=False).head(2000),use_container_width=True,hide_index=True,height=520)
    st.download_button("Export selected CSV",view.to_csv(index=False).encode(),"microclimate_selected_data.csv","text/csv")

def model_center_page():
    st.markdown("## 🧠 Isolation Forest Model Center")
    c=st.columns(5); c[0].metric("Algorithm","Isolation Forest"); c[1].metric("Trees","200"); c[2].metric("Contamination","2%"); c[3].metric("Features","5"); c[4].metric("Anomalies",f"{len(anomalies):,}")
    st.write("Features: temperature, humidity, CO₂, PM2.5 and light intensity.")
    if model is not None: st.success("Model file loaded successfully.")
    else: st.error("Model file not found. Run src/train_model.py.")
    if "anomaly_score" in df.columns and df.anomaly_score.notna().any():
        fig=go.Figure(go.Histogram(x=df.anomaly_score.dropna(),nbinsx=40)); chart_layout(fig,300,False); fig.update_layout(xaxis_title="Isolation Forest decision score",yaxis_title="Count"); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.info("Anomaly scores describe unusualness relative to the prototype model; they do not prove a physical cause or safety condition.")

# ---------------- Render selected page ----------------
if page == "Dashboard": dashboard_page()
elif page == "Campus Zone": campus_zone_page()
elif page == "Detection Status": detection_page()
elif page == "Severity": severity_page()
elif page == "Time Window": time_window_page()
elif page == "AI Simulation": simulation_page()
elif page == "Anomaly Lab": anomaly_lab_page()
elif page == "Data Explorer": data_explorer_page()
elif page == "Model Center": model_center_page()

# ---------------- Persistent executive briefing/export ----------------
report_text = f"""MicroClimate AI Executive Brief
Period: {start_date} to {end_date}
Location: {location}
Observations: {len(filtered):,}
Anomalies: {len(anomalies):,}
Anomaly rate: {anomaly_rate:.2f}%
Average temperature: {filtered.temperature.mean():.2f} °C
Average humidity: {filtered.humidity.mean():.2f}%
Average CO₂: {filtered.co2.mean():.2f} ppm
Average PM2.5: {filtered.pm25.mean():.2f} µg/m³
Comfort score (prototype): {comfort:.0f}%

Important: current data are synthetic prototype observations. Scenario and comfort calculations are decision-support prototypes and require real sensor calibration before operational use.
"""
if page != "Dashboard":
    st.download_button("📄 Export Executive Brief",report_text.encode("utf-8"),"microclimate_executive_brief.txt","text/plain",key="persistent_export")

st.markdown(
    "<div class='foot'>MicroClimate AI · Live campus navigation is GPS/routing based · Environmental dataset is synthetic · "
    "Replace the starter CAMPUS_LOCATIONS coordinates with exact surveyed campus points before operational use.</div>",
    unsafe_allow_html=True
)

