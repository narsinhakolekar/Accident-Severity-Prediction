# app.py — Premium Accident Severity Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AccidentIQ — Severity Prediction",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark industrial theme with amber accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Barlow+Condensed:wght@300;400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

:root {
    --bg-primary:    #0d0f14;
    --bg-card:       #141720;
    --bg-elevated:   #1c2030;
    --accent-amber:  #f5a623;
    --accent-red:    #e84040;
    --accent-green:  #2ecc71;
    --accent-blue:   #4a9eff;
    --text-primary:  #eef0f5;
    --text-muted:    #7a8099;
    --border:        #252a3a;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    color: var(--text-primary);
}
.stApp {
    background-color: var(--bg-primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 15px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 6px 0;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: var(--accent-amber); }

/* Hero title */
.hero-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 72px;
    font-weight: 800;
    line-height: 0.9;
    letter-spacing: -2px;
    color: var(--text-primary);
    margin: 0;
    padding: 0;
}
.hero-accent { color: var(--accent-amber); }
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 14px;
}

/* Section header */
.section-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 34px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--text-primary);
    border-left: 4px solid var(--accent-amber);
    padding-left: 14px;
    margin: 32px 0 18px 0;
}

/* Metric cards */
.metric-row { display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    flex: 1;
    min-width: 150px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent-amber);
}
.metric-card.red::before  { background: var(--accent-red); }
.metric-card.green::before { background: var(--accent-green); }
.metric-card.blue::before  { background: var(--accent-blue); }
.metric-val {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 42px;
    font-weight: 700;
    line-height: 1;
    color: var(--text-primary);
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 4px;
}

/* Prediction result */
.pred-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    margin: 24px 0;
}
.pred-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 64px;
    font-weight: 800;
    letter-spacing: -1px;
}
.pred-fatal   { color: var(--accent-red); }
.pred-serious { color: var(--accent-amber); }
.pred-slight  { color: var(--accent-green); }
.pred-conf {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 8px;
    letter-spacing: 2px;
}

/* Prob bar */
.prob-row { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
.prob-name {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    width: 70px;
    text-align: right;
    color: var(--text-muted);
    letter-spacing: 1px;
}
.prob-bar-wrap {
    flex: 1;
    background: var(--bg-elevated);
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
}
.prob-bar { height: 100%; border-radius: 4px; }
.prob-pct {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    width: 50px;
    color: var(--text-primary);
}

/* Input form styling */
.stSelectbox label, .stNumberInput label, .stSlider label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div > input {
    background: var(--bg-elevated) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'Barlow', sans-serif !important;
}

/* Chart area */
.chart-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    margin: 12px 0;
}
.chart-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 12px;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}

/* Tip box */
.tip-box {
    background: rgba(245,166,35,0.08);
    border-left: 3px solid var(--accent-amber);
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 16px 0;
    font-size: 14px;
    color: var(--text-muted);
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Streamlit button */
.stButton > button {
    background: var(--accent-amber) !important;
    color: #000 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px 32px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Model info table */
.info-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Barlow', sans-serif;
    font-size: 14px;
}
.info-table th {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    text-align: left;
}
.info-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
}
.info-table tr:last-child td { border-bottom: none; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 1px;
}
.badge-high   { background: rgba(232,64,64,0.15);  color: var(--accent-red);   }
.badge-med    { background: rgba(245,166,35,0.15); color: var(--accent-amber); }
.badge-low    { background: rgba(46,204,113,0.15); color: var(--accent-green); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME (dark)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#141720',
    'axes.facecolor':    '#1c2030',
    'axes.edgecolor':    '#252a3a',
    'axes.labelcolor':   '#7a8099',
    'axes.titlecolor':   '#eef0f5',
    'xtick.color':       '#7a8099',
    'ytick.color':       '#7a8099',
    'text.color':        '#eef0f5',
    'grid.color':        '#252a3a',
    'grid.linestyle':    '--',
    'grid.alpha':        0.6,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'font.family':       'monospace',
})

AMBER  = '#f5a623'
RED    = '#e84040'
GREEN  = '#2ecc71'
BLUE   = '#4a9eff'
PURPLE = '#a78bfa'
COLORS = [RED, AMBER, GREEN, BLUE, PURPLE, '#f472b6', '#34d399', '#fb923c']

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model        = joblib.load('best_model.pkl')
    scaler       = joblib.load('scaler.pkl')
    le_weather   = joblib.load('le_weather.pkl')
    le_road      = joblib.load('le_road.pkl')
    le_light      = joblib.load('le_light.pkl')
    le_vehicle   = joblib.load('le_vehicle.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    return model, scaler, le_weather, le_road, le_light, le_vehicle, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv('unified_accident_data.csv')
    return df

model, scaler, le_weather, le_road, le_light, le_vehicle, feature_cols = load_models()
df = load_data()

# Decode severity
severity_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
if 'Accident_Severity' in df.columns:
    df['Severity_Label'] = df['Accident_Severity'].map(severity_map).fillna('Unknown')

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 10px 0;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:28px;
                    font-weight:800;color:#eef0f5;line-height:1;">
            ACCIDENT<span style="color:#f5a623;">IQ</span>
        </div>
        <div style="font-family:'Space Mono',monospace;font-size:9px;
                    letter-spacing:3px;color:#7a8099;margin-top:4px;">
            SEVERITY INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "",
        ["🏠  Home", "🔮  Predict Severity", "📊  EDA Dashboard", "🤖  Model Info"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:10px;
                color:#7a8099;line-height:2;letter-spacing:1px;">
    MODEL · Random Forest<br>
    ACCURACY · 69.9%<br>
    RECORDS · 59,998<br>
    FEATURES · 12<br>
    CLASSES · Fatal / Serious / Slight
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: render chart in dark card
# ─────────────────────────────────────────────────────────────────────────────
def chart_card(title, fig):
    st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠  Home":
    # Hero
    st.markdown("""
    <div style="padding:40px 0 20px 0;">
        <p class="hero-sub">● LIVE PREDICTION SYSTEM</p>
        <h1 class="hero-title">ROAD<br><span class="hero-accent">ACCIDENT</span><br>SEVERITY</h1>
        <p style="font-family:'Barlow',sans-serif;font-size:18px;
                  color:#7a8099;max-width:540px;margin-top:24px;line-height:1.7;">
            A machine-learning dashboard trained on <strong style="color:#eef0f5;">59,998</strong>
            UK road accident records. Predict severity in real time, explore patterns,
            and understand what makes roads dangerous.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    total    = len(df)
    fatal_n  = (df['Accident_Severity'] == 1).sum() if 'Accident_Severity' in df.columns else 0
    serious_n = (df['Accident_Severity'] == 2).sum() if 'Accident_Severity' in df.columns else 0
    slight_n  = (df['Accident_Severity'] == 3).sum() if 'Accident_Severity' in df.columns else 0

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-val">{total:,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        <div class="metric-card red">
            <div class="metric-val">{fatal_n:,}</div>
            <div class="metric-label">Fatal Accidents</div>
        </div>
        <div class="metric-card" style="--accent-amber:#f5a623;">
            <div class="metric-val">{serious_n:,}</div>
            <div class="metric-label">Serious Injuries</div>
        </div>
        <div class="metric-card green">
            <div class="metric-val">{slight_n:,}</div>
            <div class="metric-label">Slight Injuries</div>
        </div>
        <div class="metric-card blue">
            <div class="metric-val">69.9%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Quick overview charts
    col1, col2 = st.columns(2)

    with col1:
        if 'Severity_Label' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 4))
            counts = df['Severity_Label'].value_counts()
            bars = ax.bar(counts.index, counts.values,
                          color=[RED, AMBER, GREEN], edgecolor='none', width=0.5)
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                        f'{val:,}', ha='center', va='bottom', fontsize=10,
                        color='#eef0f5', fontfamily='monospace')
            ax.set_title('Severity Distribution', pad=12)
            ax.set_ylabel('Count')
            ax.yaxis.grid(True); ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            fig.tight_layout()
            chart_card("SEVERITY BREAKDOWN", fig)

    with col2:
        if 'hour' in df.columns and 'Accident_Severity' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 4))
            for sev, color, label in [(1, RED, 'Fatal'), (2, AMBER, 'Serious'), (3, GREEN, 'Slight')]:
                sub = df[df['Accident_Severity'] == sev]
                hourly = sub.groupby('hour').size()
                ax.plot(hourly.index, hourly.values, color=color, lw=2, label=label, alpha=0.9)
            ax.set_title('Accidents by Hour of Day', pad=12)
            ax.set_xlabel('Hour'); ax.set_ylabel('Count')
            ax.legend(fontsize=10)
            ax.yaxis.grid(True); ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            fig.tight_layout()
            chart_card("HOURLY PATTERN", fig)

    st.markdown("""
    <div class="tip-box">
        💡 Use the sidebar to navigate — try <strong>Predict Severity</strong> to get a live prediction,
        or <strong>EDA Dashboard</strong> to explore all patterns in the data.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT SEVERITY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮  Predict Severity":
    st.markdown('<h2 class="section-header">Predict Accident Severity</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#7a8099;font-size:14px;margin-bottom:28px;">
    Enter accident conditions below. The model will predict whether it's
    <span style="color:#e84040;font-weight:bold;">Fatal</span>,
    <span style="color:#f5a623;font-weight:bold;">Serious</span>, or
    <span style="color:#2ecc71;font-weight:bold;">Slight</span>.
    </p>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown("##### 🕐 Time & Road")
        c1, c2 = st.columns(2)
        with c1:
            hour = st.slider("Hour of Day", 0, 23, 8)
            speed_limit = st.selectbox("Speed Limit (mph)", [20, 30, 40, 50, 60, 70], index=1)
        with c2:
            num_vehicles = st.number_input("Number of Vehicles", 1, 20, 2)
            num_casualties = st.number_input("Number of Casualties", 1, 30, 1)

        st.markdown("##### 🌦️ Conditions")
        c3, c4 = st.columns(2)
        with c3:
            weather_options = list(le_weather.classes_)
            weather = st.selectbox("Weather", weather_options)
        with c4:
            light_options = list(le_light.classes_)
            light = st.selectbox("Light Condition", light_options)

        st.markdown("##### 🛣️ Road & Vehicle")
        c5, c6 = st.columns(2)
        with c5:
            road_options = list(le_road.classes_)
            road_type = st.selectbox("Road Type", road_options)
        with c6:
            vehicle_options = list(le_vehicle.classes_)
            vehicle_type = st.selectbox("Vehicle Type", vehicle_options)

        st.markdown("##### 👤 Driver Info")
        avg_driver_age = st.slider("Average Driver Age", 16, 90, 35)

    with col_b:
        st.markdown("##### 📊 Computed Features (Auto)")

        casualty_per_vehicle = round(num_casualties / max(num_vehicles, 1), 3)
        is_rush_hour = 1 if hour in list(range(7,10)) + list(range(16,20)) else 0
        is_night     = 1 if hour < 6 or hour >= 22 else 0

        st.markdown(f"""
        <div style="background:#1c2030;border-radius:8px;padding:16px 20px;margin-bottom:16px;">
            <div style="font-family:'Space Mono',monospace;font-size:11px;color:#7a8099;
                        letter-spacing:1px;margin-bottom:12px;">AUTO-COMPUTED</div>
            <table style="width:100%;font-family:'Barlow',sans-serif;font-size:14px;">
                <tr>
                    <td style="color:#7a8099;padding:5px 0;">Casualties / Vehicle</td>
                    <td style="text-align:right;color:#f5a623;font-weight:bold;">{casualty_per_vehicle}</td>
                </tr>
                <tr>
                    <td style="color:#7a8099;padding:5px 0;">Rush Hour?</td>
                    <td style="text-align:right;color:{'#2ecc71' if is_rush_hour else '#7a8099'};
                               font-weight:bold;">{'YES' if is_rush_hour else 'NO'}</td>
                </tr>
                <tr>
                    <td style="color:#7a8099;padding:5px 0;">Night Time?</td>
                    <td style="text-align:right;color:{'#e84040' if is_night else '#7a8099'};
                               font-weight:bold;">{'YES' if is_night else 'NO'}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Risk indicator
        risk_score = 0
        if speed_limit >= 60: risk_score += 2
        elif speed_limit >= 40: risk_score += 1
        if is_night: risk_score += 2
        if is_rush_hour: risk_score += 1
        if num_vehicles >= 3: risk_score += 1
        if avg_driver_age < 25 or avg_driver_age > 70: risk_score += 1

        risk_label = "HIGH" if risk_score >= 5 else ("MODERATE" if risk_score >= 3 else "LOW")
        risk_color = RED if risk_score >= 5 else (AMBER if risk_score >= 3 else GREEN)

        st.markdown(f"""
        <div style="background:#1c2030;border-radius:8px;padding:16px 20px;
                    border-left:4px solid {risk_color};">
            <div style="font-family:'Space Mono',monospace;font-size:9px;
                        color:#7a8099;letter-spacing:2px;margin-bottom:6px;">PRE-ASSESSMENT</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:36px;
                        font-weight:800;color:{risk_color};">
                {risk_label} RISK
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:10px;
                        color:#7a8099;margin-top:4px;">Risk score: {risk_score}/7</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ PREDICT SEVERITY")

    # PREDICTION LOGIC
    if predict_btn:
        try:
            weather_enc = le_weather.transform([weather])[0]
            road_enc    = le_road.transform([road_type])[0]
            light_enc   = le_light.transform([light])[0]
            vehicle_enc = le_vehicle.transform([vehicle_type])[0]

            input_dict = {
                'hour':                 hour,
                'casualty_per_vehicle': casualty_per_vehicle,
                'avg_driver_age':       avg_driver_age,
                'Speed_limit':          speed_limit,
                'Number_of_Vehicles':   num_vehicles,
                'Number_of_Casualties': num_casualties,
                'vehicle_enc':          vehicle_enc,
                'road_enc':             road_enc,
                'weather_enc':          weather_enc,
                'light_enc':            light_enc,
                'is_rush_hour':         is_rush_hour,
                'is_night':             is_night,
            }

            input_df = pd.DataFrame([input_dict])[feature_cols]
            proba    = model.predict_proba(input_df)[0]
            pred_idx = np.argmax(proba)
            labels   = ['Fatal', 'Serious', 'Slight']
            pred_label = labels[pred_idx]
            css_class  = f"pred-{pred_label.lower()}"

            st.markdown(f"""
            <div class="pred-box">
                <div style="font-family:'Space Mono',monospace;font-size:10px;
                            letter-spacing:3px;color:#7a8099;margin-bottom:12px;">
                    PREDICTED SEVERITY
                </div>
                <div class="pred-label {css_class}">{pred_label.upper()}</div>
                <div class="pred-conf">Confidence: {max(proba)*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            bar_colors = {'Fatal': RED, 'Serious': AMBER, 'Slight': GREEN}
            st.markdown("**Probability Breakdown**")
            for label, prob in zip(labels, proba):
                color = bar_colors[label]
                pct   = prob * 100
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-name">{label}</div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar" style="width:{pct:.1f}%;background:{color};"></div>
                    </div>
                    <div class="prob-pct">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Interpretation
            tips = {
                'Fatal':   "⚠️ Very high-risk conditions. Multiple dangerous factors detected — high speed, poor visibility, or vulnerable road users.",
                'Serious': "🟡 Moderate-to-high risk. Conditions may lead to significant injuries. Exercise extreme caution.",
                'Slight':  "✅ Lower risk profile. Standard road safety precautions are sufficient."
            }
            st.markdown(f"""
            <div class="tip-box" style="margin-top:16px;">
                {tips[pred_label]}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}. Please check your input values.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊  EDA Dashboard":
    st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)

    tabs = st.tabs(["📅 Time Patterns", "🌍 Road & Weather", "🚗 Vehicles & Speed",
                    "👤 Demographics", "🔥 Correlations"])

    # ── TAB 1: TIME PATTERNS ──────────────────────────────────────────────────
    with tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            if 'hour' in df.columns and 'Accident_Severity' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                for sev, color, label in [(1, RED, 'Fatal'), (2, AMBER, 'Serious'), (3, GREEN, 'Slight')]:
                    sub = df[df['Accident_Severity'] == sev]
                    hourly = sub.groupby('hour').size()
                    ax.fill_between(hourly.index, hourly.values, alpha=0.25, color=color)
                    ax.plot(hourly.index, hourly.values, color=color, lw=2, label=label)
                ax.axvspan(7, 9, alpha=0.1, color=AMBER, label='Rush hour')
                ax.axvspan(16, 19, alpha=0.1, color=AMBER)
                ax.set_title('Accidents by Hour — All Severity Classes')
                ax.set_xlabel('Hour of Day'); ax.set_ylabel('Accident Count')
                ax.legend(fontsize=9)
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                fig.tight_layout()
                chart_card("HOURLY DISTRIBUTION", fig)

        with col2:
            if 'Day_of_Week' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                day_names = {1:'Sun',2:'Mon',3:'Tue',4:'Wed',5:'Thu',6:'Fri',7:'Sat'}
                day_counts = df.groupby('Day_of_Week').size().rename(index=day_names)
                bars = ax.bar(day_counts.index, day_counts.values, color=COLORS[:7],
                              edgecolor='none', width=0.6)
                ax.set_title('Accidents by Day of Week')
                ax.set_ylabel('Count')
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                fig.tight_layout()
                chart_card("DAY OF WEEK", fig)
            else:
                st.info("Day_of_Week column not available.")

        col3, col4 = st.columns(2)
        with col3:
            if 'is_rush_hour' in df.columns and 'Severity_Label' in df.columns:
                fig, ax = plt.subplots(figsize=(5, 4))
                rush_data = df.groupby(['is_rush_hour','Severity_Label']).size().unstack(fill_value=0)
                rush_data.index = ['Non-Rush', 'Rush Hour']
                rush_data.plot(kind='bar', ax=ax, color=[RED, AMBER, GREEN], edgecolor='none',
                               width=0.5, stacked=False)
                ax.set_title('Rush Hour vs Non-Rush Severity')
                ax.set_xlabel(''); ax.set_ylabel('Count')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.legend(title='Severity', fontsize=9)
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                fig.tight_layout()
                chart_card("RUSH HOUR EFFECT", fig)

        with col4:
            if 'is_night' in df.columns and 'Severity_Label' in df.columns:
                fig, ax = plt.subplots(figsize=(5, 4))
                night_data = df.groupby(['is_night','Severity_Label']).size().unstack(fill_value=0)
                night_data.index = ['Daytime', 'Night']
                night_data.plot(kind='bar', ax=ax, color=[RED, AMBER, GREEN], edgecolor='none',
                                width=0.5, stacked=False)
                ax.set_title('Day vs Night Severity')
                ax.set_xlabel(''); ax.set_ylabel('Count')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.legend(title='Severity', fontsize=9)
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                fig.tight_layout()
                chart_card("DAY vs NIGHT", fig)

    # ── TAB 2: ROAD & WEATHER ─────────────────────────────────────────────────
    with tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            weather_col = None
            for c in ['Weather_Conditions', 'weather_enc', 'Weather']:
                if c in df.columns:
                    weather_col = c
                    break
            if weather_col:
                fig, ax = plt.subplots(figsize=(6, 4))
                wc = df[weather_col].value_counts().head(8)
                ax.barh(range(len(wc)), wc.values, color=BLUE, alpha=0.8, edgecolor='none')
                ax.set_yticks(range(len(wc)))
                ax.set_yticklabels([str(x)[:25] for x in wc.index], fontsize=9)
                ax.set_title('Weather Conditions Distribution')
                ax.set_xlabel('Count')
                ax.xaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                fig.tight_layout()
                chart_card("WEATHER CONDITIONS", fig)

        with col2:
            road_col = None
            for c in ['Road_Type', 'road_enc', 'Road_Surface_Conditions']:
                if c in df.columns:
                    road_col = c
                    break
            if road_col:
                fig, ax = plt.subplots(figsize=(6, 4))
                rc = df[road_col].value_counts().head(8)
                ax.barh(range(len(rc)), rc.values, color=PURPLE, alpha=0.8, edgecolor='none')
                ax.set_yticks(range(len(rc)))
                ax.set_yticklabels([str(x)[:25] for x in rc.index], fontsize=9)
                ax.set_title('Road Type Distribution')
                ax.set_xlabel('Count')
                ax.xaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                fig.tight_layout()
                chart_card("ROAD TYPES", fig)

        col3, col4 = st.columns(2)
        with col3:
            light_col = None
            for c in ['Light_Conditions', 'light_enc']:
                if c in df.columns:
                    light_col = c
                    break
            if light_col and 'Severity_Label' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                fatal_light = df[df['Accident_Severity']==1][light_col].value_counts().head(6)
                ax.bar(range(len(fatal_light)), fatal_light.values, color=RED, alpha=0.8, edgecolor='none')
                ax.set_xticks(range(len(fatal_light)))
                ax.set_xticklabels([str(x)[:15] for x in fatal_light.index], rotation=30, ha='right', fontsize=9)
                ax.set_title('Fatal Accidents by Light Condition')
                ax.set_ylabel('Count')
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                fig.tight_layout()
                chart_card("LIGHT CONDITIONS (FATAL ONLY)", fig)

        with col4:
            if 'Number_of_Casualties' in df.columns and 'Severity_Label' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                for sev, color, label in [(1, RED, 'Fatal'), (2, AMBER, 'Serious'), (3, GREEN, 'Slight')]:
                    sub = df[df['Accident_Severity']==sev]['Number_of_Casualties'].clip(0, 10)
                    ax.hist(sub, bins=10, color=color, alpha=0.6, label=label, density=True)
                ax.set_title('Casualty Distribution by Severity')
                ax.set_xlabel('Number of Casualties'); ax.set_ylabel('Density')
                ax.legend(fontsize=9)
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                fig.tight_layout()
                chart_card("CASUALTIES PER ACCIDENT", fig)

    # ── TAB 3: VEHICLES & SPEED ───────────────────────────────────────────────
    with tabs[2]:
        col1, col2 = st.columns(2)

        with col1:
            if 'Speed_limit' in df.columns and 'Severity_Label' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                speed_sev = df.groupby(['Speed_limit','Severity_Label']).size().unstack(fill_value=0)
                speed_sev.plot(kind='bar', ax=ax, color=[RED, GREEN, AMBER],
                               edgecolor='none', width=0.7, stacked=True)
                ax.set_title('Severity by Speed Limit')
                ax.set_xlabel('Speed Limit (mph)'); ax.set_ylabel('Count')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.legend(title='Severity', fontsize=9)
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                fig.tight_layout()
                chart_card("SPEED LIMIT vs SEVERITY", fig)

        with col2:
            if 'Number_of_Vehicles' in df.columns and 'Severity_Label' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                veh_clipped = df.copy()
                veh_clipped['veh_clip'] = veh_clipped['Number_of_Vehicles'].clip(1, 8)
                for sev, color, label in [(1, RED, 'Fatal'), (2, AMBER, 'Serious'), (3, GREEN, 'Slight')]:
                    sub = veh_clipped[veh_clipped['Accident_Severity']==sev]
                    counts = sub['veh_clip'].value_counts().sort_index()
                    ax.plot(counts.index, counts.values, color=color, lw=2.5, marker='o',
                            markersize=5, label=label)
                ax.set_title('Accidents by Number of Vehicles')
                ax.set_xlabel('Number of Vehicles'); ax.set_ylabel('Count')
                ax.legend(fontsize=9)
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                fig.tight_layout()
                chart_card("VEHICLE COUNT EFFECT", fig)

        col3, col4 = st.columns(2)
        with col3:
            vehicle_col = None
            for c in ['Vehicle_Type', 'vehicle_enc']:
                if c in df.columns:
                    vehicle_col = c
                    break
            if vehicle_col:
                fig, ax = plt.subplots(figsize=(6, 4))
                vc = df[vehicle_col].value_counts().head(8)
                ax.barh(range(len(vc)), vc.values, color=COLORS[:len(vc)], edgecolor='none')
                ax.set_yticks(range(len(vc)))
                ax.set_yticklabels([str(x)[:25] for x in vc.index], fontsize=9)
                ax.set_title('Vehicle Type Distribution')
                ax.set_xlabel('Count')
                ax.xaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                fig.tight_layout()
                chart_card("VEHICLE TYPES", fig)

        with col4:
            if 'casualty_per_vehicle' in df.columns and 'Severity_Label' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                for sev, color, label in [(1, RED, 'Fatal'), (2, AMBER, 'Serious'), (3, GREEN, 'Slight')]:
                    sub = df[df['Accident_Severity']==sev]['casualty_per_vehicle'].clip(0, 5)
                    ax.hist(sub, bins=15, color=color, alpha=0.6, label=label, density=True)
                ax.set_title('Casualties per Vehicle by Severity')
                ax.set_xlabel('Casualties / Vehicle'); ax.set_ylabel('Density')
                ax.legend(fontsize=9)
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                fig.tight_layout()
                chart_card("CASUALTY RATE", fig)

    # ── TAB 4: DEMOGRAPHICS ───────────────────────────────────────────────────
    with tabs[3]:
        col1, col2 = st.columns(2)

        with col1:
            if 'avg_driver_age' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                age_data = df['avg_driver_age'].clip(16, 90)
                ax.hist(age_data, bins=30, color=BLUE, edgecolor='none', alpha=0.8)
                ax.axvline(age_data.mean(), color=AMBER, lw=2, linestyle='--',
                           label=f'Mean: {age_data.mean():.1f}')
                ax.set_title('Driver Age Distribution')
                ax.set_xlabel('Age'); ax.set_ylabel('Count')
                ax.legend(fontsize=9)
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                fig.tight_layout()
                chart_card("DRIVER AGE", fig)

        with col2:
            if 'avg_driver_age' in df.columns and 'Accident_Severity' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                for sev, color, label in [(1, RED, 'Fatal'), (2, AMBER, 'Serious'), (3, GREEN, 'Slight')]:
                    sub = df[df['Accident_Severity']==sev]['avg_driver_age'].clip(16, 90)
                    ax.hist(sub, bins=20, color=color, alpha=0.5, label=label, density=True)
                ax.set_title('Age Distribution by Severity')
                ax.set_xlabel('Driver Age'); ax.set_ylabel('Density')
                ax.legend(fontsize=9)
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                fig.tight_layout()
                chart_card("AGE vs SEVERITY", fig)

        # Age group analysis
        if 'avg_driver_age' in df.columns and 'Accident_Severity' in df.columns:
            df_age = df.copy()
            bins_age = [0, 25, 35, 50, 65, 100]
            labels_age = ['<25', '25-35', '35-50', '50-65', '65+']
            df_age['age_group'] = pd.cut(df_age['avg_driver_age'], bins=bins_age,
                                          labels=labels_age, include_lowest=True)
            col3, col4 = st.columns(2)
            with col3:
                fig, ax = plt.subplots(figsize=(6, 4))
                age_grp = df_age.groupby('age_group').size()
                ax.bar(age_grp.index, age_grp.values, color=COLORS[:5], edgecolor='none', width=0.6)
                ax.set_title('Accidents by Age Group')
                ax.set_xlabel('Age Group'); ax.set_ylabel('Count')
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                fig.tight_layout()
                chart_card("ACCIDENTS BY AGE GROUP", fig)

            with col4:
                fig, ax = plt.subplots(figsize=(6, 4))
                fatal_age = df_age[df_age['Accident_Severity']==1].groupby('age_group').size()
                total_age = df_age.groupby('age_group').size()
                fatal_rate = (fatal_age / total_age * 100).fillna(0)
                ax.bar(fatal_rate.index, fatal_rate.values, color=RED, edgecolor='none', width=0.6, alpha=0.8)
                ax.set_title('Fatal Rate (%) by Age Group')
                ax.set_xlabel('Age Group'); ax.set_ylabel('Fatal %')
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                fig.tight_layout()
                chart_card("FATAL RATE BY AGE", fig)

    # ── TAB 5: CORRELATIONS ───────────────────────────────────────────────────
    with tabs[4]:
        numeric_cols = ['hour', 'Speed_limit', 'Number_of_Vehicles',
                        'Number_of_Casualties', 'avg_driver_age',
                        'casualty_per_vehicle', 'is_rush_hour', 'is_night',
                        'Accident_Severity']
        available_cols = [c for c in numeric_cols if c in df.columns]

        col1, col2 = st.columns(2)
        with col1:
            if len(available_cols) >= 3:
                fig, ax = plt.subplots(figsize=(7, 6))
                corr = df[available_cols].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                            ax=ax, vmin=-1, vmax=1, square=True,
                            annot_kws={'size': 9},
                            linewidths=0.5, linecolor='#0d0f14',
                            cbar_kws={'shrink': 0.8})
                ax.set_title('Feature Correlation Matrix')
                plt.xticks(rotation=45, ha='right', fontsize=9)
                plt.yticks(rotation=0, fontsize=9)
                fig.tight_layout()
                chart_card("CORRELATION HEATMAP", fig)

        with col2:
            if 'Speed_limit' in df.columns and 'Number_of_Casualties' in df.columns:
                fig, ax = plt.subplots(figsize=(7, 6))
                for sev, color, label in [(1, RED, 'Fatal'), (2, AMBER, 'Serious'), (3, GREEN, 'Slight')]:
                    sub = df[df['Accident_Severity']==sev].sample(min(300, len(df[df['Accident_Severity']==sev])))
                    ax.scatter(sub['Speed_limit'] + np.random.normal(0, 0.5, len(sub)),
                               sub['Number_of_Casualties'].clip(0,10),
                               color=color, alpha=0.4, s=20, label=label)
                ax.set_title('Speed Limit vs Casualties')
                ax.set_xlabel('Speed Limit (mph)')
                ax.set_ylabel('Number of Casualties')
                ax.legend(fontsize=9)
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                fig.tight_layout()
                chart_card("SPEED vs CASUALTIES (SCATTER)", fig)

        # Pairplot substitute
        if len(available_cols) >= 4:
            key_cols = ['Speed_limit', 'avg_driver_age', 'casualty_per_vehicle',
                        'Number_of_Vehicles', 'Accident_Severity']
            key_cols = [c for c in key_cols if c in df.columns][:4]
            st.markdown('<div class="chart-title">FEATURE DISTRIBUTIONS (ALL)</div>', unsafe_allow_html=True)
            fig, axes = plt.subplots(1, len(key_cols), figsize=(14, 3))
            if len(key_cols) == 1:
                axes = [axes]
            for ax, col_name in zip(axes, key_cols):
                for sev, color in [(1, RED), (2, AMBER), (3, GREEN)]:
                    sub = df[df['Accident_Severity']==sev][col_name].dropna()
                    ax.hist(sub, bins=20, color=color, alpha=0.5, density=True)
                ax.set_title(col_name.replace('_', ' '), fontsize=10)
                ax.yaxis.grid(True); ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖  Model Info":
    st.markdown('<h2 class="section-header">Model Information</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Model summary
        st.markdown("""
        <div class="chart-title">MODEL SUMMARY</div>
        <table class="info-table">
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Algorithm</td><td>Random Forest Classifier</td></tr>
            <tr><td>n_estimators</td><td>200</td></tr>
            <tr><td>max_depth</td><td>15</td></tr>
            <tr><td>Training samples</td><td>~47,998</td></tr>
            <tr><td>Test samples</td><td>12,000</td></tr>
            <tr><td>Balancing</td><td>SMOTE + class weights</td></tr>
            <tr><td>Overall Accuracy</td><td><span class="badge badge-med">69.9%</span></td></tr>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Class performance
        st.markdown("""
        <div class="chart-title" style="margin-top:16px;">CLASS PERFORMANCE</div>
        <table class="info-table">
            <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Risk</th></tr>
            <tr>
                <td><span class="badge badge-high">Fatal</span></td>
                <td>0.04</td><td>0.18</td><td>0.07</td>
                <td><span class="badge badge-high">High</span></td>
            </tr>
            <tr>
                <td><span class="badge badge-med">Serious</span></td>
                <td>0.31</td><td>0.34</td><td>0.33</td>
                <td><span class="badge badge-med">Medium</span></td>
            </tr>
            <tr>
                <td><span class="badge badge-low">Slight</span></td>
                <td>0.86</td><td>0.76</td><td>0.81</td>
                <td><span class="badge badge-low">Low</span></td>
            </tr>
        </table>
        <div class="tip-box" style="margin-top:12px;font-size:12px;">
            ℹ️ Fatal class has low precision due to severe class imbalance
            (129 Fatal vs 10,395 Slight in test set). This is a known challenge
            in accident severity prediction.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Feature importance chart
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            fig, ax = plt.subplots(figsize=(5, 5))
            colors_fi = [AMBER if i == 0 else BLUE for i in range(len(feature_cols))]
            bars = ax.barh(range(len(feature_cols)),
                           [importances[i] for i in reversed(indices)],
                           color=[AMBER if j == 0 else BLUE for j in range(len(feature_cols))],
                           edgecolor='none', alpha=0.85)
            ax.set_yticks(range(len(feature_cols)))
            ax.set_yticklabels([feature_cols[i] for i in reversed(indices)], fontsize=9)
            ax.set_title('Feature Importance', pad=10)
            ax.set_xlabel('Importance Score')
            ax.xaxis.grid(True); ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            fig.tight_layout()
            chart_card("FEATURE IMPORTANCE", fig)
        except Exception as e:
            st.error(f"Could not load feature importance: {e}")

    # Confusion matrix visualization
    st.markdown('<div class="chart-title" style="margin-top:24px;">CONFUSION MATRIX</div>',
                unsafe_allow_html=True)

    conf_path = 'confusion_matrix_v2.png'
    alt_path  = 'confusion_matrix_tuned.png'

    if os.path.exists(conf_path):
        st.image(conf_path, use_container_width=True)
    elif os.path.exists(alt_path):
        st.image(alt_path, use_container_width=True)
    else:
        # Generate it fresh
        try:
            from sklearn.metrics import confusion_matrix as cm_fn
            X_test_full = pd.DataFrame(columns=feature_cols)
            st.info("No saved confusion matrix found. Re-run the training notebook to generate one.")
        except Exception:
            st.info("Run the model training notebook to generate confusion matrix images.")

    # Feature importance (text table)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">FEATURE RANKING</div>', unsafe_allow_html=True)

    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fi_df = pd.DataFrame({
            'Rank': range(1, len(feature_cols)+1),
            'Feature': [feature_cols[i] for i in indices],
            'Importance': [f"{importances[i]:.4f}" for i in indices],
            'Bar': ['█' * int(importances[i]*80) for i in indices]
        })
        st.dataframe(fi_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Feature importance table error: {e}")

    # Pipeline summary
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="chart-title">DATA PIPELINE</div>
    <table class="info-table">
        <tr><th>Stage</th><th>Details</th><th>Status</th></tr>
        <tr><td>Data Sources</td><td>UK STATS19 accident datasets (merged)</td>
            <td><span class="badge badge-low">✅ Done</span></td></tr>
        <tr><td>Cleaning</td><td>Null removal, type conversion, deduplication</td>
            <td><span class="badge badge-low">✅ Done</span></td></tr>
        <tr><td>Feature Eng.</td><td>hour, rush_hour, night, casualty_per_vehicle, avg_driver_age</td>
            <td><span class="badge badge-low">✅ Done</span></td></tr>
        <tr><td>Encoding</td><td>LabelEncoder for weather, road, light, vehicle</td>
            <td><span class="badge badge-low">✅ Done</span></td></tr>
        <tr><td>Scaling</td><td>StandardScaler on numeric features</td>
            <td><span class="badge badge-low">✅ Done</span></td></tr>
        <tr><td>Balancing</td><td>SMOTE oversampling on training set</td>
            <td><span class="badge badge-low">✅ Done</span></td></tr>
        <tr><td>Training</td><td>Random Forest, 200 trees</td>
            <td><span class="badge badge-low">✅ Done</span></td></tr>
    </table>
    """, unsafe_allow_html=True)