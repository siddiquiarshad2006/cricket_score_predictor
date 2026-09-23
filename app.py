import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="T20 Cricket Score Predictor",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 50% 10%, #0d2818 0%, #05140b 45%, #020805 100%);
        color: #e8f5e9;
    }

    /* Hero Header */
    .hero-container {
        text-align: center;
        padding: 1.5rem 1rem 1rem 1rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.03) 100%);
        border: 1px solid rgba(52, 211, 153, 0.2);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #34d399 0%, #a7f3d0 50%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        color: #a7f3d0;
        font-size: 0.95rem;
        font-weight: 400;
        opacity: 0.85;
    }

    /* Input Card Container */
    .glass-card {
        background: rgba(13, 40, 24, 0.6);
        border: 1px solid rgba(52, 211, 153, 0.15);
        border-radius: 14px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.5rem;
    }

    /* Labels and Selectors */
    label {
        color: #d1fae5 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.01em;
    }

    /* Styled Prediction Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.03em;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
        transition: all 0.3s ease;
        margin-top: 1rem;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        box-shadow: 0 0 30px rgba(52, 211, 153, 0.6);
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.4);
    }

    /* Prediction Result Box */
    .result-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.3) 100%);
        border: 2px solid #34d399;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
        animation: fadeIn 0.5s ease-in-out;
        box-shadow: 0 0 35px rgba(52, 211, 153, 0.25);
    }

    .result-title {
        color: #a7f3d0;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    .result-score {
        font-size: 3.5rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
        text-shadow: 0 0 25px rgba(52, 211, 153, 0.8);
        line-height: 1;
        margin: 0.5rem 0;
    }

    .stats-row {
        display: flex;
        justify-content: space-around;
        margin-top: 1.2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(52, 211, 153, 0.2);
    }

    .stat-item {
        text-align: center;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #6ee7b7;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stat-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Backward and forward compatibility for numpy._core deserialization
import sys
try:
    import numpy._core
except ImportError:
    try:
        import numpy.core as _core
        sys.modules['numpy._core'] = _core
        sys.modules['numpy._core.numeric'] = _core.numeric
        if hasattr(_core, 'multiarray'):
            sys.modules['numpy._core.multiarray'] = _core.multiarray
        if hasattr(_core, 'umath'):
            sys.modules['numpy._core.umath'] = _core.umath
    except Exception:
        pass

# Load model pipeline
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    st.stop()

teams = [
    'Australia',
    'India',
    'Bangladesh',
    'New Zealand',
    'South Africa',
    'England',
    'West Indies',
    'Afghanistan',
    'Pakistan',
    'Sri Lanka'
]

cities = [
    'Colombo',
    'Mirpur',
    'Johannesburg',
    'Dubai',
    'Auckland',
    'Cape Town',
    'London',
    'Pallekele',
    'Barbados',
    'Sydney',
    'Melbourne',
    'Durban',
    'St Lucia',
    'Wellington',
    'Lauderhill',
    'Hamilton',
    'Centurion',
    'Manchester',
    'Abu Dhabi',
    'Mumbai',
    'Nottingham',
    'Southampton',
    'Mount Maunganui',
    'Chittagong',
    'Kolkata',
    'Lahore',
    'Delhi',
    'Nagpur',
    'Chandigarh',
    'Adelaide',
    'Bangalore',
    'St Kitts',
    'Cardiff',
    'Christchurch',
    'Trinidad'
]

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🏏 T20 Cricket Score Predictor</div>
    <div class="hero-subtitle">High-precision Machine Learning pipeline trained on historical international T20 matches</div>
</div>
""", unsafe_allow_html=True)

# Helper function for columns compatible across Streamlit versions
def get_columns(num):
    if hasattr(st, 'columns'):
        return st.columns(num)
    elif hasattr(st, 'beta_columns'):
        return st.beta_columns(num)
    return [st] * num

# Match Setup Section
col1, col2 = get_columns(2)

with col1:
    batting_team = st.selectbox('🏏 Batting Team', sorted(teams), index=sorted(teams).index('India') if 'India' in teams else 0)
with col2:
    bowling_team = st.selectbox('🎯 Bowling Team', sorted(teams), index=sorted(teams).index('Australia') if 'Australia' in teams else 1)

city = st.selectbox('📍 Match Venue / City', sorted(cities), index=sorted(cities).index('Melbourne') if 'Melbourne' in cities else 0)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Live Match Match Statistics
col3, col4, col5 = get_columns(3)

with col3:
    current_score = st.number_input('📊 Current Score', min_value=0, max_value=350, value=75, step=1)
with col4:
    overs = st.number_input('⏱️ Overs Completed (> 5)', min_value=5.1, max_value=19.5, value=10.0, step=0.1, format="%.1f")
with col5:
    wickets = st.number_input('❌ Wickets Out (0-9)', min_value=0, max_value=9, value=2, step=1)

last_five = st.number_input('🔥 Runs Scored in Last 5 Overs', min_value=0, max_value=150, value=38, step=1)

# Prediction Logic
if st.button('🚀 Predict Final Score'):
    # Calculate ball and over metrics accurately
    overs_int = int(overs)
    balls_part = int(round((overs - overs_int) * 10))
    balls_bowled = (overs_int * 6) + balls_part
    balls_left = max(0, 120 - balls_bowled)
    wickets_left = max(0, 10 - wickets)
    
    if balls_bowled > 0:
        crr = (current_score * 6) / balls_bowled
    else:
        crr = 0.0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'current_score': [current_score],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'crr': [crr],
        'last_five': [last_five]
    })

    try:
        prediction = pipe.predict(input_df)
        predicted_score = max(current_score, int(round(prediction[0])))
        required_rrr = ((predicted_score - current_score) * 6 / balls_left) if balls_left > 0 else 0

        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">Projected Final Score</div>
            <div class="result-score">{predicted_score}</div>
            <div style="color: #a7f3d0; font-size: 0.95rem;">
                Estimated Range: <b>{predicted_score - 5} - {predicted_score + 5} runs</b>
            </div>
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-label">Current Run Rate</div>
                    <div class="stat-value">{crr:.2f}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Balls Remaining</div>
                    <div class="stat-value">{balls_left}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Wickets in Hand</div>
                    <div class="stat-value">{wickets_left}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Projected RR (Last {balls_left//6} ov)</div>
                    <div class="stat-value">{required_rrr:.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as err:
        st.error(f"Prediction failed: {err}")
