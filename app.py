
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="MindScore | Wellbeing Predictor",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# CUSTOM CSS - MOBILE FRIENDLY & HIGH CONTRAST
# --------------------------------------------------
st.markdown("""
<style>
    :root {
        color-scheme: light;
    }

    .stApp {
        background: linear-gradient(
            180deg, #f0faf5 0%, #ffffff 55%, #f3f7ff 100%
        );
        color: #20352d !important;
    }

    [data-testid="stAppViewContainer"] {
        background: transparent;
    }

    .block-container {
        max-width: 900px;
        padding: 1.2rem 1rem 2.5rem 1rem;
    }

    /* Main heading */
    .hero {
        padding: 1.5rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #08684e, #12866c);
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 25px rgba(8, 104, 78, 0.12);
    }

    .hero h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: clamp(1.7rem, 5vw, 2.2rem);
    }

    .hero p {
        color: #e5fff4 !important;
        margin: 0.6rem 0 0 0;
        font-size: 1rem;
    }

    /* Readable text throughout the app */
    .stApp p,
    .stApp label,
    .stApp li,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp [data-testid="stMarkdownContainer"] {
        color: #20352d;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3 {
        color: #173d30 !important;
    }

    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p {
        color: #53665e !important;
    }

    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stWidgetLabel"] label {
        color: #20352d !important;
        font-weight: 600;
    }

    /* Input cards and metrics */
    .stApp [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dcebe3;
        border-radius: 14px;
        padding: 14px;
    }

    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricLabel"] p {
        color: #52665d !important;
    }

    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricValue"] div {
        color: #126b50 !important;
    }

    /* Prediction button */
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 3rem;
        border-radius: 12px;
        background: #0d7256 !important;
        border: 1px solid #0d7256 !important;
        color: #ffffff !important;
        font-weight: 700;
    }

    div[data-testid="stFormSubmitButton"] button p,
    div[data-testid="stFormSubmitButton"] button span {
        color: #ffffff !important;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #095941 !important;
        border-color: #095941 !important;
    }

    /* Responsible-use notice */
    .disclaimer {
        background: #fff8df;
        border: 1px solid #f0dda0;
        border-radius: 12px;
        padding: 1rem;
        color: #5c4a16 !important;
        font-size: 0.9rem;
    }

    .disclaimer strong {
        color: #4b3b0d !important;
    }

    @media (max-width: 600px) {
        .block-container {
            padding: 0.8rem 0.85rem 2rem 0.85rem;
        }

        .hero {
            padding: 1.15rem;
            border-radius: 17px;
        }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TRAIN RANDOM FOREST REGRESSION MODEL
# --------------------------------------------------
@st.cache_resource
def train_model():
    # Synthetic data for educational demonstration only.
    rng = np.random.default_rng(42)
    n = 1200

    sleep = rng.uniform(3, 10, n)
    study = rng.uniform(0, 12, n)
    screen = rng.uniform(0, 12, n)
    exercise = rng.uniform(0, 2, n)
    stress = rng.integers(1, 11, n)
    social = rng.uniform(0, 5, n)

    # Illustrative target formula, not a clinical scoring system.
    score = (
        5.2
        + 0.55 * (sleep - 6.5)
        - 0.24 * np.abs(study - 5)
        - 0.20 * (screen - 3.5)
        + 0.65 * exercise
        - 0.38 * (stress - 5)
        + 0.12 * social
        + rng.normal(0, 0.9, n)
    )

    score = np.clip(score, 0, 10)

    X = pd.DataFrame({
        "Sleep hours": sleep,
        "Study hours": study,
        "Screen time hours": screen,
        "Exercise hours": exercise,
        "Self-rated stress": stress,
        "Social time hours": social,
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X, score, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=160,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return model, metrics


model, metrics = train_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🌿 MindScore</h1>
    <p>Student Wellbeing Score Predictor · Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Enter your daily routine details to see how a "
    "**Random Forest Regressor** generates a score estimate."
)

st.caption(
    "Academic demonstration · Uses synthetic example data, "
    "not real student records."
)


# --------------------------------------------------
# USER INPUT FORM
# --------------------------------------------------
with st.form("prediction_form"):

    st.subheader("Your Daily Routine")

    left, right = st.columns(2)

    with left:
        sleep = st.slider(
            "Sleep (hours/day)",
            min_value=3.0,
            max_value=10.0,
            value=7.0,
            step=0.5,
        )

        study = st.slider(
            "Study (hours/day)",
            min_value=0.0,
            max_value=12.0,
            value=5.0,
            step=0.5,
        )

        screen = st.slider(
            "Screen time (hours/day)",
            min_value=0.0,
            max_value=12.0,
            value=4.0,
            step=0.5,
        )

    with right:
        exercise = st.slider(
            "Exercise (hours/day)",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.25,
        )

        stress = st.slider(
            "Self-rated stress (1 = low, 10 = high)",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
        )

        social = st.slider(
            "Social / family time (hours/day)",
            min_value=0.0,
            max_value=5.0,
            value=1.5,
            step=0.5,
        )

    submitted = st.form_submit_button(
        "✨ Predict Demo Score"
    )


# --------------------------------------------------
# PREDICTION RESULT
# --------------------------------------------------
if submitted:

    sample = pd.DataFrame([{
        "Sleep hours": sleep,
        "Study hours": study,
        "Screen time hours": screen,
        "Exercise hours": exercise,
        "Self-rated stress": stress,
        "Social time hours": social,
    }])

    prediction = float(
        np.clip(model.predict(sample)[0], 0, 10)
    )

    st.divider()
    st.subheader("Your Demo Result")

    st.metric(
        label="Estimated Wellbeing Score",
        value=f"{prediction:.1f} / 10",
    )

    st.progress(int(round(prediction * 10)))

    if prediction >= 7:
        st.success(
            "The demonstration model estimated a higher score "
            "for these inputs."
        )
    elif prediction >= 4:
        st.info(
            "The demonstration model estimated a mid-range score "
            "for these inputs."
        )
    else:
        st.warning(
            "The demonstration model estimated a lower score "
            "for these inputs."
        )

    st.caption(
        "This is an illustrative model output, not a psychological "
        "assessment or diagnosis."
    )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------
with st.expander("📊 About the Machine Learning Model"):

    st.write("**Algorithm:** Random Forest Regressor")
    st.write("**Learning type:** Supervised Learning")
    st.write("**Task:** Regression")

    st.write(
        "The model learns patterns from synthetic example data "
        "and estimates a numerical value for the submitted inputs."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Test MAE", f"{metrics['mae']:.2f}")

    with col2:
        st.metric("Test R²", f"{metrics['r2']:.2f}")

    st.caption(
        f"Training samples: {metrics['train_size']} | "
        f"Test samples: {metrics['test_size']}"
    )

    st.warning(
        "These evaluation metrics measure performance on synthetic "
        "data only. They do not demonstrate clinical validity."
    )


# --------------------------------------------------
# PROJECT WORKFLOW
# --------------------------------------------------
with st.expander("🧠 Project Workflow"):

    st.markdown("""
    1. Generate an illustrative dataset.
    2. Split data into training and testing sets.
    3. Train a Random Forest Regression model.
    4. Evaluate predictions using MAE and R².
    5. Accept user inputs and predict a demo score.
    """)


# --------------------------------------------------
# RESPONSIBLE-USE NOTICE
# --------------------------------------------------
st.markdown("""
<div class="disclaimer">
    <strong>Responsible-use note:</strong>
    This is an academic Machine Learning demonstration.
    It does not diagnose depression, anxiety, or any other
    mental health condition. The data is synthetic, and the
    score is not clinically validated. For concerns about
    mental health, consult a qualified professional.
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;color:#718078;"
    "font-size:0.8rem;margin-top:1.5rem;'>"
    "MindScore · Educational Machine Learning Project"
    "</p>",
    unsafe_allow_html=True,
)
