import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="MindScore | Student Wellbeing Demo",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #f4fbf7 0%, #ffffff 48%, #f4f8ff 100%); }
    .block-container { max-width: 820px; padding-top: 1.2rem; padding-bottom: 2rem; }
    .hero { padding: 1.35rem; border-radius: 22px; background: linear-gradient(120deg,#0b6b50,#138a70); color: white; margin-bottom: 1rem; }
    .hero h1 { color: white; margin: 0; font-size: 2rem; }
    .hero p { color: #e2fff3; margin: .5rem 0 0 0; }
    .metric-card { background: white; border: 1px solid #e2eee8; border-radius: 16px; padding: 1rem; }
    div.stButton > button { width: 100%; border-radius: 12px; min-height: 3rem; font-weight: 700; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e2eee8; border-radius: 14px; padding: 12px; }
    .disclaimer { font-size: .86rem; color: #53665e; background: #fff8e6; border: 1px solid #f3df9f; padding: .9rem; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_demo_model():
    # Synthetic teaching data only. It is not clinical or real student data.
    rng = np.random.default_rng(42)
    n = 1200
    sleep = rng.uniform(3, 10, n)
    study = rng.uniform(0, 12, n)
    screen = rng.uniform(0, 12, n)
    exercise = rng.uniform(0, 2, n)
    stress = rng.integers(1, 11, n)
    social = rng.uniform(0, 5, n)

    # Illustrative target formula to create a reproducible demo dataset.
    # The resulting score is not a validated measure of mental health.
    score = (
        5.2 + 0.55 * (sleep - 6.5) - 0.24 * np.abs(study - 5)
        - 0.20 * (screen - 3.5) + 0.65 * exercise
        - 0.38 * (stress - 5) + 0.12 * social
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
    y = score
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=160, random_state=42, max_depth=12)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return model, metrics, list(X.columns)

model, metrics, feature_names = train_demo_model()

st.markdown("""
<div class="hero">
  <h1>🌿 MindScore</h1>
  <p>Student wellbeing score predictor · Machine Learning project demo</p>
</div>
""", unsafe_allow_html=True)

st.write("Enter a few lifestyle details to see how a trained **Random Forest Regressor** produces a score estimate.")
st.caption("Demo version · Uses generated example data, not real student records.")

with st.form("prediction_form"):
    st.subheader("Your daily routine")
    c1, c2 = st.columns(2)
    with c1:
        sleep = st.slider("Sleep (hours/day)", 3.0, 10.0, 7.0, 0.5)
        study = st.slider("Study (hours/day)", 0.0, 12.0, 5.0, 0.5)
        screen = st.slider("Screen time (hours/day)", 0.0, 12.0, 4.0, 0.5)
    with c2:
        exercise = st.slider("Exercise (hours/day)", 0.0, 2.0, 0.5, 0.25)
        stress = st.slider("Self-rated stress (1 = low, 10 = high)", 1, 10, 5)
        social = st.slider("Social / family time (hours/day)", 0.0, 5.0, 1.5, 0.5)
    submitted = st.form_submit_button("✨ Predict demo score")

if submitted:
    sample = pd.DataFrame([{
        "Sleep hours": sleep,
        "Study hours": study,
        "Screen time hours": screen,
        "Exercise hours": exercise,
        "Self-rated stress": stress,
        "Social time hours": social,
    }])
    prediction = float(np.clip(model.predict(sample)[0], 0, 10))
    st.markdown("---")
    st.subheader("Your demo result")
    st.metric("Estimated wellbeing score", f"{prediction:.1f} / 10")
    st.progress(int(round(prediction * 10)))
    if prediction >= 7:
        st.success("The demo model estimated a higher score for these inputs.")
    elif prediction >= 4:
        st.info("The demo model estimated a mid-range score for these inputs.")
    else:
        st.warning("The demo model estimated a lower score for these inputs.")
    st.caption("This output reflects an illustrative synthetic dataset and is not a psychological assessment or diagnosis.")

with st.expander("📊 About the Machine Learning model"):
    st.write("**Algorithm:** Random Forest Regressor (supervised learning, regression).")
    st.write("The model learns patterns from generated example data, then estimates a numerical score for the entered features.")
    st.write("**Important:** Because the training data is synthetic and based on an illustrative formula, the score is only for demonstrating the ML workflow. It must not be used to make decisions about anyone's mental health.")
    st.metric("Demo test MAE", f"{metrics['mae']:.2f}")
    st.metric("Demo test R²", f"{metrics['r2']:.2f}")
    st.caption(f"Training samples: {metrics['n_train']} · Test samples: {metrics['n_test']}")

with st.expander("🧠 Project workflow"):
    st.markdown("""
    1. Generate a teaching dataset (synthetic).
    2. Split the data into training and test sets.
    3. Train a Random Forest Regressor.
    4. Evaluate it with MAE and R².
    5. Use the trained model to predict a score from the form inputs.
    """)

st.markdown("""
<div class="disclaimer">
<strong>Responsible-use note:</strong> This is an academic demonstration only. It does not diagnose depression, anxiety, or any mental health condition. If you are concerned about your wellbeing, speak with a qualified professional or a trusted person.
</div>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align:center;color:#7b8983;font-size:.8rem;margin-top:1.5rem'>Made for an educational Machine Learning project</p>", unsafe_allow_html=True)
