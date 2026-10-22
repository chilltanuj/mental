import os, zipfile, textwrap

base = "/mnt/data/mental_health_score_predictor_fixed"
os.makedirs(base, exist_ok=True)

app = r'''
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

# High-contrast styling. Keep text colors explicit so they remain readable
# even when the host Streamlit theme is dark or applies unexpected defaults.
st.markdown("""
<style>
:root {
  color-scheme: light;
}
.stApp {
  background: linear-gradient(180deg, #f2faf6 0%, #ffffff 55%, #f3f7ff 100%);
  color: #172b24 !important;
}
[data-testid="stAppViewContainer"] {
  background: transparent;
}
[data-testid="stHeader"] { background: rgba(255,255,255,0.85); }
.block-container {
  max-width: 900px;
  padding-top: 1.2rem;
  padding-bottom: 2.5rem;
  padding-left: 1rem;
  padding-right: 1rem;
}
.stApp, .stApp p, .stApp label, .stApp li, .stApp span,
.stApp div[data-testid="stMarkdownContainer"] {
  color: #20352d;
}
.hero {
  padding: 1.4rem 1.5rem;
  border-radius: 22px;
  background: linear-gradient(120deg, #08684e, #12866c);
  color: #ffffff !important;
  margin-bottom: 1rem;
  box-shadow: 0 10px 28px rgba(8, 104, 78, 0.14);
}
.hero h1, .hero p, .hero span {
  color: #ffffff !important;
}
.hero h1 { margin: 0; font-size: clamp(1.7rem, 5vw, 2.2rem); }
.hero p { margin: .55rem 0 0 0; color: #e5fff4 !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  color: #173d30 !important;
}
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p {
  color: #53665e !important;
}
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label,
.stApp label {
  color: #20352d !important;
  font-weight: 600;
}
.stApp input, .stApp textarea {
  color: #172b24 !important;
}
.stApp [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
  color: #20352d !important;
}
.stApp [data-testid="stSlider"] [data-baseweb="slider"] div {
  /* leave Streamlit's slider mechanics intact */
}
.stApp [data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid #dcebe3;
  border-radius: 14px;
  padding: 12px 14px;
}
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricLabel"] p {
  color: #52665d !important;
}
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricValue"] div {
  color: #126b50 !important;
}
div.stButton > button, div[data-testid="stFormSubmitButton"] button {
  width: 100%;
  min-height: 3rem;
  border-radius: 12px;
  font-weight: 700;
  background: #0d7256 !important;
  border: 1px solid #0d7256 !important;
  color: #ffffff !important;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
  background: #095941 !important;
  border-color: #095941 !important;
  color: #ffffff !important;
}
.stApp [data-testid="stAlert"] p { color: inherit !important; }
.disclaimer {
  font-size: .9rem;
  color: #5c4a16 !important;
  background: #fff8df;
  border: 1px solid #f0dda0;
  padding: .95rem;
  border-radius: 12px;
}
.disclaimer strong { color: #4b3b0d !important; }
@media (max-width: 600px) {
  .block-container { padding-top: .7rem; padding-left: .85rem; padding-right: .85rem; }
  .hero { padding: 1.1rem; border-radius: 17px; }
  [data-testid="stHorizontalBlock"] { gap: .7rem; }
}

.stApp,
.stApp p,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp span {
    color: #20352d !important;
}

.hero h1,
.hero p {
    color: #ffffff !important;
}

.stApp [data-testid="stCaptionContainer"] p {
    color: #53665e !important;
}

div[data-testid="stFormSubmitButton"] button {
    background-color: #0d7256 !important;
    color: #ffffff !important;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_demo_model():
    # Synthetic teaching data only; not real student or clinical data.
    rng = np.random.default_rng(42)
    n = 1200
    sleep = rng.uniform(3, 10, n)
    study = rng.uniform(0, 12, n)
    screen = rng.uniform(0, 12, n)
    exercise = rng.uniform(0, 2, n)
    stress = rng.integers(1, 11, n)
    social = rng.uniform(0, 5, n)

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, score, test_size=0.2, random_state=42
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
    return model, metrics

model, metrics = train_demo_model()

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
    left, right = st.columns(2)
    with left:
        sleep = st.slider("Sleep (hours/day)", 3.0, 10.0, 7.0, 0.5)
        study = st.slider("Study (hours/day)", 0.0, 12.0, 5.0, 0.5)
        screen = st.slider("Screen time (hours/day)", 0.0, 12.0, 4.0, 0.5)
    with right:
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
    st.divider()
    st.subheader("Your demo result")
    st.metric("Estimated wellbeing score", f"{prediction:.1f} / 10")
    st.progress(int(round(prediction * 10)))
    if prediction >= 7:
        st.success("The demo model estimated a higher score for these inputs.")
    elif prediction >= 4:
        st.info("The demo model estimated a mid-range score for these inputs.")
    else:
        st.warning("The demo model estimated a lower score for these inputs.")
    st.caption("This output reflects synthetic demonstration data and is not a psychological assessment or diagnosis.")

with st.expander("📊 About the Machine Learning model"):
    st.write("**Algorithm:** Random Forest Regressor (supervised learning, regression).")
    st.write("The model learns patterns from generated example data, then estimates a numerical score for the entered features.")
    st.write("Because the training data is synthetic and based on an illustrative formula, the score is only for demonstrating the ML workflow.")
    st.metric("Demo test MAE", f"{metrics['mae']:.2f}")
    st.metric("Demo test R²", f"{metrics['r2']:.2f}")
    st.caption(f"Training samples: {metrics['n_train']} · Test samples: {metrics['n_test']}")

with st.expander("🧠 Project workflow"):
    st.markdown("""
    1. Generate a teaching dataset (synthetic).
    2. Split the data into training and test sets.
    3. Train a Random Forest Regressor.
    4. Evaluate it with MAE and R².
    5. Predict a score from the form inputs.
    """)

st.markdown("""
<div class="disclaimer">
<strong>Responsible-use note:</strong> This is an academic demonstration only. It does not diagnose depression, anxiety, or any mental health condition. If you are concerned about your wellbeing, speak with a qualified professional or a trusted person.
</div>
""", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#718078;font-size:.8rem;margin-top:1.5rem'>Made for an educational Machine Learning project</p>", unsafe_allow_html=True)
'''.strip() + "\n"

requirements = "streamlit>=1.36\npandas>=2.0\nnumpy>=1.24\nscikit-learn>=1.3\n"
readme = """# MindScore — Contrast-Fixed Mobile Demo

Replace your previous `app.py` with this version. It keeps the same Random Forest demo and improves text contrast for mobile and desktop.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

If deployed on Streamlit Community Cloud, commit and push the updated `app.py`, then refresh/reboot the app.

**Note:** This is a teaching demo trained on synthetic data, not a clinically valid mental-health assessment.
"""

for name, content in {"app.py": app, "requirements.txt": requirements, "README.md": readme}.items():
    with open(os.path.join(base, name), "w", encoding="utf-8") as f:
        f.write(content)

zip_path = "/mnt/data/MindScore_Contrast_Fix.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ["app.py", "requirements.txt", "README.md"]:
        z.write(os.path.join(base, name), arcname=f"mental_health_score_predictor/{name}")

print("Created corrected project ZIP:", zip_path)
print("Also created updated app.py:", os.path.join(base, "app.py"))
