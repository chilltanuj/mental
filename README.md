# MindScore — Mental Health Score Predictor (Demo)

A mobile-friendly Streamlit project demonstrating a supervised machine-learning regression workflow.

## Important note
This version uses **synthetic teaching data** generated inside `app.py`. It is not trained on real student records, and its output is **not a validated mental-health score, diagnosis, or clinical assessment**. Do not use it for health decisions.

## Run on a computer
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```
5. Open the local URL shown in the terminal.

## Show it on your phone
The easiest way is to deploy the folder to Streamlit Community Cloud:
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. Sign in to Streamlit Community Cloud at https://share.streamlit.io/
3. Choose your repository and select `app.py` as the main file.
4. Deploy, then open the generated app URL on your phone.

You can also run it on your PC and open it on the same Wi-Fi network by starting:
```bash
streamlit run app.py --server.address 0.0.0.0
```
Then visit `http://YOUR_PC_LOCAL_IP:8501` on your phone. Your firewall may need to allow access.

## Model details
- Algorithm: `RandomForestRegressor`
- Learning type: supervised learning
- Task: regression
- Metrics shown: MAE and R² on a held-out test split

## Suggested report description
“MindScore demonstrates a supervised machine-learning regression pipeline using a Random Forest Regressor. Lifestyle-related inputs are entered through a mobile-friendly interface, and the trained model returns an illustrative numerical estimate. The current prototype uses synthetic data and is intended only to demonstrate model training, evaluation, and inference.”
