"""Streamlit dashboard: the logged API requests and the Evidently data drift report.

Only a thin display layer: all the logic lives in the insurance.monitoring module.
Run: streamlit run dashboard/app.py   then open http://localhost:8501
"""

import pandas as pd
import streamlit as st

from insurance.config import FEATURES, MIN_ROWS_FOR_DRIFT_REPORT, PREDICTION_LOG_PATH
from insurance.monitoring import load_current, load_reference, run_drift_report

st.set_page_config(page_title="Insurance model monitoring", layout="wide")
st.title("Insurance charges model: monitoring")
st.button("Refresh")

reference = load_reference()
current = load_current()

left, right = st.columns(2)
left.metric("Training rows (reference)", len(reference))
right.metric("Logged API requests (current)", len(current))

if len(current) < MIN_ROWS_FOR_DRIFT_REPORT:
    st.info(
        f"The drift report needs at least {MIN_ROWS_FOR_DRIFT_REPORT} logged requests. "
        "Send some with: python -m insurance.simulate --n 50"
    )
    st.stop()

summary, report_html = run_drift_report(reference, current)

drifted, share, rows = st.columns(3)
drifted.metric("Drifted columns", f"{summary.drifted_columns} / {len(FEATURES)}")
share.metric("Share of drifted columns", f"{summary.drift_share:.0%}")
rows.metric("Requests analysed", summary.current_rows)

st.subheader("Evidently data drift report")
# Safe to embed: we generate this HTML ourselves, and the API only logs values its input validation accepted.
st.iframe(report_html, height=1100)

st.subheader("Latest predictions")
st.dataframe(pd.read_csv(PREDICTION_LOG_PATH).tail(20), width="stretch")
