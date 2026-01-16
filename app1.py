
# FULL app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import Optional

st.set_page_config(page_title="Spam Detector", page_icon="📧")

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def infer_text_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["text", "message", "Message", "email", "Email", "body", "content", "sms", "v2"]:
        if c in df.columns:
            return c
    obj = [c for c in df.columns if df[c].dtype == object]
    if obj:
        avg = {c: df[c].astype(str).str.len().mean() for c in obj}
        return max(avg, key=avg.get)
    return None

st.title("📧 Spam Detector")

model = load_model("spam_model.pkl")

tab1, tab2 = st.tabs(["Single", "Batch"])

with tab1:
    text = st.text_area("Enter message", height=150)
    if st.button("Predict"):
        if text.strip():
            ptext = text.lower()
            pred = model.predict([ptext])[0]
            proba = model.predict_proba([ptext])[0]
            idx = list(model.named_steps["clf"].classes_).index("spam")
            conf = proba[idx] if pred == "spam" else 1 - proba[idx]
            st.subheader(f"{'🚨 SPAM' if pred=='spam' else '✅ HAM'}")
            st.caption(f"Confidence: {conf:.2%}")

with tab2:
    file = st.file_uploader("Upload CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        col = infer_text_column(df)
        df[col] = df[col].astype(str).str.lower()
        preds = model.predict(df[col])
        proba = model.predict_proba(df[col])
        idx = list(model.named_steps["clf"].classes_).index("spam")
        conf = [p[idx] if preds[i]=="spam" else 1-p[idx] for i,p in enumerate(proba)]
        df["prediction"] = preds
        df["confidence"] = conf
        st.dataframe(df.head())
        st.download_button("Download", df.to_csv(index=False).encode("utf-8"), "output.csv")
