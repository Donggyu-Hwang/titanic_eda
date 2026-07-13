from pathlib import Path

import streamlit as st
import pandas as pd


st.set_page_config(page_title="Diag 03 - Local Data", page_icon="🧪", layout="centered")

st.title("Diagnostic Stage 03")
st.write("This stage verifies local file access to data/titanic.csv.")

data_path = Path(__file__).resolve().parents[1] / "data" / "titanic.csv"
df = pd.read_csv(data_path)

st.write(f"Loaded rows: {len(df)}")
st.write(f"Loaded columns: {len(df.columns)}")
st.dataframe(df.head())
st.success("Reached stage 03: local CSV loaded")
