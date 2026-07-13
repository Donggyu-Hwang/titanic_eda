import streamlit as st

import numpy as np
import pandas as pd


st.set_page_config(page_title="Diag 01 - Core Imports", page_icon="🧪", layout="centered")

st.title("Diagnostic Stage 01")
st.write("This stage verifies pandas and numpy imports.")

sample = pd.DataFrame({"value": np.arange(3)})
st.dataframe(sample)
st.success("Reached stage 01: pandas and numpy imports")
