import streamlit as st

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Diag 02 - Plotting Imports", page_icon="🧪", layout="centered")

st.title("Diagnostic Stage 02")
st.write("This stage verifies matplotlib and seaborn imports.")

fig, ax = plt.subplots(figsize=(4, 3))
sns.barplot(x=["A", "B"], y=[1, 2], ax=ax)
ax.set_xlabel("Category")
ax.set_ylabel("Value")
st.pyplot(fig)
plt.close(fig)
st.success("Reached stage 02: plotting imports")
