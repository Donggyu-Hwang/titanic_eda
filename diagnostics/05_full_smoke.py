from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Diag 05 - Full Smoke", page_icon="🧪", layout="centered")

st.title("Diagnostic Stage 05")
st.write("This stage runs the full import, data, plot, and model stack in one place.")

data_path = Path(__file__).resolve().parents[1] / "data" / "titanic.csv"
df = pd.read_csv(data_path)

fig, ax = plt.subplots(figsize=(4, 3))
sns.barplot(data=df, x="sex", y="survived", ax=ax, order=["male", "female"])
ax.set_xlabel("Sex")
ax.set_ylabel("Survival Rate")
st.pyplot(fig)
plt.close(fig)

features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
numeric_features = ["age", "sibsp", "parch", "fare"]
categorical_features = ["pclass", "sex", "embarked"]

data = df[features + ["survived"]].dropna().copy()
X = data[features]
y = data["survived"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(X, y)
train_accuracy = accuracy_score(y, model.predict(X))

st.write(f"Loaded rows: {len(df)}")
st.write(f"Training accuracy: {train_accuracy:.3f}")
st.success("Reached stage 05: full stack smoke test completed")
