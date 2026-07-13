from pathlib import Path

import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Diag 04 - Model Train", page_icon="🧪", layout="centered")

st.title("Diagnostic Stage 04")
st.write("This stage verifies the scikit-learn training pipeline on local data.")

data_path = Path(__file__).resolve().parents[1] / "data" / "titanic.csv"
df = pd.read_csv(data_path)

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

st.write(f"Training rows: {len(data)}")
st.write(f"Training accuracy: {train_accuracy:.3f}")
st.success("Reached stage 04: model training completed")
