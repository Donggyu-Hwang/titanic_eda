from pathlib import Path

import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Diag 06 - Predict Click Only", page_icon="🧪", layout="centered")

st.title("Diagnostic Stage 06")
st.write("This stage isolates the button-click prediction path with no extra UI effects.")

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
        ("classifier", LogisticRegression(solver="liblinear", max_iter=200)),
    ]
)

model.fit(X, y)

st.write("Training complete. Click the button to test prediction only.")

pclass = st.selectbox("Passenger Class", options=[1, 2, 3], index=2)
sex = st.radio("Sex", options=["male", "female"], horizontal=True)
age = st.slider("Age", min_value=0, max_value=80, value=30)
embarked = st.selectbox("Embarkation Port", options=["S", "C", "Q"])
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=6, value=0)
fare = st.slider("Fare", min_value=0.0, max_value=512.0, value=32.0, step=0.5)

input_df = pd.DataFrame(
    {
        "pclass": [pclass],
        "sex": [sex],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "embarked": [embarked],
    }
)

if st.button("Predict"):
    probability = float(model.predict_proba(input_df)[0][1])
    st.write(f"Probability: {probability:.6f}")
    st.code(input_df.to_string(index=False))
