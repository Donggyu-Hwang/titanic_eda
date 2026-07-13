import os
from pathlib import Path

# Safety settings to reduce rare BLAS-related crashes in some cloud environments.
# These must be set before importing numpy/scikit-learn.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------
st.set_page_config(page_title="Titanic Survival Probability Predictor", page_icon="🚢", layout="wide")


# -------------------------------------------------------------------
# Chart styling
# -------------------------------------------------------------------
plt.rcParams["axes.unicode_minus"] = False


# -------------------------------------------------------------------
# Data loading (cached so reruns do not reload the file every time)
# -------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "titanic.csv"
    df = pd.read_csv(data_path)
    return df


# -------------------------------------------------------------------
# Model training (cached so reruns do not retrain every time)
# -------------------------------------------------------------------
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
NUMERIC_FEATURES = ["age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["pclass", "sex", "embarked"]


@st.cache_resource
def train_model(df: pd.DataFrame):
    data = df[FEATURES + ["survived"]].dropna().copy()
    X = data[FEATURES]
    y = data["survived"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
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
    return model, train_accuracy, len(data)


# -------------------------------------------------------------------
# Data and model prep
# -------------------------------------------------------------------
df = load_data()
model, train_accuracy, n_train = train_model(df)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.title("🚢 Titanic Survival Probability Predictor")
st.write(
    "Explore real passenger data from the 1912 Titanic voyage and use a machine learning model "
    "to estimate what your survival probability might have been."
)

tab_eda, tab_predict = st.tabs(["📊 Data Exploration (EDA)", "🔮 Survival Probability Prediction"])

# =====================================================================
# TAB 1. EDA
# =====================================================================
with tab_eda:
    st.header("Titanic Passenger Data Exploration")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Passengers", f"{len(df):,}")
    col2.metric("Survivors", f"{int(df['survived'].sum()):,}")
    col3.metric("Overall Survival Rate", f"{df['survived'].mean() * 100:.1f}%")
    col4.metric("Average Age", f"{df['age'].mean():.1f}")

    st.subheader("Data Preview")
    st.dataframe(df.head(10))

    with st.expander("View Column Descriptions"):
        st.markdown(
            """
| Column | Description |
|---|---|
| survived | Survival status (0 = did not survive, 1 = survived) |
| pclass | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| sex | Sex |
| age | Age |
| sibsp | Number of siblings/spouses aboard |
| parch | Number of parents/children aboard |
| fare | Fare |
| embarked | Port of embarkation (S = Southampton, C = Cherbourg, Q = Queenstown) |
"""
        )

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        st.bar_chart(missing)
        st.caption("The dataset has many missing values, especially in 'deck' and 'age'.")
    else:
        st.write("No missing values found.")

    st.subheader("Survival Rate by Feature")

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Survival Rate by Passenger Class**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=df, x="pclass", y="survived", ax=ax, color="#4C72B0")
        ax.set_xlabel("Passenger Class")
        ax.set_ylabel("Survival Rate")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("1st class passengers have the highest survival rate, and it drops toward 3rd class.")

    with c2:
        st.write("**Survival Rate by Sex**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=df, x="sex", y="survived", ax=ax, color="#C44E52", order=["male", "female"])
        ax.set_xlabel("Sex")
        ax.set_ylabel("Survival Rate")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Male", "Female"])
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Female passengers had a much higher survival rate under the 'women and children first' norm.")

    c3, c4 = st.columns(2)
    with c3:
        st.write("**Age Distribution by Survival Status**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(data=df, x="age", hue="survived", kde=True, ax=ax, multiple="stack", palette=["#C44E52", "#4C72B0"])
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        legend = ax.get_legend()
        if legend:
            legend.set_title("Survival Status")
            for t, label in zip(legend.texts, ["Did Not Survive", "Survived"]):
                t.set_text(label)
        st.pyplot(fig)
        plt.close(fig)

    with c4:
        st.write("**Survival Rate by Embarkation Port**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=df, x="embarked", y="survived", ax=ax, color="#55A868")
        ax.set_xlabel("Embarkation Port")
        ax.set_ylabel("Survival Rate")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Correlation Between Numeric Features")
    numeric_df = df[["survived", "pclass", "age", "sibsp", "parch", "fare"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Passenger class shows a negative correlation with survival rate as the class number increases.")


# =====================================================================
# TAB 2. 생존 확률 예측
# =====================================================================
with tab_predict:
    st.header("Enter Your Details and Predict Survival Probability")
    st.write("Enter your details below and the trained logistic regression model will estimate survival probability.")
    st.caption(
        f"This model was trained on data from {n_train} Titanic passengers, and the training accuracy is about {train_accuracy * 100:.1f}%. "
        "It is a historical simulation for reference only."
    )

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            index=2,
            format_func=lambda x: {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}[x],
        )
        sex = st.radio(
            "Sex",
            options=["male", "female"],
            format_func=lambda x: "Male" if x == "male" else "Female",
            horizontal=True,
        )
        age = st.slider("Age", min_value=0, max_value=80, value=30)
        embarked = st.selectbox(
            "Embarkation Port",
            options=["S", "C", "Q"],
            format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x],
        )

    with col2:
        sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
        parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=6, value=0)
        fare = st.slider("Fare (Pounds)", min_value=0.0, max_value=512.0, value=32.0, step=0.5)

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

    predict_clicked = st.button("Predict Survival Probability", type="primary")

    if predict_clicked:
        proba = model.predict_proba(input_df)[0][1]
        percent = proba * 100

        st.subheader("Prediction Result")
        result_col1, result_col2 = st.columns([1, 2])
        with result_col1:
            st.metric("Predicted Survival Probability", f"{percent:.1f}%")
        with result_col2:
            st.progress(min(int(percent), 100))

        if percent >= 66:
            st.success(f"The survival probability is {percent:.1f}%, which is quite high! 🎉")
            st.balloons()
        elif percent >= 33:
            st.warning(f"The survival probability is {percent:.1f}%, so it is around a coin flip. 😐")
        else:
            st.error(f"The survival probability is {percent:.1f}%, which is relatively low. 😢")

        with st.expander("Review Entered Information"):
            st.dataframe(input_df)

        st.caption(
            "The result comes from a simple model using only a few variables such as class, sex, and age, "
            "and does not represent historical fact or personal destiny."
        )
    else:
        st.info("Enter your information on the left, then click 'Predict Survival Probability'.")


st.markdown("---")
st.caption("AI for Future Workforce · Titanic Survival Probability Streamlit app · Built for educational purposes.")
