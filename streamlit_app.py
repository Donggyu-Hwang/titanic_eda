import os

# 일부 클라우드 환경(예: 무료 티어)에서 NumPy/scikit-learn이 사용하는 BLAS 라이브러리의
# 멀티스레딩으로 인해 드물게 발생할 수 있는 충돌을 방지하기 위한 안전 설정입니다.
# 반드시 numpy/scikit-learn을 import하기 전에 설정해야 합니다.
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
import matplotlib.font_manager as fm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------
# 페이지 기본 설정
# -------------------------------------------------------------------
st.set_page_config(page_title="타이타닉 생존 확률 예측기", page_icon="🚢", layout="wide")


# -------------------------------------------------------------------
# 한글 폰트 설정 (설치되어 있는 경우에만 적용, 없으면 기본 폰트 사용)
# Streamlit Community Cloud에 배포할 때는 packages.txt에 fonts-nanum을
# 추가해두면 아래에서 NanumGothic이 자동으로 적용됩니다.
# -------------------------------------------------------------------
@st.cache_resource
def set_korean_font():
    candidates = ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic", "NanumBarunGothic"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in installed:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False
    return True


set_korean_font()


# -------------------------------------------------------------------
# 데이터 불러오기 (캐시: 앱이 다시 실행돼도 매번 새로 불러오지 않음)
# -------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = sns.load_dataset("titanic")
    # 예측에 사용할 핵심 컬럼만 남기고, 결측치가 있는 행은 모델 학습 시 별도로 처리합니다.
    return df


# -------------------------------------------------------------------
# 모델 학습 (캐시: 앱이 다시 실행돼도 매번 재학습하지 않음)
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
# 데이터 & 모델 준비
# -------------------------------------------------------------------
df = load_data()
model, train_accuracy, n_train = train_model(df)


# -------------------------------------------------------------------
# 헤더
# -------------------------------------------------------------------
st.title("🚢 타이타닉 생존 확률 예측기")
st.write(
    "1912년 타이타닉호에 탑승했던 실제 승객 데이터를 살펴보고(EDA), "
    "여러분이 그 배에 타고 있었다면 생존 확률이 얼마나 되었을지 머신러닝 모델로 예측해보는 앱입니다."
)

tab_eda, tab_predict = st.tabs(["📊 데이터 탐색 (EDA)", "🔮 나의 생존 확률 예측"])

# =====================================================================
# TAB 1. EDA
# =====================================================================
with tab_eda:
    st.header("타이타닉 승객 데이터 탐색적 분석")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 승객 수", f"{len(df):,}명")
    col2.metric("생존자 수", f"{int(df['survived'].sum()):,}명")
    col3.metric("전체 생존율", f"{df['survived'].mean() * 100:.1f}%")
    col4.metric("평균 나이", f"{df['age'].mean():.1f}세")

    st.subheader("데이터 미리보기")
    st.dataframe(df.head(10))

    with st.expander("컬럼 설명 보기"):
        st.markdown(
            """
| 컬럼명 | 설명 |
|---|---|
| survived | 생존 여부 (0 = 사망, 1 = 생존) |
| pclass | 좌석 등급 (1 = 1등석, 2 = 2등석, 3 = 3등석) |
| sex | 성별 |
| age | 나이 |
| sibsp | 함께 탑승한 형제자매/배우자 수 |
| parch | 함께 탑승한 부모/자녀 수 |
| fare | 운임 |
| embarked | 탑승 항구 (S=Southampton, C=Cherbourg, Q=Queenstown) |
"""
        )

    st.subheader("결측치 확인")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        st.bar_chart(missing)
        st.caption("특히 'deck'(선실 구역)과 'age'(나이) 컬럼에 결측치가 많습니다.")
    else:
        st.write("결측치가 없습니다.")

    st.subheader("변수별 생존율 살펴보기")

    c1, c2 = st.columns(2)
    with c1:
        st.write("**좌석 등급(Pclass)별 생존율**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=df, x="pclass", y="survived", ax=ax, color="#4C72B0")
        ax.set_xlabel("좌석 등급")
        ax.set_ylabel("생존율")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("1등석 승객의 생존율이 가장 높고, 3등석으로 갈수록 낮아집니다.")

    with c2:
        st.write("**성별 생존율**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=df, x="sex", y="survived", ax=ax, color="#C44E52", order=["male", "female"])
        ax.set_xlabel("성별")
        ax.set_ylabel("생존율")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["남성", "여성"])
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("'여성과 어린이 먼저(Women and children first)' 원칙에 따라 여성의 생존율이 훨씬 높습니다.")

    c3, c4 = st.columns(2)
    with c3:
        st.write("**나이 분포 (생존 여부별)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(data=df, x="age", hue="survived", kde=True, ax=ax, multiple="stack", palette=["#C44E52", "#4C72B0"])
        ax.set_xlabel("나이")
        ax.set_ylabel("인원 수")
        legend = ax.get_legend()
        if legend:
            legend.set_title("생존 여부")
            for t, label in zip(legend.texts, ["사망", "생존"]):
                t.set_text(label)
        st.pyplot(fig)
        plt.close(fig)

    with c4:
        st.write("**탑승 항구별 생존율**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=df, x="embarked", y="survived", ax=ax, color="#55A868")
        ax.set_xlabel("탑승 항구")
        ax.set_ylabel("생존율")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("수치형 변수 간 상관관계")
    numeric_df = df[["survived", "pclass", "age", "sibsp", "parch", "fare"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("좌석 등급(pclass)이 낮을수록(숫자가 클수록) 생존율과 음의 상관관계를 보입니다.")


# =====================================================================
# TAB 2. 생존 확률 예측
# =====================================================================
with tab_predict:
    st.header("나의 정보를 입력하고 생존 확률 예측하기")
    st.write("아래 항목에 나의 정보를 입력하면, 학습된 머신러닝 모델(로지스틱 회귀)이 생존 확률을 계산해줍니다.")
    st.caption(
        f"※ 이 모델은 실제 타이타닉 탑승자 {n_train}명의 데이터로 학습되었으며, "
        f"학습 데이터 기준 정확도는 약 {train_accuracy * 100:.1f}% 입니다. "
        "역사적 데이터를 바탕으로 한 재미있는 시뮬레이션이니 참고용으로만 봐주세요."
    )

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox(
            "좌석 등급",
            options=[1, 2, 3],
            index=2,
            format_func=lambda x: {1: "1등석", 2: "2등석", 3: "3등석"}[x],
        )
        sex = st.radio(
            "성별",
            options=["male", "female"],
            format_func=lambda x: "남성" if x == "male" else "여성",
            horizontal=True,
        )
        age = st.slider("나이", min_value=0, max_value=80, value=30)
        embarked = st.selectbox(
            "탑승 항구",
            options=["S", "C", "Q"],
            format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x],
        )

    with col2:
        sibsp = st.number_input("함께 탑승한 형제자매/배우자 수", min_value=0, max_value=8, value=0)
        parch = st.number_input("함께 탑승한 부모/자녀 수", min_value=0, max_value=6, value=0)
        fare = st.slider("운임 (파운드)", min_value=0.0, max_value=512.0, value=32.0, step=0.5)

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

    predict_clicked = st.button("생존 확률 예측하기", type="primary")

    if predict_clicked:
        proba = model.predict_proba(input_df)[0][1]
        percent = proba * 100

        st.subheader("예측 결과")
        result_col1, result_col2 = st.columns([1, 2])
        with result_col1:
            st.metric("예측 생존 확률", f"{percent:.1f}%")
        with result_col2:
            st.progress(min(int(percent), 100))

        if percent >= 66:
            st.success(f"생존 확률이 {percent:.1f}%로, 상당히 높은 편입니다! 🎉")
            st.balloons()
        elif percent >= 33:
            st.warning(f"생존 확률이 {percent:.1f}%로, 반반에 가깝습니다. 😐")
        else:
            st.error(f"생존 확률이 {percent:.1f}%로, 낮은 편입니다. 😢")

        with st.expander("입력한 정보 확인하기"):
            st.dataframe(input_df)

        st.caption(
            "예측 결과는 좌석 등급, 성별, 나이 등 몇 가지 변수만을 이용한 단순 모델의 결과이며, "
            "실제 역사적 사실이나 개인의 운명을 의미하지 않습니다."
        )
    else:
        st.info("왼쪽에서 정보를 입력한 뒤 '생존 확률 예측하기' 버튼을 눌러주세요.")


st.markdown("---")
st.caption("AI for Future Workforce · 타이타닉 생존 확률 예측 Streamlit 앱 · 교육용으로 제작되었습니다.")
