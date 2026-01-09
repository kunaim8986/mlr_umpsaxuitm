import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

# ===============================
# Page configuration
# ===============================
st.set_page_config(page_title="Linear Regression", layout="wide")

# Sidebar - Logo and Developers
st.sidebar.image(
    "https://tse3.mm.bing.net/th/id/OIP.eDhfRYRnhXheu42_SBMI4AHaDB?rs=1&pid=ImgDetMain&o=7&rm=3",
    use_container_width=True
)
st.sidebar.image(
    "https://assets.nst.com.my/images/articles/Uitmlogo_1683739413.jpg",
    use_container_width=True
)

st.sidebar.header("Developers:")
for dev in ["Ku Muhammad Naim Ku Khalif", "Ashraff Ruslan"]:
    st.sidebar.write(f"- {dev}")

st.title("Linear Regression Framework with Prediction")

# ===============================
# File upload
# ===============================
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

# ===============================
# Helper functions
# ===============================
def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)

def standardize(s):
    std = s.std(ddof=0)
    return (s - s.mean()) / std if std != 0 else s * 0

def regression_equation(params, predictors):
    eq = f"ŷ = {params.iloc[0]:.4f}"
    for i, var in enumerate(predictors, start=1):
        sign = "+" if params.iloc[i] >= 0 else "-"
        eq += f" {sign} {abs(params.iloc[i]):.4f}({var})"
    return eq

# ===============================
# Main workflow
# ===============================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    features = st.sidebar.multiselect("Select independent variables (X)", df.columns)
    target = st.sidebar.selectbox("Select dependent variable (y)", df.columns)

    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
    point_size = st.sidebar.slider("Scatter point size", 10, 200, 60)

    if features and target and target not in features:
        numeric_features = [c for c in features if is_numeric(df[c])]

        if not is_numeric(df[target]):
            st.error("Target must be numeric.")
            st.stop()

        if not numeric_features:
            st.error("Select at least one numeric predictor.")
            st.stop()

        work_df = df[numeric_features + [target]].dropna()
        X = work_df[numeric_features]
        y = work_df[target]

        # ===============================
        # 1) Correlation Analysis
        # ===============================
        st.subheader("Correlation Analysis (Pearson)")

        corr_matrix = work_df[numeric_features + [target]].corr()
        corr_target = corr_matrix[target].drop(target)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Correlation with target**")
            st.dataframe(
                corr_target.to_frame("Corr(X, y)").style.format("{:.4f}"),
                use_container_width=True
            )

        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

        # ===============================
        # 2) Train/Test + sklearn model
        # ===============================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Model Performance (Test Set)")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
        st.write(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

        # ===============================
        # 3) Scatter Plot (Best Fit Only)
        # ===============================
        st.subheader("Actual vs Predicted (Best Fitted Line)")

        y_test_arr = y_test.to_numpy()
        y_pred_arr = np.asarray(y_pred)

        r_ap = np.corrcoef(y_test_arr, y_pred_arr)[0, 1]

        fig_sc, ax_sc = plt.subplots(figsize=(7, 5))
        ax_sc.scatter(y_test_arr, y_pred_arr, s=point_size)
        ax_sc.set_xlabel("Actual (y)")
        ax_sc.set_ylabel("Predicted (ŷ)")
        ax_sc.set_title(f"Actual vs Predicted (Pearson r = {r_ap:.4f})")

        # Best fitted line
        m, c = np.polyfit(y_test_arr, y_pred_arr, 1)
        x_line = np.linspace(min(y_test_arr), max(y_test_arr), 200)
        y_line = m * x_line + c
        ax_sc.plot(x_line, y_line, linewidth=2,
                   label=f"Best fit: ŷ = {m:.3f}y + {c:.3f}")

        ax_sc.legend()
        st.pyplot(fig_sc)

        # ===============================
        # 4) SPSS-STYLE MODEL SUMMARY ONLY
        # ===============================
        st.subheader("Linear Regression Model Summary)")

        X_sm = sm.add_constant(X)
        ols = sm.OLS(y, X_sm).fit()

        model_summary = pd.DataFrame({
            "R": [np.sqrt(ols.rsquared)],
            "R Square": [ols.rsquared],
            "Adjusted R Square": [ols.rsquared_adj],
            "Std. Error of the Estimate": [np.sqrt(ols.mse_resid)],
            "N": [int(ols.nobs)]
        })

        st.dataframe(
            model_summary.style.format({
                "R": "{:.4f}",
                "R Square": "{:.4f}",
                "Adjusted R Square": "{:.4f}",
                "Std. Error of the Estimate": "{:.4f}",
                "N": "{:.0f}"
            }),
            use_container_width=True
        )

        # ===============================
        # 5) Regression Equation
        # ===============================
        st.subheader("Regression Equation")
        st.write(regression_equation(ols.params, numeric_features))

        # ===============================
        # 6) Prediction
        # ===============================
        st.subheader("Make a Prediction")
        inputs = []
        for f in numeric_features:
            inputs.append(
                st.number_input(f"Input {f}", value=float(X[f].mean()))
            )

        if st.button("Predict"):
            pred = model.predict(np.array(inputs).reshape(1, -1))[0]
            st.success(f"Predicted {target}: {pred:.4f}")

    else:
        st.info("Select at least one predictor and one target (target ≠ predictors).")

else:
    st.info("Upload a CSV file to start.")
