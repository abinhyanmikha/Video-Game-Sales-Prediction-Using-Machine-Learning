"""Streamlit app: Video Game Global Sales Prediction (no sklearn Pipeline)."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Video Game Sales Prediction", page_icon="🎮", layout="wide")
st.title("🎮 Video Game Sales Prediction")
st.caption("Explore historical game sales and train regression models — preprocessing is performed manually, without Pipeline.")

NUMERIC_CANDIDATES = [
    "Year_of_Release", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales",
    "Global_Sales", "Critic_Score", "Critic_Count", "User_Score", "User_Count"
]
CATEGORICAL_CANDIDATES = ["Name", "Platform", "Genre", "Publisher", "Developer", "Rating"]

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    from io import BytesIO
    df = pd.read_csv(BytesIO(file_bytes))
    df.columns = df.columns.str.strip()
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_CANDIDATES:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    if "Global_Sales" not in df.columns:
        raise ValueError("The uploaded CSV must contain a 'Global_Sales' column.")
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=["Global_Sales"]).copy()
    # Scores that are impossible are marked missing rather than deleting the whole game row.
    if "Critic_Score" in df.columns:
        df.loc[~df["Critic_Score"].between(0, 100), "Critic_Score"] = np.nan
    if "User_Score" in df.columns:
        df.loc[~df["User_Score"].between(0, 10), "User_Score"] = np.nan
    for col in ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    df = df.dropna(subset=["Global_Sales"]).copy()
    return df

with st.sidebar:
    st.header("1. Dataset")
    uploaded_file = st.file_uploader("Upload video game sales CSV", type=["csv"])
    st.caption("Expected target column: Global_Sales. Common dataset: Video_Games_Sales_as_at_22_Dec_2016.csv")
    st.divider()
    st.header("2. Model settings")
    test_size = st.slider("Test set proportion", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    random_state = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    train_rf = st.checkbox("Train Random Forest (may take longer)", value=True)

if uploaded_file is None:
    default_path = Path("Video_Games_Sales_as_at_22_Dec_2016.csv")
    if default_path.exists():
        raw_bytes = default_path.read_bytes()
        source_name = default_path.name
        st.info(f"Using local dataset: {source_name}. Upload another CSV in the sidebar to change it.")
    else:
        st.info("Upload your video game sales CSV using the sidebar to get started.")
        st.markdown("### Expected columns\n`Global_Sales`, `Platform`, `Genre`, `Publisher`, `Developer`, `Rating`, `Year_of_Release`, `Critic_Score`, `Critic_Count`, `User_Score`, `User_Count`.")
        st.stop()
else:
    raw_bytes = uploaded_file.getvalue()
    source_name = uploaded_file.name

try:
    df = load_and_clean(raw_bytes, source_name)
except Exception as exc:
    st.error(f"Could not read or clean this file: {exc}")
    st.stop()

if df.empty:
    st.error("No usable rows remain after cleaning. Check that Global_Sales contains numeric values.")
    st.stop()

st.success(f"Loaded {source_name} — {len(df):,} rows and {len(df.columns):,} columns after basic cleaning.")

# Overview
st.header("Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{len(df.columns):,}")
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
with st.expander("Preview data"):
    st.dataframe(df.head(100), use_container_width=True)
with st.expander("Data types and missing values"):
    quality = pd.DataFrame({"Data type": df.dtypes.astype(str), "Missing count": df.isna().sum(), "Missing %": (df.isna().mean()*100).round(2)})
    st.dataframe(quality, use_container_width=True)

st.download_button("Download cleaned dataset", data=df.to_csv(index=False).encode("utf-8"), file_name="cleaned_video_game_sales.csv", mime="text/csv")

# EDA
st.header("Exploratory data analysis")
eda1, eda2, eda3 = st.tabs(["Sales distribution", "Platforms & genres", "Relationships"])
with eda1:
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df["Global_Sales"].dropna(), bins=50, ax=ax)
    ax.set_title("Global sales distribution")
    ax.set_xlabel("Global sales (millions)")
    st.pyplot(fig)
    plt.close(fig)
    if "Name" in df.columns:
        cols = [c for c in ["Name", "Platform", "Year_of_Release", "Genre", "Global_Sales"] if c in df.columns]
        st.subheader("Top 10 selling games")
        st.dataframe(df.nlargest(10, "Global_Sales")[cols], use_container_width=True)
with eda2:
    col_a, col_b = st.columns(2)
    with col_a:
        if "Platform" in df.columns:
            counts = df["Platform"].fillna("Unknown").value_counts().head(10).sort_values()
            fig, ax = plt.subplots(figsize=(7, 4))
            counts.plot(kind="barh", ax=ax)
            ax.set_title("Top platforms by game count")
            st.pyplot(fig); plt.close(fig)
    with col_b:
        if "Genre" in df.columns:
            genre_sales = df.groupby("Genre", dropna=False)["Global_Sales"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(7, 4))
            genre_sales.plot(kind="barh", ax=ax)
            ax.set_title("Average global sales by genre")
            ax.set_xlabel("Mean global sales (millions)")
            st.pyplot(fig); plt.close(fig)
with eda3:
    corr_cols = [c for c in ["Global_Sales", "Year_of_Release", "Critic_Score", "Critic_Count", "User_Score", "User_Count"] if c in df.columns]
    if len(corr_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[corr_cols].corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Numeric feature correlation")
        st.pyplot(fig); plt.close(fig)
    else:
        st.write("Not enough numeric columns for a correlation plot.")

# Model section
st.header("Train and evaluate regression models")
st.warning("Regional sales columns are excluded from predictors because they contribute directly to Global_Sales and can cause target leakage when predicting sales before launch.")
cat_features = [c for c in ["Platform", "Genre", "Publisher", "Developer", "Rating"] if c in df.columns]
num_features = [c for c in ["Year_of_Release", "Critic_Score", "Critic_Count", "User_Score", "User_Count"] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
feature_cols = cat_features + num_features
if not feature_cols:
    st.error("No supported predictor columns were found. Check the CSV column names.")
    st.stop()

with st.expander("Choose predictors", expanded=True):
    st.write("Categorical features")
    selected_cat = st.multiselect("Categorical predictors", options=cat_features, default=cat_features)
    st.write("Numeric features")
    selected_num = st.multiselect("Numeric predictors", options=num_features, default=num_features)
features = selected_cat + selected_num
if not features:
    st.error("Select at least one predictor to train a model.")
    st.stop()

model_button = st.button("Train models", type="primary")
if model_button:
    model_df = df[features + ["Global_Sales"]].copy()
    model_df = model_df.dropna(subset=["Global_Sales"])
    X = model_df[features].copy()
    y = model_df["Global_Sales"].astype(float).copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(random_state))

    # Manual preprocessing only; fit imputation statistics on training data.
    for col in selected_num:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        median_value = X_train[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        X_train[col] = X_train[col].fillna(median_value)
        X_test[col] = X_test[col].fillna(median_value)
    for col in selected_cat:
        X_train[col] = X_train[col].astype("object").where(X_train[col].notna(), "Unknown").astype(str)
        X_test[col] = X_test[col].astype("object").where(X_test[col].notna(), "Unknown").astype(str)
    X_train_enc = pd.get_dummies(X_train, columns=selected_cat, dtype=int)
    X_test_enc = pd.get_dummies(X_test, columns=selected_cat, dtype=int)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    models = {
        "Baseline": DummyRegressor(strategy="mean"),
        "Decision Tree": DecisionTreeRegressor(max_depth=15, min_samples_leaf=5, random_state=int(random_state)),
    }
    if train_rf:
        models["Random Forest"] = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=int(random_state), n_jobs=-1)

    rows, fitted, pred_by_model = [], {}, {}
    progress = st.progress(0, text="Training models...")
    for i, (name, model) in enumerate(models.items(), start=1):
        model.fit(X_train_enc, y_train)
        pred = model.predict(X_test_enc)
        mse = mean_squared_error(y_test, pred)
        rows.append({"Model": name, "MAE": mean_absolute_error(y_test, pred), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R²": r2_score(y_test, pred)})
        fitted[name] = model
        pred_by_model[name] = pred
        progress.progress(i / len(models), text=f"Trained {name}")
    progress.empty()
    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = results.iloc[0]["Model"]
    st.session_state["vg_results"] = results
    st.session_state["vg_fitted"] = fitted
    st.session_state["vg_predictions"] = pred_by_model
    st.session_state["vg_y_test"] = y_test
    st.session_state["vg_features"] = list(X_train_enc.columns)
    st.session_state["vg_best_name"] = best_name
    st.session_state["vg_rf_importance"] = None
    if "Random Forest" in fitted:
        st.session_state["vg_rf_importance"] = pd.DataFrame({"Feature": X_train_enc.columns, "Importance": fitted["Random Forest"].feature_importances_}).sort_values("Importance", ascending=False)

if "vg_results" in st.session_state:
    results = st.session_state["vg_results"]
    st.subheader("Model comparison")
    st.dataframe(results.round(4), use_container_width=True)
    st.caption(f"Best model by RMSE: {st.session_state['vg_best_name']}. Lower MAE/RMSE is better; higher R² is generally better.")
    st.download_button("Download model metrics", data=results.to_csv(index=False).encode("utf-8"), file_name="model_metrics.csv", mime="text/csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(results["Model"], results["RMSE"])
    ax.set_ylabel("RMSE (millions of sales)")
    ax.set_title("Model comparison by RMSE (lower is better)")
    st.pyplot(fig); plt.close(fig)

    y_test = st.session_state["vg_y_test"]
    pred = st.session_state["vg_predictions"][st.session_state["vg_best_name"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, pred, alpha=0.4)
    max_value = max(float(np.nanmax(y_test)), float(np.nanmax(pred)))
    ax.plot([0, max_value], [0, max_value], "k--")
    ax.set_xlabel("Actual Global_Sales")
    ax.set_ylabel("Predicted Global_Sales")
    ax.set_title(f"Actual vs predicted — {st.session_state['vg_best_name']}")
    st.pyplot(fig); plt.close(fig)

    importance = st.session_state.get("vg_rf_importance")
    if importance is not None:
        st.subheader("Random Forest feature importance")
        st.dataframe(importance.head(20), use_container_width=True)
        top = importance.head(15).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["Feature"], top["Importance"])
        ax.set_xlabel("Importance")
        ax.set_title("Top 15 features")
        st.pyplot(fig); plt.close(fig)
        st.download_button("Download feature importance", data=importance.to_csv(index=False).encode("utf-8"), file_name="feature_importance.csv", mime="text/csv")

st.divider()
st.caption("Note: this app reports association and predictive performance, not causal effects. Results depend on dataset quality and the selected predictors.")
