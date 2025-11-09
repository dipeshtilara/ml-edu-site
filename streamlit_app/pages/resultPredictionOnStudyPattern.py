# streamlit_app/pages/resultPredictionOnStudyPattern.py
import streamlit as st

st.set_page_config(page_title="Result Prediction (Study Pattern)", layout="wide")
st.title("Result Prediction based on Study Pattern — Demo")

# Lazy imports so the app shows a friendly message if dependencies are missing
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        accuracy_score,
        roc_auc_score,
        roc_curve,
    )
except Exception as e:
    st.error("A required package is missing. Please add the required packages to requirements.txt and redeploy.")
    st.write("Import error:", e)
    st.stop()

st.markdown(
    """
This demo shows a simple classification flow using Logistic Regression.
Choose a dataset, adjust split / CV, then click **Train & Evaluate**.
"""
)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Settings")
dataset_name = st.sidebar.selectbox(
    "Choose dataset",
    [
        "Study Pattern (Manual, small toy example)",
        "Iris (sklearn)",
        "Wine (sklearn)",
        "Breast Cancer (sklearn)",
        "Upload CSV"
    ],
)
test_size = st.sidebar.slider("Test set fraction", 0.1, 0.5, 0.25, step=0.05)
cv_folds = st.sidebar.slider("Cross-validation folds", 2, 10, 4)
random_state = 42

# -------------------------
# Load dataset based on selection
# -------------------------
def load_manual_study_pattern():
    data = {
        "Study Hours": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "Attendance":  [80, 85, 78, 90, 88, 92, 95, 98],
        "Interest":    [3, 4, 2, 5, 4, 5, 5, 5],
        "Test Scores": [40, 50, 45, 60, 65, 70, 75, 80],
    }
    df = pd.DataFrame(data)
    # convert to binary result (pass if score >=50)
    df["Result"] = df["Test Scores"].apply(lambda x: 1 if x >= 50 else 0)
    X = df[["Study Hours", "Attendance", "Interest"]]
    y = df["Result"]
    feature_names = list(X.columns)
    return df, X, y, feature_names

def load_sklearn_dataset(loader):
    data = loader(as_frame=True)
    df = data.frame.copy()
    # If frame already has the target as a column, fine; otherwise make 'target'
    if "target" not in df.columns:
        df["target"] = data.target
    X = df[data.feature_names]
    y = df["target"]
    feature_names = list(data.feature_names)
    return df, X, y, feature_names

df = None
X = None
y = None
feature_names = []

if dataset_name == "Study Pattern (Manual, small toy example)":
    df, X, y, feature_names = load_manual_study_pattern()
elif dataset_name == "Iris (sklearn)":
    df, X, y, feature_names = load_sklearn_dataset(load_iris)
elif dataset_name == "Wine (sklearn)":
    df, X, y, feature_names = load_sklearn_dataset(load_wine)
elif dataset_name == "Breast Cancer (sklearn)":
    df, X, y, feature_names = load_sklearn_dataset(load_breast_cancer)
elif dataset_name == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV file (first row should be header)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            # Let user choose features and target from uploaded file
            cols = df.columns.tolist()
            target_col = st.sidebar.selectbox("Select target column", cols)
            feature_cols = st.sidebar.multiselect("Select feature columns", [c for c in cols if c != target_col], default=[c for c in cols if c != target_col][:3])
            if len(feature_cols) == 0:
                st.warning("Pick at least one feature column from the uploaded CSV in the sidebar.")
                st.stop()
            X = df[feature_cols]
            y = df[target_col]
            feature_names = feature_cols
        except Exception as e:
            st.error("Could not read the uploaded CSV. Check file format and try again.")
            st.write("Error:", e)
            st.stop()
    else:
        st.info("Upload a CSV to use a custom dataset or choose one of the built-in datasets.")
        st.stop()

# Show dataset preview and distribution
st.subheader("Dataset preview")
st.dataframe(df.head())

st.write("Shape:", df.shape)
st.write("Feature columns:", feature_names)
st.write("Target distribution:")
try:
    dist = pd.Series(y).value_counts().rename_axis("label").reset_index(name="count")
    st.table(dist)
except Exception:
    st.write("Could not compute label distribution.")

# -------------------------
# Train & Evaluate (user triggers)
# -------------------------
if st.button("Train & Evaluate"):

    # Robust train-test split (stratify if possible)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        used_stratify = True
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        used_stratify = False

    st.write(f"Using stratify in split: **{used_stratify}**")
    st.write(f"Train shape: {X_train.shape} — Test shape: {X_test.shape}")

    # Pipeline: scaler + logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", random_state=random_state)),
    ])

    with st.spinner("Training the model..."):
        pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    # Confusion matrix
    st.subheader("Confusion matrix (test set)")
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    # For binary, labels will be [0,1] typically; for multiclass cm shape matches unique labels
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax_cm,
                xticklabels=np.unique(y), yticklabels=np.unique(y), cmap="Blues")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # Classification report
    st.subheader("Classification report (test set)")
    cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cr_df = pd.DataFrame(cr).transpose()
    st.dataframe(cr_df)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy (test set): **{acc:.3f}**")

    # ROC AUC for binary only
    unique_labels = np.unique(y_test)
    if y_proba is not None and len(unique_labels) == 2:
        try:
            auc = roc_auc_score(y_test, y_proba)
            st.write(f"ROC AUC (test set): **{auc:.3f}**")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, linewidth=2)
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            st.pyplot(fig_roc)
        except Exception as e:
            st.info("Could not compute ROC AUC: " + str(e))
    else:
        st.info("ROC curve shown only for binary classification with predicted probabilities available.")

    # Cross-validation (F1)
    st.subheader("Cross-validation (k-fold) — F1 scores")
    try:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_weighted")
        st.write(f"F1-weighted scores (k={cv_folds}): {np.round(cv_scores, 3).tolist()}")
        st.write(f"Mean F1-weighted: **{cv_scores.mean():.3f}**, Std: {cv_scores.std():.3f}")
    except Exception as e:
        st.info("Cross-validation failed (too few samples for chosen folds). Try fewer folds or a larger dataset.")
        st.write("CV error:", e)

    # Model coefficients for interpretability (if features are present and classifier exposes coef_)
    st.subheader("Model coefficients (feature importance)")
    try:
        coefs = pipeline.named_steps["clf"].coef_
        # For multiclass LogisticRegression, coef_ shape = (n_classes, n_features)
        if coefs.ndim == 1 or coefs.shape[0] == 1:
            # binary or single set
            flat = coefs.flatten()
            coef_df = pd.DataFrame({
                "feature": feature_names,
                "coefficient": flat,
                "abs_coeff": np.abs(flat)
            }).sort_values("abs_coeff", ascending=False).reset_index(drop=True)
            st.table(coef_df[["feature", "coefficient"]])
        else:
            # multiclass: show coefficient per class
            coef_multi = pd.DataFrame(coefs.T, index=feature_names)
            coef_multi.columns = [f"class_{c}" for c in range(coef_multi.shape[1])]
            st.dataframe(coef_multi)
    except Exception as e:
        st.info("Could not extract model coefficients: " + str(e))

    st.success("Training & evaluation finished.")
    st.markdown("""
    **Notes & next steps**
    - For reliable performance estimates, use a larger dataset and more cross-validation.
    - Logistic Regression benefits from feature scaling (done in pipeline).
    - For multiclass ROC, consider one-vs-rest or other strategies (not shown here).
    - If you use an uploaded CSV, ensure the target column is categorical/numeric and features are numeric or preprocessed.
    """)
else:
    st.info("Click **Train & Evaluate** to train the model on the selected dataset.")
