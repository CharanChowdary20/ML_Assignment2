import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Page Setup
st.set_page_config(page_title="BITS ML Assignment 2", layout="wide")

st.title("📦 Product Success Classification Dashboard")
st.markdown("### M.Tech (DSE) - Machine Learning Assignment 2")

# --- SIDEBAR: DATA UPLOAD  ---
st.sidebar.header("Step 1: Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv")

# --- SIDEBAR: MODEL SELECTION  ---
st.sidebar.header("Step 2: Select Model")
model_choice = st.sidebar.selectbox(
    "Choose a Classification Model",
    ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost")
)

# Model Mapping to .pkl files saved in /model folder 
model_map = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

def preprocess_test_data(df):
    """Encodes categorical data and defines features ( 30)."""
    data = df.copy()
    
    # Create target based on Success_Percentage
    if 'Target' not in data.columns and 'Success_Percentage' in data.columns:
        data['Target'] = (data['Success_Percentage'] > 50).astype(int)
    
    # Label Encoding for Category/Sub_category
    le = LabelEncoder()
    cat_cols = ['Category', 'Sub_category']
    for col in cat_cols:
        if col in data.columns:
            data[col] = le.fit_transform(data[col].astype(str))
            
    # Required features 
    features = ['Price', 'Rating', 'No_rating', 'Discount', 'M_Spend', 
                'Supply_Chain_E', 'Sales_y', 'Sales_m', 'Market_T', 
                'Seasonality_T', 'Category', 'Sub_category']
    
    available_features = [f for f in features if f in data.columns]
    return data[available_features], data['Target'] if 'Target' in data.columns else None

# Main Logic
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.success("Dataset Uploaded!")
    
    with st.expander("Preview CSV Data"):
        st.dataframe(df_test.head())

    try:
        # Load Model
        model_path = f"model/{model_map[model_choice]}"
        model = joblib.load(model_path)
        
        # Prepare and Scale Data
        X_test, y_true = preprocess_test_data(df_test)
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # ---  (c): METRICS DISPLAY ---
        st.subheader(f"📊 Evaluation Metrics: {model_choice}") # FIXED LINE
        if y_true is not None:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
            c2.metric("AUC Score", f"{roc_auc_score(y_true, y_prob):.3f}")
            c3.metric("Precision", f"{precision_score(y_true, y_pred):.3f}")
            c4.metric("Recall", f"{recall_score(y_true, y_pred):.3f}")
            c5.metric("F1 Score", f"{f1_score(y_true, y_pred):.3f}")
            c6.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.3f}")
            
            # ---  (d): CONFUSION MATRIX  ---
            st.divider()
            left, right = st.columns(2)
            
            with left:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)
                
            with right:
                st.subheader("Classification Report")
                st.code(classification_report(y_true, y_pred))
        else:
            st.warning("Please ensure CSV has 'Success_Percentage' for metric calculation.")
            
    except Exception as e:
        st.error(f"Error: {e}. Check if your model files are in the 'model/' folder.")
else:
    st.info("👋 Upload your test CSV in the sidebar to begin.")
