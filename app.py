"""
Salani ML Explorer — Interactive Machine Learning App

Built at SMIT Hackathon by Imran Ali
Upload any CSV dataset, select a model, tune hyperparameters, and get instant results.

Supports:
- Supervised Learning: Decision Tree, Random Forest, SVM
- Unsupervised Learning: KMeans, Agglomerative Clustering, DBSCAN

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import io


def preprocess_data(data):
    """
    Automatically preprocesses a given DataFrame.
    - Fills missing values (numeric with median, categorical with mode).
    - Encodes categorical features.
    - Scales numeric features.
    """
    preprocessed_data = data.copy()

    for col in preprocessed_data.columns:
        if pd.api.types.is_numeric_dtype(preprocessed_data[col]):
            # Fill NaNs with the median
            median = preprocessed_data[col].median()
            if pd.isna(median): median = 0
            preprocessed_data[col] = preprocessed_data[col].fillna(median)

            # Scale the numeric feature
            scaler = StandardScaler()
            preprocessed_data[col] = scaler.fit_transform(preprocessed_data[[col]])

        else: # Column is categorical
            # Fill NaNs with the mode
            modes = preprocessed_data[col].mode()
            if not modes.empty:
                mode_val = modes[0]
            else:
                mode_val = 'Missing'

            preprocessed_data[col] = preprocessed_data[col].fillna(mode_val)

            # Encode the categorical feature
            encoder = LabelEncoder()
            preprocessed_data[col] = encoder.fit_transform(preprocessed_data[col].astype(str))

    # Catch any NaNs created by StandardScaler (if a column had zero variance)
    preprocessed_data = preprocessed_data.fillna(0)

    return preprocessed_data


# --- Main App ---
st.set_page_config(page_title="Salani ML Explorer", page_icon="🧠", layout="wide")

# --- Sidebar ---
st.sidebar.title("ML Model Explorer")
learning_type = st.sidebar.selectbox("Select Learning Type", ["Supervised", "Unsupervised"])
uploaded_file = st.sidebar.file_uploader("Upload your CSV Dataset", type=["csv"])

# --- Main Panel ---
st.title("🚀 Salani Machine Learning Explorer")
st.info("Welcome! Upload any CSV, select a model from the sidebar, and get instant results.")

data = None
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        with st.expander("Show Dataset Preview"):
            st.dataframe(data.head())
    except Exception as e:
        st.error(f"Error reading the file: {e}")
else:
    st.info("Please upload a CSV file using the sidebar to get started.")

# --- Supervised Learning Module ---
if learning_type == "Supervised" and data is not None:

    st.sidebar.header("Supervised Learning Settings")
    all_columns = data.columns.tolist()
    target_column = st.sidebar.selectbox("Select Target Column (y)", all_columns)
    model_name = st.sidebar.selectbox("Select Model",
                                      ["Decision Tree Classifier", "Random Forest Classifier", "Support Vector Machine (SVM)"])
    st.sidebar.subheader("Model Hyperparameters")
    model = None

    if model_name == "Decision Tree Classifier":
        max_depth = st.sidebar.slider("max_depth", 2, 32, 5, key="dt_depth")
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif model_name == "Random Forest Classifier":
        n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100, key="rf_est")
        max_depth = st.sidebar.slider("max_depth", 2, 32, 5, key="rf_depth")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_name == "Support Vector Machine (SVM)":
        C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0, key="svm_c")
        model = SVC(C=C, random_state=42)

    if st.sidebar.button("Train Model"):
        try:
            st.header("📊 Preprocessing Results")
            data_for_processing = data.copy()
            y = data_for_processing[target_column]
            X_data = data_for_processing.drop(columns=[target_column])
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            X = preprocess_data(X_data)

            st.success("✅ Preprocessing complete.")
            with st.expander("Show Processed Data (X)"):
                st.dataframe(X.head())

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.header(f"📈 Results for {model_name}")
            model.fit(X_train, y_train)
            st.success("✅ Model trained successfully.")

            y_pred = model.predict(X_test)
            st.subheader("🎯 Performance Metrics")
            accuracy = accuracy_score(y_test, y_pred)
            st.metric(label="Accuracy", value=f"{accuracy*100:.2f}%")

            all_labels = le_target.transform(le_target.classes_)
            all_class_names = le_target.classes_.astype(str)

            st.text("Classification Report:")
            report_stream = io.StringIO()
            print(classification_report(y_test, y_pred, labels=all_labels, target_names=all_class_names, zero_division=0),
                  file=report_stream)
            st.code(report_stream.getvalue())

            st.subheader("🖼️ Visualization")
            st.text("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred, labels=all_labels)
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, xticklabels=all_class_names, yticklabels=all_class_names)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Unsupervised Learning Module ---
if learning_type == "Unsupervised" and data is not None:

    st.sidebar.header("Unsupervised Learning Settings")
    model_name = st.sidebar.selectbox("Select Model", ["KMeans", "Agglomerative Clustering", "DBSCAN"])
    st.sidebar.subheader("Model Hyperparameters")
    model = None

    if model_name == "KMeans":
        n_clusters = st.sidebar.slider("n_clusters (K)", 2, 10, 3, key="kmeans_k")
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    elif model_name == "Agglomerative Clustering":
        n_clusters = st.sidebar.slider("n_clusters", 2, 10, 3, key="agg_k")
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif model_name == "DBSCAN":
        eps = st.sidebar.slider("eps (max distance)", 0.1, 5.0, 0.5, key="db_eps")
        min_samples = st.sidebar.slider("min_samples", 2, 10, 5, key="db_min")
        model = DBSCAN(eps=eps, min_samples=min_samples)

    if st.sidebar.button("Run Model"):
        try:
            st.header("📊 Preprocessing Results")
            X_processed = preprocess_data(data.copy())
            st.success("✅ Preprocessing complete.")
            with st.expander("Show Processed Data"):
                st.dataframe(X_processed.head())

            st.header(f"📈 Results for {model_name}")
            clusters = model.fit_predict(X_processed)
            st.success("✅ Model run successfully.")

            st.subheader("🎯 Performance Metrics")
            if len(np.unique(clusters)) > 1:
                score = silhouette_score(X_processed, clusters)
                st.metric(label="Silhouette Score", value=f"{score:.2f}")
                st.info("Silhouette Score (+1 is best, -1 is worst) measures how good the clusters are.")
            else:
                st.warning("Only one cluster was found. Silhouette Score cannot be calculated.")

            st.subheader("🖼️ Visualization (2D Cluster Plot via PCA)")
            st.info("We use PCA to reduce all the columns down to 2 dimensions for plotting.")

            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_processed)
            plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            plot_df['Cluster'] = clusters

            fig, ax = plt.subplots(figsize=(10, 7))
            sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Cluster',
                            palette='Set1', s=100, alpha=0.7, ax=ax)
            plt.title(f"Cluster Plot for {model_name}")
            plt.xlabel("Principal Component 1 (PC1)")
            plt.ylabel("Principal Component 2 (PC2)")
            plt.legend(title="Cluster")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
