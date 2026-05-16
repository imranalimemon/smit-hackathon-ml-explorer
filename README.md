# 🧠 Salani ML Explorer — SMIT Hackathon Project

![Python](https://img.shields.io/badge/Python-3.8+-3670A0?logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **Built at SMIT Hackathon** 🏆  
> An interactive ML web app that lets you upload **any CSV dataset**, pick a machine learning model, tune hyperparameters, and get instant results — **no coding required**.

---

## 📋 Table of Contents

- [Features](#-features)
- [Supported Models](#-supported-models)
- [How It Works](#-how-it-works)
- [How to Run](#-how-to-run)
- [Screenshots](#-app-flow)
- [Tech Stack](#-tech-stack)
- [Key Learnings](#-key-learnings)

---

## ✨ Features

- 📂 **Upload any CSV** — the app auto-detects column types
- 🔧 **Auto-preprocessing** — handles missing values, encodes categoricals, scales numerics
- 🎛️ **Tunable hyperparameters** — interactive sliders for every model
- 📊 **Instant metrics** — accuracy, F1-score, confusion matrix, silhouette score
- 📈 **Visualizations** — confusion matrix heatmaps, PCA cluster plots
- 🔀 **Two learning modes** — supervised classification & unsupervised clustering

---

## 🤖 Supported Models

### Supervised Learning (Classification)

| Model | Hyperparameters |
|-------|----------------|
| **Decision Tree** | `max_depth` (2–32) |
| **Random Forest** | `n_estimators` (50–500), `max_depth` (2–32) |
| **SVM** | `C` regularization (0.01–10.0) |

### Unsupervised Learning (Clustering)

| Model | Hyperparameters |
|-------|----------------|
| **KMeans** | `n_clusters` (2–10) |
| **Agglomerative Clustering** | `n_clusters` (2–10) |
| **DBSCAN** | `eps` (0.1–5.0), `min_samples` (2–10) |

---

## ⚙️ How It Works

```
Upload CSV → Auto-Preprocess → Select Model → Tune Params → Train/Run → View Results
```

### Preprocessing Pipeline
1. **Numeric columns**: Fill NaN with median → StandardScaler
2. **Categorical columns**: Fill NaN with mode → LabelEncoder
3. **Target variable**: LabelEncoder (for supervised mode)

### Evaluation
- **Supervised**: Accuracy, Classification Report, Confusion Matrix
- **Unsupervised**: Silhouette Score, PCA 2D Cluster Plot

---

## 🚀 How to Run

### Local Setup

```bash
# Clone the repository
git clone https://github.com/imranalimemon/smit-hackathon-ml-explorer.git
cd smit-hackathon-ml-explorer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Google Colab
The original notebook (`SMIT_Project.ipynb`) runs on Google Colab with a Cloudflare tunnel to expose the Streamlit app publicly.

---

## 🔄 App Flow

1. **Upload** a CSV file using the sidebar
2. **Choose** Supervised or Unsupervised learning
3. **Select** a model and tune hyperparameters with sliders
4. **Click** "Train Model" or "Run Model"
5. **View** preprocessing results, metrics, and visualizations

### Example: Supervised Mode
```
Dataset: Iris.csv (150 rows, 5 columns)
Model: Random Forest (100 trees, max_depth=5)
Target: Species

Results:
  Accuracy: 96.67%
  Confusion Matrix: [heatmap visualization]
```

### Example: Unsupervised Mode
```
Dataset: Mall_Customers.csv
Model: KMeans (K=4)

Results:
  Silhouette Score: 0.55
  Cluster Plot: [PCA 2D scatter plot]
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Streamlit** | Web interface & interactivity |
| **scikit-learn** | ML models, preprocessing, metrics |
| **Pandas** | Data loading & manipulation |
| **Matplotlib + Seaborn** | Visualizations |
| **PCA** | Dimensionality reduction for cluster plots |

---

## 📁 Project Structure

```
smit-hackathon-ml-explorer/
├── app.py                 # Main Streamlit application
├── SMIT_Project.ipynb     # Original Colab notebook (hackathon submission)
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## 💡 Key Learnings

1. **Auto-preprocessing is powerful** — handling any CSV without manual feature engineering makes ML accessible to non-coders
2. **Streamlit makes deployment easy** — building a full ML web app in a single Python file
3. **Hyperparameter tuning matters** — interactive sliders let users see how parameters affect results in real-time
4. **PCA for visualization** — reducing high-dimensional data to 2D makes clustering results interpretable
5. **Hackathon time management** — scoping to a working MVP with both supervised and unsupervised modes

---

## 📄 License

MIT License

---

*Built by [Imran Ali](https://github.com/imranalimemon) at SMIT Hackathon — MUET Jamshoro, CS 2026*
