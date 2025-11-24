# ❤️ Heart Disease Prediction: An End-to-End Medical AI Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

This repository contains a machine learning project to predict heart disease using the **Heart Disease UCI dataset**. It implements an end-to-end pipeline, starting from data cleaning and exploratory data analysis (EDA), moving through feature engineering, and concluding with a baseline model evaluation.
> 🚧 **Current Status:** The project is currently in the **Optimization & Deployment Phase**. We have completed the core modeling and clustering stages.

---
## 📅 Project Roadmap & Progress

| Phase | Task | Status |
| :--- | :--- | :---: |
| **Phase 1** | Data Preprocessing & Cleaning | ✅ Completed |
| **Phase 2** | Exploratory Data Analysis (EDA) | ✅ Completed |
| **Phase 3** | Feature Engineering (PCA, RFE) | ✅ Completed |
| **Phase 4** | Supervised Learning (Model Selection) | ✅ Completed |
| **Phase 5** | Unsupervised Learning (Clustering) | ✅ Completed |
| **Phase 6** | **Hyperparameter Tuning (GridSearch)** | ⏳ **Next Step** |
| **Phase 7** | **Building Streamlit UI** | ⏳ **Planned** |
| **Phase 8** | **Deployment (Ngrok)** | ⏳ **Planned** |

---
## 🏆 Current Results (Baseline Models)

We compared 4 different classifiers. **Logistic Regression (with PCA)** is currently the champion model due to its high sensitivity to detecting disease.

| 🥇 Rank | Model | Feature Set | Accuracy | Recall (Sensitivity) | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1st** | **Logistic Regression** | **PCA (10 Comps)** | **90.16%** | **0.91** | **0.93** |
| **2nd** | SVM | RF Features | 90.16% | 0.88 | 0.94 |
| **3rd** | Random Forest | PCA | 88.52% | 0.85 | 0.92 |
| **4th** | Decision Tree | RFE | 78.69% | 0.76 | 0.79 |

> **Medical Insight:** The Champion Model achieved a **Recall of 0.91**, meaning it successfully detected **91% of actual heart patients**.

---
## 🕵️‍♂️ Unsupervised Learning Insights

We applied **K-Means Clustering (K=2)** to find hidden patterns without using diagnosis labels.
* **Cluster 1:** Contained **139 Patients** and only 1 Healthy individual.
* **Result:** The model successfully separated the majority of "High Risk" patients purely based on their medical vitals.
 --- 

## 📊 Completed Workflow

This project has successfully implemented the following workflow:

### 1. Data Preprocessing & Cleaning
* **Loaded Data:** The `processed_cleveland.csv` dataset was loaded, and descriptive column names were assigned.
* **Handled Missing Values:** Replaced `'?'` placeholders with `np.nan`.
* **Imputation:** Filled missing values for `ca` and `thal` using the mode.
* **Target Variable:** Binarized the target `num` (diagnosis) into **0 (No Disease)** and **1 (Has Disease)**.
* **Scaling:** Applied `StandardScaler` (to `age`, `thalach`) and `RobustScaler` (to `oldpeak`, `chol`, `trestbps`) to normalize feature distributions.
* **Encoding:** Applied `OneHotEncoder` to the categorical `thal` feature.

### 2. Exploratory Data Analysis (EDA)
* **Visualized Distributions:** Generated `histplot` and `boxplot` for numerical features (`age`, `chol`, etc.) to check for outliers and skew.
* **Analyzed Categories:** Used `countplot` for categorical features (`sex`, `cp`, etc.) to understand class balances.
* **Correlation Analysis:** Generated a `heatmap` to visualize the correlation matrix between features.

### 3. Feature Engineering & Selection
Three distinct feature sets were engineered to compare model performance:

* **A) Principal Component Analysis (PCA):**
    * Applied PCA to the full dataset.
    * Plotted the cumulative explained variance and selected the **top 10 principal components** (which retained ~95% of the variance).

* **B) Random Forest Feature Importance:**
    * Trained a `RandomForestClassifier` on the full dataset.
    * Extracted and ranked features by Gini importance, selecting the **top 9 features**.

* **C) Recursive Feature Elimination (RFE):**
    * Used `RFE` with a `LogisticRegression` estimator.
    * Programmatically selected the **top 9 features**.

### 4. Supervised Learning (Model Comparison)
We trained **Logistic Regression, SVM, Random Forest, and Decision Trees** across all feature sets.
* *Result:* Linear models (LR & SVM) outperformed tree-based models on this dataset.

### 5. Unsupervised Learning (Pattern Discovery) 🕵️‍♂️
We hid the diagnosis labels and applied Clustering to see if the AI could find natural patterns.
* **K-Means (K=2):** Validated using the **Elbow Method**.
* **Hierarchical Clustering:** Validated using the **Dendrogram**.
* **Insight:** The model successfully grouped **128 healthy individuals** into a single pure cluster, proving the data has strong natural separability.

---

## 🛠️ Technologies & Tools Used

| Category | Library/Tool | Purpose |
| :--- | :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white) | The core programming language. |
| **Data Manipulation** | **Pandas & NumPy** | For data cleaning, handling missing values, and numerical operations. |
| **Visualization** | **Matplotlib & Seaborn** | For generating heatmaps, boxplots, and distributions during EDA. |
| **Machine Learning** | **Scikit-Learn** | The main engine for PCA, RFE, GridSearch, Clustering, and Models (LR, SVM, RF). |
| **Scientific Computing** | **SciPy** | Used for hierarchical clustering and dendrograms. |
| **Environment** | **Jupyter Notebook** | For interactive coding and experimentation. |



---

### 👨‍💻 Author

**Developed by [Rahma Eldardery]**

If you found this project useful or interesting, please consider giving it a ⭐!
Your support is greatly appreciated.

---
