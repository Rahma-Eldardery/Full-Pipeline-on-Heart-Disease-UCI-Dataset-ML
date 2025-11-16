# 🩺 End-to-End Heart Disease Prediction Pipeline

This repository contains a machine learning project to predict heart disease using the **Heart Disease UCI dataset**. It implements an end-to-end pipeline, starting from data cleaning and exploratory data analysis (EDA), moving through feature engineering, and concluding with a baseline model evaluation.

---

## 🚀 Project Status

**✅ Completed Stages:**
* **Data Preprocessing & Cleaning**
* **Exploratory Data Analysis (EDA)**
* **Feature Engineering (PCA, RF Importance, RFE)**
* **Baseline Model Training (Logistic Regression)**

This README reflects all work completed to date. The project is modular, allowing for more models and features to be added.

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

### 4. Baseline Model Training
* **Data Split:** All three feature sets (PCA, RF-selected, RFE-selected) were split into 80% training and 20% testing sets.
* **Baseline Model:** A `LogisticRegression` model was trained and evaluated on all three feature sets.
* **Evaluation:** The `accuracy_score` was used to compare the performance of each feature set.

---

## 📈 Preliminary Results

This baseline evaluation provides a benchmark for all future models. The `LogisticRegression` model's performance on the test set for each feature set was:

| Feature Set | Features | Test Accuracy |
| :--- | :--- | :--- |
| **PCA** | 10 Components | 88.52% |
| **RFE** | 9 Selected Features | 88.52% |
| **Random Forest** | 9 Selected Features | 86.89% |

These results show that both PCA and RFE provided an excellent, compact feature set for a linear model.

---

## 🛠️ Technologies & Libraries

* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Preprocessing & Modeling:** Scikit-learn (Sklearn)
    * `StandardScaler`, `RobustScaler`, `OneHotEncoder`
    * `PCA`, `RFE`, `RandomForestClassifier`, `LogisticRegression`
    * `train_test_split`, `accuracy_score`
* **Notebook Modularity:** `import-ipynb`

---

## 💻 How to Run

1.  Clone this repository.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn import-ipynb
    ```
3.  Ensure the `processed_cleveland.csv` dataset is located in the correct path.
4.  Run the main notebook (e.g., `model_training.ipynb`) that imports the other modules (`data_preprocessing`, `pca_analysis`, `feature_selection`).
