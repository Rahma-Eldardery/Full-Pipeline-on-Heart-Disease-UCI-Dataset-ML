# 🩺 End-to-End Heart Disease Prediction Pipeline

This repository contains a complete machine learning project to predict heart disease using the **Heart Disease UCI dataset**. It implements an end-to-end pipeline, starting from data cleaning and exploratory data analysis (EDA), moving through feature engineering and model comparison, and concluding with model deployment in a live web application.

---

## Project Status

**🚧 In Progress 🚧**

This project is in active development. The workflow below outlines all planned steps.

* **[Done]** 2.1: Data Preprocessing & Cleaning
* **[Done]** 2.2: Dimensionality Reduction (PCA)
* **[Done]** 2.3: Feature Selection (RF & RFE)
* **[Done]** 2.4: Supervised Learning (Baseline `LogisticRegression`)
* **[Up Next]** 2.4: Supervised Learning (Comparing KNN, SVM, Decision Trees)
* **[Todo]** 2.5: Unsupervised Learning (Clustering)
* **[Todo]** 2.6: Hyperparameter Tuning
* **[Todo]** 2.7: Model Export
* **[Todo]** 2.8: Streamlit Web UI
* **[Todo]** 2.9: Deployment

---

## 🚀 Project Workflow

This project is broken down into the following distinct stages:

### 2.1 Data Preprocessing & Cleaning
* **Load Data:** Load the `processed_cleveland.csv` dataset and assign descriptive column names.
* **Handle Missing Values:** Replace `'?'` placeholders with `np.nan`.
* **Imputation:** Fill missing values for categorical features (`ca`, `thal`) using the mode (most frequent value).
* **EDA:** Perform exploratory data analysis with `histplot` and `boxplot` for numerical features and `countplot` for categorical features.
* **Target Variable:** Binarize the target `num` (diagnosis) into `0` (No Disease) and `1` (Has Disease).

### 2.2 Dimensionality Reduction - PCA
* **Apply PCA:** Use Principal Component Analysis to reduce the dimensionality of the feature set.
* **Analyze Variance:** Plot the cumulative explained variance to determine the optimal number of components to retain (e.g., 10 components capturing 95% of the variance).

### 2.3 Feature Selection
Compare different automated feature selection techniques to create optimal subsets of features.
* **Random Forest Importance:** Train a `RandomForestClassifier` and select the top 9 features based on Gini importance.
* **Recursive Feature Elimination (RFE):** Use `RFE` with a `LogisticRegression` estimator to programmatically select the 9 most impactful features.

### 2.4 Supervised Learning - Classification Models
Train and evaluate multiple classification models on the different feature sets (Full, PCA, RF-selected, RFE-selected) to find the best performer.
* **Data Split:** Split all datasets into 80% training and 20% testing sets.
* **Models:**
    * Logistic Regression (Baseline)
    * K-Nearest Neighbors (KNN)
    * Decision Trees
    * Support Vector Machine (SVM)
* **Evaluation:** Assess models using `accuracy_score`, `precision`, `recall`, `f1-score`, and a confusion matrix.

### 2.5 Unsupervised Learning - Clustering
* **K-Means Clustering:** Apply K-Means to identify natural groupings in the data. Use the "Elbow Method" to determine the optimal number of clusters (K).
* **Hierarchical Clustering:** Use agglomerative clustering and plot a dendrogram to analyze the data's hierarchical structure.

### 2.6 Hyperparameter Tuning
* **Optimize Best Model:** Take the best-performing model from step 2.4.
* **Tune:** Use `GridSearchCV` or `RandomizedSearchCV` to find the optimal set of hyperparameters, maximizing the evaluation metric (e.g., F1-score or Accuracy).

### 2.7 Model Export & Deployment
* **Create Pipeline:** Build a formal `sklearn.pipeline.Pipeline` that includes all preprocessing steps (scaling, encoding) and the final tuned model.
* **Save Model:** Export the entire pipeline object as a `.pkl` or `.joblib` file for later use.

### 2.8 Streamlit Web UI Development [Bonus]
* **Build Interface:** Create an interactive web application using Streamlit.
* **User Inputs:** Add UI components (sliders, text inputs, dropdowns) for a user to input their own health metrics.
* **Live Prediction:** Load the saved model pipeline to provide a real-time "Heart Disease" (Yes/No) prediction based on user inputs.

### 2.9 Deployment using Ngrok [Bonus]
* **Run Locally:** Serve the Streamlit application locally.
* **Expose Publicly:** Use `Ngrok` to create a secure public URL, making the web application accessible to anyone on the internet for demonstration.

---

## 📊 Preliminary Results

Initial comparison of a `LogisticRegression` model across the engineered feature sets:

| Feature Set | Features | Test Accuracy |
| :--- | :--- | :--- |
| **PCA** | 10 Components | 88.52% |
| **RFE** | 9 Selected Features | 88.52% |
| **Random Forest** | 9 Selected Features | 86.89% |

*These results will be updated as more models are trained and tuned.*

---

## 🛠️ Key Technologies & Libraries

* **Data:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **ML & Preprocessing:** Scikit-learn (Sklearn)
* **Web App:** Streamlit
* **Deployment:** Ngrok
* **Notebooks:** Jupyter, `import-ipynb`
