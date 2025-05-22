
# 📊 Business Analytics & Data Mining: Comprehensive Reference Guide

This repository brings together a suite of lectures, concepts, code, and documentation aimed at mastering the principles and applications of **business analytics**, **data science**, and **machine learning**. It follows a structured learning path aligned with the **CRISP-DM** methodology and includes content on both supervised and unsupervised learning, predictive modeling, evaluation metrics, and deployment best practices.

---

## 🧠 Topics Covered

### 📘 Foundational Concepts
- **[Introduction to Business Analytics](notes/1_introduction_to_business_analytics/introduction_to_business_analytics.md)**  
  Learn the role of data-driven decision-making, business understanding, and how predictive analytics delivers value through transparency and automation.

- **[CRISP-DM Process](https://miro.medium.com/v2/resize:fit:1400/1*JYbymHifAk7aQ1pHm_IdMQ.png)**   
  Business Understanding → Data Understanding → Preparation → Modeling → Evaluation → Deployment

---

### 🧮 Supervised Learning

#### Classification
- **[Intro to Classification](notes/2_introduction_to_classification/intro_to_classification.md)**  
  Understand target variable segmentation, entropy, and information gain.

- **[Classification Methods](notes/3_classification_method/classification_methods.md)**  
  Dive deeper into k-NN, Decision Trees, and Logistic Regression with discussions on decision boundaries, model objectives, and overfitting.

- **[Class Probability and Ranking Models](notes/4_class_probability_and_ranking_models/class_probability_ranking_models.md)**  
  Learn how models estimate class membership probabilities and rank instances using ROC curves and AUC.

#### Regression
- **[Fundamentals of Numeric Predictions](notes/5_fundamentals_of_numeric_predictions/fundamentals_of_numeric_predictions.md)**  
  Focused on numeric prediction (./e.g., Linear Regression), model evaluation (./RMSE, MAE, MAPE), and variable selection methods.

---

### 🧭 Unsupervised Learning

- **[Clustering and Unsupervised Learning](notes/6_clustering_and_unsupervised_learning/clustering_unsupervised_learning.md)**  
  Covers k-Means clustering, similarity-based grouping, and optimal cluster evaluation (./Elbow Method, domain knowledge).

---

## 🧪 Projects

### 🔍 [Customer Churn Prediction](project/customer_churn_prediction.py)
- Uses `Churn_Modeling.csv` dataset
- Applies Logistic Regression, kNN, and Decision Trees
- Final model deployment and evaluation using ROC/AUC

### 🔢 [Regression Models Comparison](exercises/regression_models/regression_models_comparison.py)
- Compares linear and non-linear regressors (./Linear Regression, k-NN, Decision Tree)
- Visualizes fit and discusses overfitting vs underfitting

### 📊 [Logistic Regression Analysis](exercises/logistic_regression/logistic_regression_analysis.py)
- Standardizes features and compares multiple models
- Generates learning curves and fitting graphs

---

## 🧾 Metrics & Model Evaluation

- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- AUC (./Area Under ROC Curve)
- RMSE, MAE, MAPE for regression
- Kappa and Matthews correlation for classification

---

## 🧠 Key Takeaways

- Decision Trees are flexible but prone to overfitting in small datasets
- Logistic Regression provides interpretable probabilistic models
- k-NN is simple but computationally intensive for large datasets
- Linear models can capture nonlinearities with engineered features
- Clustering helps in understanding data segments and customer profiling

---

## 📌 Usage

Clone the repo or download the individual scripts and markdowns. Each `.py` script is standalone and can be run directly in a Python 3 environment with necessary dependencies:

```bash
python script_name.py
```

---

## 🎓 Recommended Next Steps

- Explore ensemble methods like Random Forests and XGBoost
- Try dimensionality reduction (./e.g., PCA) before modeling
- Use automated hyperparameter tuning (./GridSearchCV, Optuna)
- Build interactive dashboards with Streamlit for stakeholder presentations

---

## 👥 Contributors

- Sean Jung — Project Lead
- Jiayan Han — Data Science
- Ryan Chen — Strategy
- Kasandra Woo — Analytics & Deployment

---
