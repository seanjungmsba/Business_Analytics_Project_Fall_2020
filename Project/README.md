
# 🏦 Customer Churn Prediction Project

This project applies the **CRISP-DM** methodology to build predictive models that help a fictional international bank reduce **customer churn**. The project uses the `Churn_Modeling.csv` dataset and implements multiple classification models to find the best approach for identifying at-risk customers.

---

## 📈 Business Goal

To identify customers likely to churn and understand which attributes most influence their decision. This will enable targeted interventions to improve **customer satisfaction** and **loyalty**.

---

## 🔄 CRISP-DM Process

This project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) approach:

### 1. Business Understanding
- Objective: Reduce churn rate while increasing customer loyalty.
- Questions:
  - What features impact churn the most?
  - What can the bank do to retain customers?

### 2. Data Understanding
- Dataset: [`churn_modeling.csv`](./churn_modeling.csv)
- Data from customers in **France, Spain, and Germany**
- Variables include: `CreditScore`, `Geography`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, etc.

### 3. Data Preparation
- Transformation of skewed variables using log-scale
- Creation of interaction terms:
  - `salary_balance = EstimatedSalary * Balance`
  - `tenure_product = Tenure * NumOfProducts`
- One-hot encoding of categorical variables like `Geography`
- Feature scaling for kNN

### 4. Modeling
- Models used:
  - **Logistic Regression**
  - **k-Nearest Neighbors (kNN)**
  - **Decision Tree**

- Metrics used:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
  - ROC Curve / AUC

| Model              | AUC    | Accuracy (CV) | F1 Score (CV) |
|-------------------|--------|---------------|---------------|
| Logistic Regression | 0.74 ± 0.02 | 0.62        | 0.63          |
| kNN                | 0.76 ± 0.01 | 0.68        | 0.67          |
| Decision Tree      | 0.83 ± 0.02 | 0.75        | 0.74          |

### 5. Evaluation
- Best model: **Decision Tree**
  - `criterion='gini', max_depth=8, min_samples_leaf=2, min_samples_split=25`
  - Accuracy: **0.85**, Precision: **0.71**, Recall: **0.47**

### 6. Deployment
- Final decision tree model identifies **key churn indicators**.
- Ethical concerns and model risks considered before deployment.
- Deployment plan includes integration into customer relationship workflows.

---

## 🧾 Files

- `customer_churn_prediction.py`: Complete Python implementation
- `Churn_Modeling.csv`: Dataset used for modeling
- `CRISP Process.pdf`: Overview of the CRISP-DM framework
- `Presentation.pdf`: Project presentation slides

---

## 📌 Next Steps

- Use ensemble models like Random Forest or Gradient Boosting
- Automate hyperparameter tuning with GridSearchCV
- Monitor and retrain models as new data comes in
