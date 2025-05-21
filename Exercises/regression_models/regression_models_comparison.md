
# 📉 Regression Models Comparison

This project explores and compares several regression algorithms for modeling relationships in datasets. It emphasizes understanding the behavior of different models under both linear and non-linear conditions.

---

## 📦 Libraries Used

- **Python 3**
- **Pandas / NumPy** for data manipulation
- **Matplotlib / Seaborn** for plotting
- **Scikit-learn** for regression modeling and evaluation

---

## 🔍 Topics Covered

### 1. Exploratory Data Analysis (EDA)
- Dataset loading, inspection, and visualization to understand distributions and patterns.

### 2. Linear Regression
- Fits a straight-line relationship to the data.
- Coefficients are estimated using the least squares method.

### 3. k-Nearest Neighbors (k-NN) Regression
- Predicts output based on average of `k` nearest neighbors.
- Non-parametric and useful for capturing local patterns in data.

### 4. Decision Tree Regression
- Splits data into regions and fits a constant in each region.
- Great for modeling non-linear patterns without requiring transformation.

### 5. Modeling Nonlinear Relationships
- Compare how each algorithm handles curves and discontinuities in data.
- Emphasizes visualization to interpret fit quality and model complexity.

---

## 📈 Evaluation

- **Visual Comparison**: Overlay predicted vs. true values.
- **Quantitative Metrics** (implied or extendable):
  - Mean Squared Error (MSE)
  - R² Score

---

## ✅ Running the Script

```bash
python regression_models_comparison.py
```

Make sure required libraries are installed in your Python environment.

---

## 📌 Outcome

This project is ideal for understanding:

- When to use linear vs non-linear models
- How tree-based and instance-based learners behave in regression settings
- Visual interpretation of fit quality and bias/variance characteristics

---

## 🧠 Extensions

You can further explore:
- Ensemble methods like Random Forests or Gradient Boosting
- Polynomial regression
- Model selection via cross-validation
