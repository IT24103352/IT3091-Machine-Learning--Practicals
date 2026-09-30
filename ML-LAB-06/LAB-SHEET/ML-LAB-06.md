**BSc (Hons) in Information Technology**
**IT3091 - Machine Learning**
**Y3.S1 – 2026**

**Faculty of Computing**

**LAB SHEET 06**

---

# Regression , SVM & SVR with Scikit-Learn

> **Learning Outcomes**
>
> By the end of this lab session, students will be able to:
>
> - Implement Simple Linear Regression from scratch using NumPy.
> - Calculate predictions, residuals, Mean Squared Error (MSE), and regression evaluation metrics.
> - Train a Linear Regression model from scratch using Gradient Descent.
> - Fit and tune a Support Vector Machine (SVM) classifier using Scikit-Learn.
> - Explain the decision boundary, margin, support vectors, and Hinge Loss used in SVM, and relate them to a fitted model's own attributes.
> - Fit and tune a Support Vector Regression (SVR) model using Scikit-Learn.
> - Explain the ε-tube and ε-insensitive loss used in SVR, and how ε and C shape the fitted model.
> - Compare linear and nonlinear (kernel-based) SVM/SVR behaviour, and relate the library-based models to the from-scratch regression model.

> **Software / Resources Required**
>
> - Python 3.x
> - Jupyter Notebook / Google Colab
> - NumPy, Pandas, Matplotlib and Scikit-learn
>
> **Important:** Part A (Linear Regression) is still implemented from scratch using basic Python/NumPy operations, exactly as before. Parts B and C (SVM and SVR) now use Scikit-Learn directly: you will fit real models, inspect their attributes, and explore how their hyperparameters change behaviour, rather than deriving the training loop by hand.

---

## Part A: Linear Regression from Scratch

### A.1 – Dataset & Problem Setup

A university wants to predict a student's **Exam Score** based on the number of **Hours Studied**.

| Student | Hours Studied (X) | Exam Score (Y) |
|---|---|---|
| A | 1 | 48 |
| B | 2 | 52 |
| C | 3 | 58 |
| D | 4 | 64 |
| E | 5 | 70 |
| F | 6 | 75 |

```python
import numpy as np

X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([48, 52, 58, 64, 70, 75])
```

#### Task 1 – Understand the Problem

1. Identify the input feature, target variable, and type of supervised learning problem.
2. Create a scatter plot of Hours Studied vs. Exam Score.
3. Describe the relationship visible in the scatter plot.
4. Why is regression more appropriate than classification for this problem?

### A.2 – Build the Prediction Function

> #### Simple Linear Regression
>
> **ŷ = β₀ + β₁x**
>
> β₀ = intercept    β₁ = regression coefficient    ŷ = predicted value

#### Task 2 – Initial Predictions

1. Initialize β₀ = 0 and β₁ = 0.
2. Complete the prediction function without using Scikit-learn.
3. Generate the initial predictions for all observations.
4. Explain the roles of β₀ and β₁.

```python
def predict(X, beta0, beta1):
    # TODO: calculate y_hat
    pass

beta0 = 0
beta1 = 0

y_hat = predict(X, beta0, beta1)
```

### A.3 – Residuals & Mean Squared Error

> **Residual:** eᵢ = yᵢ − ŷᵢ
>
> **Mean Squared Error:** MSE = (1/n) Σ(yᵢ − ŷᵢ)²

#### Task 3 – Calculate Error from Scratch

1. Calculate the residual for every observation.
2. Implement MSE from scratch.
3. Display the actual value, predicted value and residual for each observation.
4. What do positive, negative and near-zero residuals indicate?
5. Why are errors squared when calculating MSE?

```python
def mse(y, y_hat):
    # TODO: implement Mean Squared Error
    pass
```

### A.4 – Train Linear Regression using Gradient Descent

> ∂J/∂β₀ = (2/n) Σ(ŷᵢ − yᵢ)
>
> ∂J/∂β₁ = (2/n) Σ(ŷᵢ − yᵢ)xᵢ
>
> **Update:** βⱼ := βⱼ − α(∂J/∂βⱼ)

#### Task 4 – Implement Gradient Descent

1. Set a suitable learning rate and number of iterations.
2. Calculate predictions and gradients at each iteration.
3. Update β₀ and β₁.
4. Store the MSE at each iteration.
5. Display the final β₀, β₁ and MSE.
6. Plot Iteration vs. MSE and explain the convergence behaviour.
7. Plot the final regression line together with the observations.

```python
learning_rate = 0.01
iterations = 1000
beta0 = 0
beta1 = 0
loss_history = []

for i in range(iterations):
    y_hat = predict(X, beta0, beta1)

    # TODO: calculate dbeta0
    # TODO: calculate dbeta1

    # TODO: update beta0 and beta1

    # TODO: calculate and store MSE
```

### A.5 – Evaluate the Regression Model

> **MAE** = (1/n) Σ|yᵢ − ŷᵢ|
>
> **RMSE** = √MSE
>
> **R²** = 1 − [Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²]

#### Task 5 – Metrics from Scratch

1. Implement MAE, MSE, RMSE and R² without using Scikit-learn metrics.
2. Interpret each value.
3. Which metrics are expressed in the same units as the target?
4. Which metric penalizes larger errors more strongly?
5. Predict the exam score for a student who studies **4.5 hours**.

### A.6 – Verify using Scikit-learn

1. Train `LinearRegression` using the same dataset.
2. Display the intercept and coefficient.
3. Predict the score for 4.5 hours.
4. Compare the results with your from-scratch model.

---

## Part B: Support Vector Machine (SVM) with Scikit-Learn

### B.1 – Binary Classification Dataset

Use the following two-dimensional dataset to investigate how a linear SVM separates two classes.

| Observation | X₁ | X₂ | Class |
|---|---|---|---|
| A | 1 | 2 | −1 |
| B | 2 | 2 | −1 |
| C | 2 | 3 | −1 |
| D | 4 | 4 | +1 |
| E | 5 | 4 | +1 |
| F | 5 | 5 | +1 |

```python
import numpy as np

X = np.array([[1, 2], [2, 2], [2, 3], [4, 4], [5, 4], [5, 5]])
y = np.array([-1, -1, -1, 1, 1, 1])
```

#### Task 6 – Visualize the Classification Problem

1. Plot the two classes using a scatter plot.
2. Can the classes be separated by a straight line?
3. Sketch or plot more than one possible separating line.

### B.2 – SVM Decision Function

> **f(x) = wᵀx + b**
>
> For two features: f(x) = w₁x₁ + w₂x₂ + b
>
> Predict +1 when f(x) ≥ 0; otherwise predict −1.

#### Task 7 – Fit an SVC and Inspect Its Decision Function

1. Fit `SVC(kernel='linear', C=1.0)` on the dataset.
2. Retrieve `svc.coef_` and `svc.intercept_` and use them to compute f(x) = wᵀx + b by hand for every observation.
3. Compare your hand-computed scores with `svc.decision_function(X)`, they should match.
4. Compare `svc.predict(X)` with the actual classes.

```python
from sklearn.svm import SVC

svc = SVC(kernel='linear', C=1.0)
svc.fit(X, y)

w, b = svc.coef_[0], svc.intercept_[0]
scores = X @ w + b              # should match svc.decision_function(X)
preds  = svc.predict(X)
```

### B.3 – Margin & Support Vectors

> **Decision boundary:** wᵀx + b = 0
>
> **Margin boundaries:** wᵀx + b = +1 and wᵀx + b = −1
>
> **Margin condition:** yᵢ(wᵀxᵢ + b) ≥ 1

#### Task 8 – Investigate the Margin

1. Using the w and b from Task 7, plot the decision boundary (wᵀx + b = 0) and both margin boundaries (wᵀx + b = ±1).
2. Print `svc.support_vectors_` and mark these points on your plot.
3. Explain why exactly these observations are called **support vectors**.
4. Why does SVM prefer a large margin instead of simply finding any separating line?

### B.4 – Hinge Loss

> **Lᵢ = max(0, 1 − yᵢf(xᵢ))**

#### Task 9 – Hinge Loss from the Fitted Model

1. Use `svc.decision_function(X)` as f(x) for every observation.
2. Calculate the Hinge Loss Lᵢ = max(0, 1 − yᵢf(xᵢ)) for each observation, one line of NumPy, no loop needed.
3. Compare points outside the margin, inside the margin and incorrectly classified points.
4. Which observations produce zero Hinge Loss? How does that compare with `svc.support_vectors_` from Task 8?

```python
scores = svc.decision_function(X)
hinge = np.maximum(0, 1 - y * scores)
```

### B.5 – Interactive Sandbox: The Effect of C on the Margin

> #### What C Controls
>
> C is the regularization parameter of `SVC`. A **small C** lets the model accept a wider margin even if that means a few points fall inside it; a **large C** forces the model to fit the training points as tightly as possible, shrinking the margin. Drag the slider below to watch the decision boundary, both margin boundaries and the set of support vectors change, computed from a real `SVC(kernel='linear', C=...)` fit on the exact dataset from B.1.

#### Task 10 – The Effect of C on the Margin

**Interactive sandbox** (plot of points A–F with the decision boundary, both margin boundaries and ringed support vectors; slider positions C = 0.05, C = 0.1, C = 0.3, C = 1; default C = 1). Legend: Class −1 (navy), Class +1 (orange); ringed points are support vectors.

| C | Margin width (2/‖w‖) | Support vectors |
|---|---|---|
| 0.05 | 4.99 | A, B, C, D, E, F |
| 0.1 | 3.60 | B, C, D, E |
| 0.3 | 2.98 | C, D |
| 1 (default) | 2.24 | C, D |

1. Slide C from its smallest to its largest value. What happens to the margin width, and why?
2. How does the number of support vectors change as C increases? Connect this to what a support vector is.
3. Between C = 0.3 and C = 1 the margin barely changes even though C increases. Why does that happen for this dataset?
4. If you were worried this dataset had a mislabelled point, would you choose a smaller or a larger C? Explain.

---

## Part C: Support Vector Regression (SVR) with Scikit-Learn

### C.1 – Reuse the Regression Dataset

Use the **Hours Studied → Exam Score** dataset from Part A. Unlike SVM classification, SVR predicts a continuous value.

> #### Linear SVR Function
>
> **f(x) = wx + b**
>
> Instead of trying to minimize every small error, SVR defines an acceptable error region around the regression function.

#### Task 11 – Fit an SVR and Inspect Its Predictions

1. Fit `SVR(kernel='linear', C=1.0, epsilon=1.0)` on the dataset.
2. Retrieve `svr.coef_` and `svr.intercept_` and use them to compute f(x) = wx + b by hand for every observation.
3. Compare your hand-computed values with `svr.predict(X)`, they should match.
4. Plot the observations together with the fitted line.

```python
from sklearn.svm import SVR

svr = SVR(kernel='linear', C=1.0, epsilon=1.0)
svr.fit(X, y)

w, b = svr.coef_[0][0], svr.intercept_[0]
preds = w * X.ravel() + b       # should match svr.predict(X)
```

### C.2 – Construct the ε-Tube

> **Upper boundary:** f(x) + ε    **Lower boundary:** f(x) − ε
>
> An observation whose residual falls inside the tube (|y − f(x)| < ε) contributes zero loss. scikit-learn calls the observations that end up *on* or *outside* the tube boundary the **support vectors**.

#### Task 12 – Visualize the ε-Tube

1. Using the w and b from Task 11, and ε = 1.0, compute the upper boundary f(x) + ε and lower boundary f(x) − ε.
2. Plot the regression line, both ε boundaries and the actual observations.
3. Print `svr.support_` and mark these observations on your plot.
4. Which observations fall strictly inside the tube, and which sit on or outside its boundary?

```python
upper = svr.predict(X) + svr.epsilon
lower = svr.predict(X) - svr.epsilon
inside = np.abs(y - svr.predict(X)) < svr.epsilon
```

### C.3 – ε-Insensitive Loss

> **L_ε(y, f(x)) = max(0, |y − f(x)| − ε)**

#### Task 13 – ε-Insensitive Loss from the Fitted Model

1. Use `svr.predict(X)` as f(x) for every observation.
2. Calculate the ε-insensitive loss L_ε = max(0, |y − f(x)| − ε) for each observation, one line of NumPy, no loop needed.
3. Display Actual, Predicted, Absolute Error and Loss for each observation.
4. Which observations produce zero loss? How does that compare with `svr.support_` from Task 12?

```python
preds = svr.predict(X)
loss = np.maximum(0, np.abs(y - preds) - svr.epsilon)
```

### C.4 – Interactive Sandbox: The Effect of ε on the Tube

> #### What ε Controls
>
> ε is the width of the no-penalty tube around the fitted line in `SVR`. A **small ε** forces the line to pass close to almost every point, so nearly all of them end up as support vectors; a **large ε** lets many points sit comfortably inside a wide tube with zero loss, leaving fewer points to constrain the line, so it is free to become flatter. Drag the slider below to watch the fitted line, both tube boundaries and the set of support vectors change, computed from a real `SVR(kernel='linear', C=1.0, epsilon=...)` fit on the exact dataset from C.1.

#### Task 14 – The Effect of ε on the Tube

**Interactive sandbox** (plot of Hours vs. Score with observations A–F, the fitted line, both tube boundaries and ringed support vectors; slider positions ε = 0.1, ε = 0.5, ε = 1, ε = 3; default ε = 1). Legend: Observation (navy); ringed points are support vectors, on or outside the tube.

| ε | Slope (w) | Support vectors |
|---|---|---|
| 0.1 | 5.40 | A, B, C, D, E, F |
| 0.5 | 5.33 | B, C, E, F |
| 1 (default) | 5.00 | B, C, E, F |
| 3 | 4.00 | B, F |

1. Slide ε from its smallest to its largest value. What happens to the width of the tube, and why?
2. How does the number of support vectors change as ε increases? Connect this to which points fall inside versus outside the tube.
3. The fitted line keeps getting flatter between ε = 1 and ε = 3, even though the number of support vectors stays the same. Why does the line keep changing?
4. If this dataset contained a genuine outlier, would a larger or a smaller ε make the fitted line less sensitive to it? Explain.

---

## Part D: Model Comparison & Understanding

#### Task 15 – Compare the Models

| Property | Linear Regression | SVM | SVR |
|---|---|---|---|
| Target type | | | |
| Main purpose | | | |
| Output | | | |
| Main loss/error concept | | | |
| Margin / ε-tube | | | |
| Role of support vectors | | | |

1. Why can Linear Regression not be directly used for the classification problem in Part B?
2. What is the main conceptual difference between SVM and SVR?
3. Why is feature scaling important when using SVM and SVR?
4. Give one suitable real-world application for Linear Regression, SVM and SVR.

---

> **End of Lab Sheet 06**
>
> Ensure that your notebook contains the completed from-scratch implementations, requested plots, observations, metric values, predictions and the Scikit-learn verification results.
