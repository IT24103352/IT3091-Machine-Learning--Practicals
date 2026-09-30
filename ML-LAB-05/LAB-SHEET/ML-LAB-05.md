# BSc (Hons) in Information Technology
## IT3091 : Machine Learning
### Year 3 · Semester 1 · 2026
**Faculty of Computing — Lab Sheet 05**

# Logistic Regression & k-Nearest Neighbors

**Learning Outcomes:**

By the end of this lab session, students will be able to:

- Explain the basic mechanics of Logistic Regression and k-Nearest Neighbors (kNN).
- Implement the Sigmoid function, Log-Loss and Gradient Descent from scratch.
- Use a trained Logistic Regression model to produce probabilities and class predictions.
- Implement kNN from scratch using distance calculation and majority voting.
- Investigate the effect of different values of k (kNN).
- Compare from-scratch implementations with Scikit-learn implementations.

---

## Part A — Logistic Regression

### A.1 Dataset & Problem Setup

A university wants to predict whether a student will **Pass (1)** or **Fail (0)** an examination based on the number of hours studied.

| Hours Studied (X) | Result (Y) |
|---|---|
| 1 | 0 |
| 2 | 0 |
| 3 | 1 |
| 4 | 1 |
| 5 | 1 |
| 6 | 1 |

**Task 1**

1. Create NumPy arrays for X and Y.
2. Display the dataset using a scatter plot.
3. Identify the input feature, target variable.

---

### A.2 Sigmoid Function

Logistic Regression first calculates a linear score:

$$z = \theta_0 + \theta_1 x$$

The Sigmoid function converts the score into a probability:

$$\sigma(z)=\frac{1}{1+e^{-z}}$$

**Task 1**

1. Implement the Sigmoid function from scratch using NumPy.
2. Initialize $\theta_0=0$ and $\theta_1=0$.
3. Calculate the predicted probabilities for all observations.
4. What probability is produced for each observation at the initial parameter values?
5. Why is the Sigmoid function required in Logistic Regression?

**Interactive: Sigmoid Explorer**

Move parameters and observe how the Logistic Regression probability curve changes. Notice that the output always remains between 0 and 1. (Controls: $\theta_0$ range −8 to 8, $\theta_1$ range −5 to 5.)

---

### A.3 Log-Loss / Binary Cross-Entropy

Use the Binary Cross-Entropy cost function:

$$J(\theta)=-\frac{1}{m}\sum_{i=1}^m\left[y_i\log(h_i)+(1-y_i)\log(1-h_i)\right]$$

where:

$$y_i = Y$$
$$h_i = Sigmoid(z)$$

**Task 1**

1. Implement the Log-Loss function from scratch.
2. Calculate and display the initial cost.
3. What does a smaller Log-Loss value indicate?
4. What happens to the loss when the model confidently makes an incorrect prediction?
5. Why is a cost function required during model training?

---

### A.4 Gradient Descent

Gradients:

$$\frac{\partial J}{\partial \theta_0}=\frac{1}{m}\sum_{i=1}^m(h_i-y_i)$$

$$\frac{\partial J}{\partial \theta_1}=\frac{1}{m}\sum_{i=1}^m(h_i-y_i)x_i$$

Update rule:

$$\theta_j := \theta_j - \alpha\frac{\partial J}{\partial \theta_j}$$

**Task 1 – Train the Model**

1. Set a suitable learning rate $\alpha$ and initialize $\theta_0$ and $\theta_1$ to zero.
2. Calculate the gradients.
3. Update the parameters using Gradient Descent.
4. Repeat the process for a sufficient number of iterations.
5. Store the cost at each iteration.
6. Display the final values of $\theta_0$ and $\theta_1$.

**Task 2 – Observe Convergence**

1. Plot iteration number against Log-Loss.
2. What happens to the loss as training progresses?
3. What does it mean when the loss begins to stabilize?
4. What could happen if the learning rate is too large?
5. What could happen if the learning rate is too small?

**Interactive: Gradient Descent Visualization**

Run Gradient Descent one step at a time and watch the fitted probability curve change while Log-Loss decreases. (Controls: learning rate $\alpha$ range 0.01–1, Step +1, Run 50, Reset.)

---

### A.5 Making Logistic Regression Predictions

For a new observation:

$$P(Y=1|X)=\sigma(\theta_0+\theta_1X)$$

Using a threshold of 0.5:

$$\hat{y}=\begin{cases}1,&P(Y=1|X)\geq0.5\\0,&P(Y=1|X)<0.5\end{cases}$$

**Task**

1. Create a prediction function from scratch.
2. Predict the result for a student who studies for **2.5 hours**.
3. Display the predicted probability and predicted class.
4. Change the threshold from 0.5 to 0.7. Does the prediction change?
5. Explain the purpose of a classification threshold.

---

### A.6 Verify Logistic Regression Using Scikit-learn

1. Train `LogisticRegression` using the same dataset.
2. Obtain the predicted probability and class for 2.5 hours.
3. Compare the Scikit-learn result with your from-scratch implementation.
4. Are the learned parameter values exactly the same? Discuss possible reasons for differences.

---

## Part B — k-Nearest Neighbors (kNN)

### B.1 Problem Setup

Predict whether a student will **Pass (1)** or **Fail (0)** using two features: Hours Studied and Practice Tests Taken.

| Student | Hours | Practice Tests | Result |
|---|---|---|---|
| A | 1 | 1 | 0 |
| B | 2 | 1.5 | 0 |
| C | 2 | 3 | 0 |
| D | 4 | 5 | 1 |
| E | 5 | 4.5 | 1 |
| F | 5 | 2.5 | 1 |

New student: **Q = (4, 2)**.

---

### B.2 Calculate Euclidean Distance

$$d(p,q)=\sqrt{\sum_{j=1}^n(p_j-q_j)^2}$$

1. Create the training dataset using NumPy.
2. Implement Euclidean distance from scratch.
3. Calculate the distance from Q to every training observation.
4. Display the distances and sort the observations from nearest to farthest.

---

### B.3 Find the k Nearest Neighbors & Majority Vote

For $k=3$, select the three smallest distances and predict using:

$$\hat{y}=\operatorname{mode}\{y_{(1)},y_{(2)},\ldots,y_{(k)}\}$$

1. Set $k=3$.
2. Identify the three nearest observations.
3. Display their distances and class labels.
4. Apply majority voting.
5. Predict whether student Q will Pass or Fail.

---

### B.4 Investigate the Effect of k

1. Repeat the prediction for **k = 1, 3 and 5**.
2. Record the predicted class for each value of k.

| k | Predicted Class |
|---|---|
| 1 | |
| 3 | |
| 5 | |

1. Does the prediction change when k changes?
2. What happens when k is very small?
3. What happens when k is very large?
4. Why is an odd value of k commonly preferred for binary classification?

---

### B.5 Visualize kNN

1. Create a scatter plot with Hours Studied on the X-axis and Practice Tests on the Y-axis.
2. Use different markers or colours for Pass and Fail observations.
3. Plot the query point Q on the same graph.
4. Identify the nearest neighbors to Q for k=3.

**Interactive: kNN Explorer**

Click anywhere inside the graph to move the new student Q. Change k and observe which neighbors vote and how the prediction changes. (k options: 1, 3, 5; Reset Q available.)

---

### B.6 Create the Complete kNN Algorithm

Combine the previous steps into a function:

```
knn_predict(X_train, y_train, query, k)
```

The function should:

1. Calculate distances.
2. Sort the distances.
3. Select the k nearest observations.
4. Obtain their labels.
5. Perform majority voting.
6. Return the predicted class.

---

### B.7 Verify kNN Using Scikit-learn

1. Train `KNeighborsClassifier` using the same dataset.
2. Set `n_neighbors=3`.
3. Predict the class of Q = (4, 2).
4. Compare the prediction with your from-scratch implementation.

---

### B.8 Why Feature Scaling Matters for kNN

Consider two features:

- Age: 18–70
- Annual Income: 20,000–500,000

1. Which feature would dominate Euclidean distance?
2. Why could this produce misleading nearest neighbors?
3. Which preprocessing technique can reduce this problem?
4. Standardize the kNN input features and repeat the prediction.

---

## Part C — Understanding the Concepts

Answer each question analytically in 3–6 sentences: justify your reasoning rather than stating a definition, and where relevant, ground your answer in what you actually observed in the Sigmoid Explorer, Gradient Descent Visualization, or kNN Explorer above.

1. If Mean Squared Error were used instead of Log-Loss to train this Logistic Regression model, explain, by reasoning about the shape of each cost surface, why Gradient Descent would struggle to reliably reach the minimum.

2. A spam filter and a cancer-screening classifier both output a probability. Argue, in terms of the relative cost of false positives versus false negatives, whether each should use a threshold above, below, or exactly at 0.5.

3. Using what you observed in the kNN Explorer, explain how increasing k moves the model along the bias-variance trade-off, and justify what the decision boundary would look like if k equaled the total number of training points.

4. A classmate claims: "kNN is strictly better than Logistic Regression because with k=1 it can always achieve 100% accuracy on the training set." Critically evaluate this claim, considering both training and unseen data.

5. Logistic Regression produced a straight decision boundary in this lab, while kNN's boundary could bend around clusters of points. Explain the mathematical reason for this difference by referring to each model's hypothesis function.
