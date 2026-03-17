# Introduction to Classification

Classification is a foundational task in **supervised machine learning** where the objective is to assign an observation to a predefined category or **class**. Unlike regression, which predicts continuous quantities, classification focuses on **discrete** and **labeled** outcomes.

Because it is a "supervised" task, classification requires a **ground truth** dataset—a collection of historical observations where the correct labels are already known. The model learns patterns from this labeled data to make predictions on new, unseen observations.

### **The Probabilistic Framework**
In a mathematical sense, we can think of classification as finding the probability $P(Y | X)$, where:
*   **$Y$** is the **Label** (the outcome we want to predict).
*   **$X$** is the set of **Features** (the input data or predictors).

Features can be **numerical** (such as age or income) or **categorical** (such as gender or zip code). The model's job is to map these features to a probability for each possible category. For example, in a medical diagnosis task, $X$ might include a patient's age and cholesterol levels, while $Y$ is the diagnosis (e.g., "Healthy" vs. "At Risk"). The model calculates the probability that a patient with those specific features belongs to the "At Risk" category.

### **Hard vs. Soft Classification**
Modern classification models typically provide two types of outputs:
1.  **Soft Predictions (Probabilities):** The model outputs a value between 0 and 1 (e.g., "There is an 85% chance this email is spam").
2.  **Hard Predictions (Labels):** Based on the probability, we apply a **decision threshold** to assign a final label (e.g., "This email is Spam").

By default, the threshold is often set at **0.5**. If the probability is greater than or equal to 0.5, the observation is assigned to Class 1; otherwise, it is assigned to Class 0. However, this threshold can be adjusted depending on the stakes. In medical testing, for instance, we might lower the threshold to 0.2 to ensure we don't miss any potential cases, even if it results in more "false alarms."

---

### **Binary vs. Multiclass Classification**

Classification tasks are generally categorized by the number of possible outcomes:

*   **Binary Classification:** The simplest form, involving exactly two classes. Examples include predicting whether a transaction is "Fraudulent" or "Legitimate," or whether a student will "Pass" or "Fail."
*   **Multiclass Classification:** Involves more than two categories. Examples include identifying handwritten digits (0–9) or classifying images into categories like "Apple," "Orange," or "Banana."

---

### **Classification vs. Regression: Why Linear Models Fail**

While both regression and classification use input features to make predictions, they differ fundamentally in their output constraints and objectives.

| Feature | **Regression** | **Classification** |
| :--- | :--- | :--- |
| **Output Type** | Continuous numerical value | Discrete class label |
| **Prediction Range** | Unbounded ($-\infty$ to $+\infty$) | Probabilistic ($0$ to $1$) |
| **Objective Function** | Minimize Residual Distance (e.g., MSE) | Maximize Likelihood (e.g., Log Loss) |
| **Examples** | House prices, Temperature | Spam detection, Medical status |

**The "Out-of-Bounds" Problem:**
A common question is: *Why can't we just use linear regression for classification?*
If we use a linear regression line to predict a binary outcome, the line will eventually extend above 1 or below 0. Since a probability cannot be 150% or -20%, linear regression is mathematically inappropriate for classification. 

**The Learning Mechanism:**
Furthermore, the "goal" of the model changes. In regression, we want to be as close to the points as possible. In classification, we use a different objective called **Log Loss** (or Cross-Entropy). This function heavily penalizes the model for being "confident and wrong"—for example, predicting a 99% probability of class 1 when the actual label is class 0.

---

### **Visualizing the Decision Boundary**

In classification, we are essentially partitioning the **feature space**. Imagine a graph where the $x$-axis is "Hours Studied" and the $y$-axis is "Attendance." If we plot students who passed and failed, a classification model attempts to draw a line—called a **Decision Boundary**—that separates the two groups.

*   Observations on one side of the boundary are classified as "Pass."
*   Observations on the other side are classified as "Fail."
*   The closer an observation is to the boundary, the more "uncertain" the model is (probability near 0.5).

### **Measuring Success: Evaluation Metrics**

Once a model is trained, we must assess how well it performs. Unlike regression, where we look at the average error, classification uses metrics based on the correctness of labels:

*   **Accuracy:** The percentage of total predictions that were correct. While intuitive, accuracy can be misleading if the classes are imbalanced (e.g., if 99% of transactions are legitimate, a model that predicts "Legitimate" for everything will have 99% accuracy but fail to find any fraud).
*   **The Confusion Matrix:** A tool used to break down errors into four categories:
    *   **True Positives (TP):** Correctly predicted the positive class.
    *   **True Negatives (TN):** Correctly predicted the negative class.
    *   **False Positives (FP):** Predicted positive when it was actually negative ("False Alarm").
    *   **False Negatives (FN):** Predicted negative when it was actually positive ("Missed Opportunity").

---

### **Core Applications in Industry**

Classification is applied across various domains where decision-making must be automated or evidence-based.

#### **1. Diagnostic Filtering (Medicine & Security)**
In high-stakes environments, classification acts as a filter to separate "signal" from "noise."
*   **Medical Diagnosis:** Identifying disease presence based on biomarkers.
*   **Spam Detection:** Filtering harmful or irrelevant content from communication channels.
*   **Technical Context:** In these fields, **False Negatives** (missing a disease) are often much costlier than **False Positives**.

#### **2. Risk Assessment (Finance)**
Financial institutions use classification to evaluate the "likelihood of default."
*   **Credit Scoring:** Determining if a loan applicant should be approved.
*   **Fraud Detection:** Flagging transactions that deviate from a user's typical behavior.
*   **Technical Context:** These datasets often suffer from **Class Imbalance** (e.g., 99.9% of transactions are legitimate), requiring models that can accurately identify the rare "Fraud" class.

#### **3. Predictive Behavior (Marketing & Social Science)**
Organizations use classification to anticipate future actions based on historical data.
*   **Customer Churn:** Predicting which users are likely to cancel a subscription so they can be offered incentives to stay.
*   **Election Outcomes:** Modeling voter turnout or candidate preference based on demographic and polling data.

---

I recommend you watch this [StatQuest introductory video](https://www.youtube.com/watch?v=yIYKR4sgzI8&list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe&index=1) before proceeding to the next section on Logistic Regression.

---
