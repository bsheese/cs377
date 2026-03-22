# Logistic Regression 
Required video: [StatQuest introduction to logistic regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
Watch the video several times if you have not had coursework on Logistic Regression before. Take notes. Ask us questions. Watch it again. 

---

Logistic regression is a fundamental algorithm in machine learning used for solving binary classification problems, where the goal is to predict one of two possible outcomes. While its name includes the word "regression," logistic regression is used for classification tasks, not for predicting continuous values like linear regression. Logistic regression predicts the probability that an observation belongs to one of two classes. Specifically, it models the probability of an event happening, such as whether an email is "spam" or "not spam," or whether a customer will "buy" or "not buy" a product.

For example:

*   In a medical context, we might want to predict whether a patient has a disease (1) or not (0).
*   In marketing, we might want to predict whether a customer will make a purchase (1) or not (0).

The key idea behind logistic regression is that instead of directly predicting whether something will happen or not (1 or 0), it predicts the probability of it happening. If the predicted probability is greater than 0.5, we might predict the event will occur (class 1), and if it's less than 0.5, we predict the event will not occur (class 0).

---

## How Does Logistic Regression Work?

At its core, logistic regression is quite similar to linear regression in that it tries to establish a relationship between the features and the outcome. The difference is that in logistic regression, the outcome is binary (e.g., yes/no, success/failure), and instead of predicting a direct value (as in linear regression), it predicts a probability between 0 and 1.

In a linear regression model, the output is calculated as:

$$y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k$$

However, in logistic regression, the output needs to be a probability, which means it must be between 0 and 1. To achieve this, logistic regression uses a special function called the **logistic function** (or **sigmoid function**), which transforms any value into a number between 0 and 1.

The logistic function is defined as:

$$p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k)}}$$

Where:

*   $p$ is the predicted probability of the event occurring (e.g., the probability of the email being spam).
*   $\beta_0$ is the intercept (the baseline value of the logit when all predictors are 0).
*   $\beta_1, \beta_2, \ldots, \beta_k$ are the coefficients that represent the impact of each predictor variable $X_1, X_2, \ldots, X_k$ on the outcome.

The logistic function squashes the output of the linear equation into the range [0, 1], which makes it ideal for predicting probabilities.

---

## Key Features of Logistic Regression

1.  Prediction of Probabilities: Logistic regression doesn't predict classes directly (0 or 1). Instead, it predicts a probability value between 0 and 1, which tells us how likely the event is to occur. We can then use a threshold (usually 0.5) to classify the event:
    *   If $p \geq 0.5$, classify the observation as class 1.
    *   If $p < 0.5$, classify the observation as class 0.

2.  Linear Relationship in the Log-Odds: Although logistic regression predicts probabilities, it is still considered a **linear model** because the relationship between the predictor variables and the **log-odds** of the outcome is linear. The log-odds is simply the logarithm of the odds of the event happening, and the logistic regression equation can be written as:

    $$ \log\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k $$

    In this equation:
    *   The left-hand side represents the log-odds of the event happening.
    *   The right-hand side is a linear combination of the predictors, just like in linear regression.

3.  Interpretation of Coefficients: Each coefficient ($\beta_j$) in logistic regression represents the effect of a one-unit increase in the corresponding predictor variable ($X_j$) on the log-odds of the event happening. In simpler terms, if $\beta_1$ is positive, increasing $X_1$ increases the log-odds (and therefore the probability) of the event. If $\beta_1$ is negative, increasing $X_1$ decreases the probability of the event.

4.  Binary Outcome: Logistic regression is mainly used for binary classification problems, meaning the outcome has two possible categories (e.g., 0 and 1). However, as we'll see later, it can be extended to handle multiclass classification as well.

---

## Example: Predicting Buying Behavior 

Let's say we want to predict whether a customer will buy a product (1 = yes, 0 = no) based on two factors: how much money they have already spent in the store and how many times they've visited the website.

We could use logistic regression to model the probability of the customer making a purchase. The equation might look something like this:

$$ \log\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 (\text{spend}) + \beta_2 (\text{visits}) $$

Where:

*   $p$ is the probability that the customer will make a purchase.
*   "spend" is the total amount of money the customer has spent so far.
*   "visits" is the number of times the customer has visited the website.

By fitting this model, we can estimate the coefficients ($\beta_0, \beta_1, \beta_2$), which tell us how spending and website visits influence the probability of making a purchase.

### Why is Logistic Regression Useful in Machine Learning?

1.  Simple and Interpretable: Logistic regression is relatively simple compared to more complex machine learning models like decision trees or neural networks. The coefficients have straightforward interpretations, which is why logistic regression is often used in fields where interpretability is important (e.g., healthcare, finance).
2.  Widely Used for Binary Classification: Logistic regression is one of the most commonly used algorithms for binary classification problems in machine learning. It's efficient, easy to implement, and provides good performance for many problems.
3.  Probabilistic Output: Logistic regression provides probabilities, which is useful in cases where we want to understand the likelihood of an outcome, not just a hard classification. For example, in medical diagnosis, knowing that a patient has a 90% chance of having a disease is more informative than just predicting that they have the disease.

---

## Mathematics of Logistic Regression

Before we get into the logistic function, it's essential to understand the difference between probability and odds, two important concepts in logistic regression.

*   Probability (p): This refers to the likelihood of an event happening, and it always takes a value between 0 and 1. For example, if you say there is a 0.8 probability that it will rain today, that means you believe there's an 80% chance of rain.

    $$ 0 \leq p \leq 1 $$

*   Odds: The odds of an event happening are defined as the ratio of the probability of the event happening to the probability of the event not happening. In other words, the odds tell you how much more likely the event is to happen compared to it not happening.

    $$ \text{Odds} = \frac{p}{1 - p} $$

    For example, if the probability of rain is 0.8, the odds are:

    $$ \frac{0.8}{1 - 0.8} = 4 $$

    This means the odds of rain are 4 to 1 in favor of rain.

### The Logistic (Sigmoid) Function

The heart of logistic regression is the logistic function, also called the sigmoid function. This function is used to map any real-valued number into the range [0, 1], which is exactly what we need for predicting probabilities. No matter how large or small the input to the logistic function is, it will always output a value between 0 and 1.

The logistic function is given by the following formula:

$$ p = \frac{1}{1 + e^{-z}} $$

Where:

*   $p$ is the predicted probability (i.e., the probability that the outcome is 1).
*   $z$ is the output of the linear equation (we'll see this next).
*   $e$ is the mathematical constant (approximately 2.718).

The logistic function has an S-shaped curve (sigmoidal), meaning that as the value of $z$ increases, the predicted probability $p$ approaches 1, and as $z$ decreases, $p$ approaches 0.

The logistic function converts a linear combination of predictor variables into a probability, which is essential for modeling binary outcomes.

### The Logit Transformation

Now that we know about the logistic function, we need to understand how it relates to log-odds. Logistic regression is based on modeling the log-odds of the outcome, which is why it's considered a generalized linear model.

The logit function is the logarithm of the odds:

$$ \text{logit}(p) = \log\left(\frac{p}{1 - p}\right) $$

This equation tells us how the log-odds of an event (such as a customer making a purchase) relate to the probability of that event happening. In logistic regression, we model the log-odds as a linear combination of the predictor variables, which allows us to use familiar linear regression techniques.

### Logistic Regression Equation

In logistic regression, we use the logit function to link the probability of the outcome (the event happening) to a linear combination of predictor variables. The logistic regression model can be written as:

$$ \text{logit}(p) = \log\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k $$

Where:

*   $p$ is the probability of the event happening (e.g., a customer making a purchase).
*   $\beta_0$ is the intercept (the baseline log-odds when all predictor variables are 0).
*   $\beta_1, \beta_2, \ldots, \beta_k$ are the coefficients that measure the impact of each predictor variable.
*   $X_1, X_2, \ldots, X_k$ are the values of the predictor variables.

In this equation:

*   The left-hand side represents the log-odds of the outcome (e.g., the log-odds of a customer making a purchase).
*   The right-hand side is a linear combination of the predictor variables (e.g., the customer's age, income, and shopping history).

The coefficients ($\beta_1, \beta_2, \ldots$) represent how much each predictor variable affects the log-odds of the event happening. For example:

*   If $\beta_1$ is positive, increasing $X_1$ will increase the log-odds of the event happening.
*   If $\beta_1$ is negative, increasing $X_1$ will decrease the log-odds of the event happening.

### Interpreting Coefficients in Logistic Regression

Interpreting the coefficients in logistic regression is slightly different from interpreting coefficients in linear regression because they represent effects on the log-odds rather than directly on the outcome.

*   A positive coefficient means that as the predictor variable increases, the log-odds of the outcome happening also increase. This, in turn, increases the probability of the event occurring.
*   A negative coefficient means that as the predictor variable increases, the log-odds of the outcome decrease, reducing the probability of the event happening.

For example, if you're predicting whether a customer will buy a product and you find that the coefficient for "number of previous purchases" is positive, it means that customers who have bought from you before are more likely to buy again.

### Converting Log-Odds Back to Probability

Once we've calculated the log-odds using the logistic regression equation, we can convert the log-odds back to a probability by applying the inverse of the logit function, which is the logistic function. This gives us the predicted probability $p$.

The formula for converting log-odds back to probability is:

$$ p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k)}} $$

This formula is used to make predictions in logistic regression. It ensures that the final result is a probability between 0 and 1, which can then be used to classify the outcome.

### Summary of the Mathematics

*   Logistic regression models the log-odds of an event happening as a linear combination of the predictor variables.
*   The logistic function (sigmoid function) is used to convert the linear combination of predictor variables into a probability between 0 and 1.
*   The coefficients represent how changes in the predictor variables affect the log-odds (and therefore the probability) of the outcome.
*   Once we have the log-odds, we can use the logistic function to convert them back to probabilities, which can be used to make predictions about the outcome.

---

## Model Fitting and Optimization: Maximum Likelihood Estimation (MLE)


Now that we understand the core mathematical concepts behind logistic regression, the next step is to explore how we fit a logistic regression model to data. The process of determining the best values for the model's coefficients is called model fitting. In logistic regression, this is typically done using a technique called Maximum Likelihood Estimation (MLE).

In this section, we will explain what MLE is, why it's used in logistic regression, and how it works. We'll also discuss optimization techniques like gradient descent, which are used to actually find the best coefficients.

### What is Maximum Likelihood Estimation (MLE)?

Maximum Likelihood Estimation is a method used to estimate the parameters (coefficients) of a model by finding the values that make the observed data most likely. In the context of logistic regression, MLE helps us find the best-fitting values for the coefficients ($\beta_0, \beta_1, \ldots, \beta_k$) so that the predicted probabilities closely match the actual outcomes in the data.

The idea behind MLE is simple: we choose the values of the coefficients that maximize the likelihood of observing the given data, assuming that the model (logistic regression in this case) is correct.

### Likelihood in Logistic Regression

To understand MLE, let's first break down the concept of likelihood. In logistic regression, the likelihood represents the probability of the observed outcomes (e.g., 1 or 0) given the model's predictions. For a single data point, the likelihood is calculated based on the predicted probability $p$ and the actual outcome $Y$:

*   If the actual outcome is 1 (i.e., the event occurred), the likelihood is equal to the predicted probability $p$ of the event occurring.
*   If the actual outcome is 0 (i.e., the event did not occur), the likelihood is equal to $1 - p$, which is the predicted probability that the event did not occur.

### Log-Likelihood Function

In practice, working with the product of probabilities can be computationally difficult, especially for large datasets. Therefore, it is common to take the logarithm of the likelihood function, which turns the product into a sum and simplifies the math. This new function is called the log-likelihood. The goal of MLE is to find the values of $\beta_0, \beta_1, \dots, \beta_k$ that maximize this log-likelihood function. In other words, we want to find the coefficients that make the observed data most probable.

### How Does MLE Work in Logistic Regression?

1.  Initial Guess: We start with an initial guess for the values of the coefficients (typically set to random values or zeros).
2.  Calculate the Likelihood: For each set of coefficients, we calculate the predicted probabilities ($p_i$) for all observations and compute the likelihood of the observed outcomes.
3.  Adjust the Coefficients: The model adjusts the coefficients ($\beta_0, \beta_1, \dots, \beta_k$) to increase the likelihood of the observed data. This process is repeated until the coefficients converge to values that maximize the likelihood.

---

### Optimization Techniques: Gradient Descent

To maximize the log-likelihood function, we need an optimization algorithm that can iteratively adjust the coefficients until we reach the maximum point. One of the most commonly used optimization algorithms in logistic regression is gradient descent. Gradient descent is an optimization algorithm that finds the minimum or maximum of a function by iteratively moving in the direction of the steepest ascent or descent. In logistic regression, we use gradient descent to adjust the coefficients to maximize the log-likelihood function.

Here's how gradient descent works:

1.  Compute the Gradient: The gradient is a vector of partial derivatives that tells us the direction and rate of change of the log-likelihood function with respect to each coefficient. In simple terms, the gradient tells us how to change the coefficients to increase the likelihood of the observed data.
2.  Update the Coefficients: The coefficients ($\beta_0, \beta_1, \ldots, \beta_k$) are updated by moving in the direction of the gradient. The size of each step is controlled by a parameter called the learning rate ($\alpha$):

    $$\beta_j = \beta_j + \alpha \frac{\partial \log L}{\partial \beta_j}$$

    This equation updates each coefficient $$\beta_j$$ by taking a small step in the direction of the gradient.
3.  Repeat Until Convergence: Gradient descent repeatedly updates the coefficients until the changes become very small, meaning that we've reached the maximum likelihood and the model has "converged." At this point, the coefficients are optimized.

In practice, gradient descent continues updating the coefficients until one of the following stopping criteria is met:

*   Maximum iterations: The algorithm stops after a fixed number of iterations, even if the log-likelihood hasn't been maximized.
*   Tolerance: The algorithm stops when the changes in the coefficients or the log-likelihood become smaller than a predefined threshold, indicating that the optimization process has converged.

Choosing the right learning rate is crucial in gradient descent. If the learning rate is too high, the algorithm might overshoot the maximum point. If it's too low, the algorithm may take too long to converge.


### Summary of Maximum Liklihood

*   Maximum Likelihood Estimation (MLE) is the method used to fit the logistic regression model by finding the coefficients that maximize the likelihood of observing the data.
*   The likelihood function represents the probability of the observed outcomes given the model's predictions, and we maximize the log-likelihood to find the best coefficients.
*   Gradient descent is a common optimization algorithm used to adjust the coefficients by iteratively moving in the direction of the steepest increase in the log-likelihood function.
*   The optimization process stops when the model converges, meaning that the coefficients are as close as possible to the optimal values.
