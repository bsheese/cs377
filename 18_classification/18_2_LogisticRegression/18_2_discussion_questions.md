# Discussion Questions: 18_2 — Logistic Regression

---

## 18_2_0_1: Logistic Regression Foundations (Reading)

### Why Not Linear Regression?

1. A linear regression model fitted to a binary outcome (0/1) can predict probabilities below 0 and above 1. Why are these predictions problematic? Give a specific example where they would fail.

2. The logistic (sigmoid) function maps any real number to the interval (0, 1). What is the shape of this function, and why does this shape match the behavior of probabilities?

3. "Logistic regression is a linear model." In what sense is this true? In what sense could it be misleading?

### Coefficients and Log-Odds

4. In logistic regression, coefficients represent effects on the *log-odds*, not on the probability directly. Why is this harder to interpret than a linear regression coefficient?

5. A positive coefficient means "increases the probability of the outcome." Why is it more precise to say "increases the log-odds"? When would these two statements lead to different interpretations?

6. Converting log-odds to an odds ratio requires $e^{\beta}$. If $\beta = 0$, what is the odds ratio? What does this mean?

### Maximum Likelihood Estimation

7. OLS minimizes squared error to fit a linear model. Logistic regression uses Maximum Likelihood Estimation (MLE). Why can't OLS be used for logistic regression?

8. The likelihood of observing a set of outcomes is the product of probabilities. Why do we use log-likelihood (sum of logs) instead of the raw likelihood (product of probabilities)?

9. MLE uses gradient descent to find optimal coefficients. What is a "learning rate," and what happens if it is set too high vs. too low?

---

## 18_2_1: Logistic Regression with Titanic

### Exploratory Data Analysis

1. The EDA shows women survived at a much higher rate than men, and 1st class passengers survived at a much higher rate than 3rd class. Why is it important to explore these patterns before fitting the model?

2. Survival rate is 38% in the training set. A naive baseline that always predicts "died" achieves 62% accuracy. What does this baseline tell you about the minimum bar a model must clear to be considered useful?

3. Fare is log-transformed before modeling. What does the distribution of raw fare look like that motivates this transformation? How does log-transformation help?

### Feature Engineering Decisions

4. `sibspouse` and `parentchild` are created as binary indicators (0 or 1) rather than keeping the raw count. What information is lost by binarizing? What is gained?

5. Passenger class is one-hot encoded with `drop_first=True`. Why is `drop_first=True` necessary? What would go wrong without it?

6. The model uses `sex` as a binary feature (male=0, female=1). What does the sign of the resulting coefficient tell you about the direction of the relationship?

### Pipelines and Data Leakage

7. `StandardScaler` is placed inside a `Pipeline` rather than applied to the whole dataset before splitting. Why does this matter? What would leak if you scaled first?

8. The pipeline chains `ColumnTransformer → LogisticRegression`. What does `remainder='passthrough'` do, and why is it needed?

9. Cross-validation produces scores [0.797, 0.782, 0.842, 0.827, 0.744]. The mean is 0.798, std is 0.069. How would you use these numbers when reporting the model's performance?

### Interpreting Odds Ratios

10. The odds ratio for sex (female=1) is approximately 12.28. In plain English, what does this mean?

11. The odds ratio for `pclass_3` (vs. 1st class) is approximately 0.17. What does an odds ratio below 1 mean? How much worse are 3rd class passengers' survival odds?

12. Age has OR ≈ 0.59. How do you interpret this? Would you prefer to know this as an odds ratio or as a probability change? Why?

13. Two features — `sibspouse` and `parentchild` — both have OR < 1. Does this surprise you? What real-world explanation might account for family size being associated with lower survival?

### Evaluation Metrics

14. The model has precision = 0.73 and recall = 0.64 for the "survived" class. What do these two numbers mean in the context of the Titanic? Which error type (false positive vs. false negative) might you care more about?

15. The model predicts deaths better than survivals (higher recall for class 0). Why might this happen, given the class imbalance (62% died)?

16. AUC = 0.842. Explain the probabilistic interpretation of AUC in plain English using a concrete example from the Titanic context.

### Threshold Tuning

17. At t=0.3 (lower threshold), recall increases for survivors but precision decreases. Explain why lowering the threshold produces this tradeoff.

18. A medical screening application needs high recall (catch as many true positives as possible). A fraud detection application might prioritize precision (avoid false alarms). What threshold direction would each context favor?

19. The threshold t=0.4 happens to maximize accuracy. Why might you NOT want to use this threshold for a specific business decision?

---

## 18_2_2: Interpretability — When the Model Must Explain Itself

### Statsmodels vs. Sklearn

1. Sklearn's `LogisticRegression` and statsmodels' `Logit` fit the same model. What does each library give you that the other does not? Why does the notebook use both?

2. Statsmodels reports a standard error and p-value for every coefficient. What question does a p-value answer about a coefficient? What does it *not* tell you?

### Confidence Intervals and the Forest Plot

3. A feature's odds ratio is 1.8 with a 95% CI of [0.9, 3.6]. Another feature's OR is 1.3 with a CI of [1.2, 1.4]. Which estimate should you trust more, and which feature has the stronger *estimated* effect? Why are these different questions?

4. On the forest plot, features whose CI crosses OR = 1 are drawn in gray. What does crossing 1 mean substantively? Why is OR = 1 (rather than 0) the "no effect" reference line?

5. In the Titanic model, `sibspouse` and `parentchild` have wide CIs that cross 1. Does this mean family size has no effect on survival? What is the more careful statement?

6. Why is the forest plot drawn on a log-scale x-axis? What would be misleading about plotting odds ratios on a linear axis?

### LR vs. XGBoost Explanations

7. Both models put `sex_f` at the top of their importance rankings, yet the notebook argues they "say very different things." List the three pieces of information the LR odds ratio gives you that XGBoost's gain-based importance does not.

8. XGBoost's feature importances are unsigned. Why can't a tree ensemble's importance score have a direction the way a regression coefficient does?

### Regulation and Individual Explanations

9. Under the Equal Credit Opportunity Act, a lender must give specific reasons for denying credit. Walk through how the individual-prediction decomposition (coefficient × scaled feature value) produces such a reason. What is the equivalent procedure for XGBoost, and why is it harder?

10. The notebook claims a 2% AUC improvement from a black-box model is "often not a legal option" in regulated domains. Construct the cost-benefit argument: when is the accuracy gain worth the interpretability loss, and who gets to decide?

---

## 18_2_3: From One Neuron to Many — The Bridge to Neural Networks

### LR = One Neuron

1. The notebook verifies numerically that `sigmoid(W·x + b)` reproduces sklearn's `predict_proba` to machine precision. What are the neural-network names for W, b, and the sigmoid in this expression?

2. If logistic regression *is* a single neuron, what exactly does a neural network add? Where in the network does a "logistic regression" still live?

### The XOR Problem

3. Explain why no straight line can separate the XOR classes. What accuracy does logistic regression achieve on XOR, and why is that number exactly what you'd expect from a coin flip?

4. The hidden layer "transforms the input into a new space where the data becomes linearly separable." Explain this idea in your own words. Why does a straight line in the transformed space look like a curve in the original space?

### Activation Functions

5. A colleague proposes using sigmoid activations in every hidden layer "so all the intermediate values are probabilities." What is wrong with this reasoning? What problem does ReLU solve?

6. Why is sigmoid still the standard choice for the *output* layer of a binary classifier, even in networks that use ReLU everywhere else?

### The Interpretability Tax

7. The shallow MLP has 145 parameters vs. logistic regression's 8, yet scores about the same on the Titanic test set. Give two reasons the extra capacity doesn't help here.

8. The notebook calls the loss of interpretability the "interpretability tax." For each of these applications, say whether you would pay it and why: (a) credit scoring, (b) photo tagging, (c) hospital readmission risk, (d) movie recommendations.

---

## 18_2_9: Credit Risk Exercise — Interpretability Constraints

1. The compliance scenario rules out XGBoost before any model is fit. Is this premature? What would you need to know about the regulatory requirement to decide whether a post-hoc explanation tool (like SHAP) could make XGBoost acceptable instead?

2. The exercise one-hot encodes the categorical features with `drop_first=True` for logistic regression, while 18_1 passed them to XGBoost natively. Why does logistic regression need the encoding when XGBoost does not?

3. After fitting, some odds ratios for rare categories have enormous confidence intervals. What does this tell you about how much data supports those coefficients? Should they appear in an adverse action notice?

4. Your forest plot will likely show `checking_status` categories among the strongest predictors — consistent with XGBoost's feature importance in 18_1_1. Why is agreement between two very different models reassuring evidence about the data rather than about either model?

5. If XGBoost achieves higher AUC than your logistic regression on this dataset, write the two-or-three sentence recommendation you would give the bank. Which model do you recommend, and what is the deciding factor?
