# Discussion Questions: 18_2 Logistic Regression

---

## 18_2_0_2: Logistic Regression in Plain Language (Reading)

### Yes-or-No Outcomes

1. A linear regression fitted to a 0/1 outcome can predict values below 0 and above 1. Why are these predictions a problem? Use the straight line fitted to the Challenger launches as your example.

2. The comparison of past flights before the Challenger launch focused on flights that had damage. What can you learn from all 23 launches that you cannot learn from the seven with damage?

3. The model gives a probability of 0.9996 for O-ring damage at 31°F. Give two reasons to be careful with that number. What do the data support saying instead?

### Odds and Log-Odds

4. Two free-throw shooters make 80% and 20% of their shots. Compare them using odds and then using log-odds. What does taking the log do to the comparison?

5. A player who goes from 50% to 59% and a player who goes from 90% to 99% both improve by 9 percentage points. Explain, using odds or log-odds, why the second change is larger in logistic regression.

### Odds Ratios and Coefficients

6. In the 1999 referral study, how did rates of 91% and 85% turn into reports that one group was "40% less likely" to be referred? Write a correct sentence about the odds ratio and a correct sentence about the probabilities.

7. When is an odds ratio close to the ratio of the two probabilities, and when are they far apart?

8. Someone asks how much a ten-degree change in temperature changes the probability of damage. Why can you not answer with one number, and what can you say that is true at every temperature?

9. "Logistic regression is a linear model." In what sense is this true? In what sense could it be misleading?

### Maximum Likelihood

10. How does maximum likelihood use the probabilities that a model gave to a launch with damage and to a launch without damage? Explain with the two weather forecasters or the two candidate curves.

---

## 18_2_0_3: The Math Behind Logistic Regression (Reading)

### Log-Odds and the Sigmoid

1. Explain in words what each of the five algebra steps undoes in going from $\ln(p/(1-p)) = z$ to the sigmoid. Why does the result always stay between 0 and 1?

2. Lowering the threshold from 0.5 to 0.2 moves the cutoff temperature from 64.8°F to 70.8°F. Why do the coefficients stay the same?

### Coefficients

3. In the algebra that shows $e^{\beta_1}$ is the odds ratio, what cancels? Which part of the cancellation is the reason for saying "all else equal"?

4. If $\beta = 0$, what is the odds ratio? What does that say about the feature?

5. A positive coefficient always raises both the log-odds and the probability. Why, then, is it more precise to describe the coefficient as a change in log-odds?

6. Measuring temperature as $(\text{temperature} - 70)/10$ changes the intercept and the coefficient but none of the predictions. What does each of the two new numbers mean?

### Likelihood and Log Loss

7. Explain the difference between probability and likelihood. When a model is being fitted, what is fixed and what is being asked about?

8. The likelihood is a product of probabilities. Why do we work with its log, and why does that not change which coefficients are best?

9. The launch at 75°F contributes about a quarter of the total loss. Why is its loss so large?

10. Linear regression chooses the line with the smallest sum of squared residuals. Why does least squares not work on the log-odds scale, and what plays its part in logistic regression?

### Gradient Descent

11. In the first step, why does the intercept go down, and why does the slope become negative? Answer in terms of the errors $y_i - p_i$.

12. What is a learning rate? What happened to the log loss with learning rates of 0.05 and 30, and why?

13. When the search has converged, the predicted probabilities for the 23 launches add up to 7. Why?

### Assumptions

14. If every launch below 65°F had damage and no other launch did, what would happen to the fitted coefficient? How does scikit-learn's default penalty help?

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

9. Cross-validation produces scores [0.797, 0.782, 0.842, 0.827, 0.744]. The mean is 0.798, and the notebook reports it as 0.798 (+/- 0.069), where 0.069 is two standard deviations. How would you use these numbers when reporting the model's performance?

### Interpreting Odds Ratios

10. The odds ratio for sex (female=1) is approximately 12.28. In plain English, what does this mean?

11. The odds ratio for `pclass_3` (vs. 1st class) is approximately 0.17. What does an odds ratio below 1 mean? How much worse are 3rd class passengers' survival odds?

12. Age has OR ≈ 0.59. Age was standardized before fitting, so this odds ratio describes a one-standard-deviation increase in age (about 14 years). How do you interpret it? Would you prefer to know this as an odds ratio or as a probability change? Why?

13. Two features, `sibspouse` and `parentchild`, both have OR < 1. But in the raw data, passengers traveling with a sibling or spouse survived at a higher rate than those without (0.47 vs. 0.35), and so did passengers traveling with a parent or child (0.51 vs. 0.35). How can both be true? What does "all else equal" add?

14. Passenger A's predicted probability goes from 0.361 to 0.874 when only `sex` is changed. Where does the odds ratio of 12.28 appear in that change, and why is it incorrect to say that the passenger became 12 times as likely to survive?

15. Explain how the intercept and the "beta * input" columns produce passenger B's predicted probability of 0.947. Which contribution is the largest?

### Evaluation Metrics

16. The model has precision = 0.73 and recall = 0.64 for the "survived" class. What do these two numbers mean in the context of the Titanic? Which error type (false positive vs. false negative) might you care more about?

17. The model predicts deaths better than survivals (higher recall for class 0). Why might this happen, given the class imbalance (62% died)?

18. AUC = 0.842. Explain the probabilistic interpretation of AUC in plain English using a concrete example from the Titanic context.

### Threshold Tuning

19. At t=0.3 (lower threshold), recall increases for survivors but precision decreases. Explain why lowering the threshold produces this tradeoff.

20. A medical screening application needs high recall (catch as many true positives as possible). A fraud detection application might prioritize precision (avoid false alarms). What threshold direction would each context favor?

21. The threshold t=0.4 happens to maximize accuracy. Why might you NOT want to use this threshold for a specific business decision?

---

## 18_2_2: Interpretability: When the Model Must Explain Its Predictions

### Statsmodels vs. Sklearn

1. Sklearn's `LogisticRegression` and statsmodels' `Logit` fit the same model. What does each library give you that the other does not? Why does the notebook use both?

2. Statsmodels reports a standard error and p-value for every coefficient. What question does a p-value answer about a coefficient? What does it *not* tell you?

### Confidence Intervals and the Forest Plot

3. A feature's odds ratio is 1.8 with a 95% CI of [0.9, 3.6]. Another feature's OR is 1.3 with a CI of [1.2, 1.4]. Which estimate should you trust more, and which feature has the stronger *estimated* effect? Why are these different questions?

4. On the forest plot, features whose CI crosses OR = 1 are drawn in gray. What does crossing 1 mean substantively? Why is OR = 1 (rather than 0) the "no effect" reference line?

5. In the Titanic model, `sibspouse` and `parentchild` have wide CIs that cross 1. Does this mean family size has no effect on survival? What is the more careful statement?

6. Why is the forest plot drawn on a log-scale x-axis? What would be misleading about plotting odds ratios on a linear axis?

### LR vs. XGBoost Explanations

7. Both models put `sex_f` at the top of their importance rankings, yet the notebook argues that they tell you quite different things. List the three pieces of information the LR odds ratio gives you that XGBoost's gain-based importance does not.

8. XGBoost's feature importances are unsigned. Why can't a tree ensemble's importance score have a direction the way a regression coefficient does?

### Regulation and Individual Explanations

9. Under the Equal Credit Opportunity Act, a lender must give specific reasons for denying credit. Walk through how the individual-prediction decomposition (coefficient × scaled feature value) produces such a reason. What is the equivalent procedure for XGBoost, and why is it harder?

10. The notebook says that in regulated domains a model that is hard to explain is "often not a legal option," even if its AUC is 2% higher. Construct the cost-benefit argument: when is the accuracy gain worth the interpretability loss, and who gets to decide?

---

## 18_2_3: From One Neuron to Many: Connecting Logistic Regression to Neural Networks

### LR = One Neuron

1. The notebook verifies numerically that `sigmoid(W·x + b)` reproduces sklearn's `predict_proba` to machine precision. What are the neural-network names for W, b, and the sigmoid in this expression?

2. If logistic regression *is* a single neuron, what exactly does a neural network add? Where in the network does a "logistic regression" still live?

### The XOR Problem

3. Explain why no straight line can separate the XOR classes. What accuracy does logistic regression achieve on XOR, and why is that number exactly what you'd expect from a coin flip?

4. The hidden layer "transforms the input into a new space where the data becomes linearly separable." Explain this idea in your own words. Why does a straight line in the transformed space look like a curve in the original space?

### Activation Functions

5. A colleague proposes using sigmoid activations in every hidden layer "so all the intermediate values are probabilities." What is wrong with this reasoning? What problem does ReLU solve?

6. Why is sigmoid still the standard choice for the *output* layer of a binary classifier, even in networks that use ReLU everywhere else?

### The Interpretability Trade-Off

7. The shallow MLP has 145 parameters vs. logistic regression's 8, yet scores about the same on the Titanic test set. Give two reasons the extra capacity doesn't help here.

8. Moving from logistic regression to a neural network can improve accuracy but costs interpretability. For each of these applications, say whether you would accept that trade-off and why: (a) credit scoring, (b) photo tagging, (c) hospital readmission risk, (d) movie recommendations.

---

## 18_2_9: Credit Risk Exercise: Interpretability Constraints

1. The compliance scenario rules out XGBoost before any model is fit. Is this premature? What would you need to know about the regulatory requirement to decide whether a post-hoc explanation tool (like SHAP) could make XGBoost acceptable instead?

2. The exercise one-hot encodes the categorical features with `drop_first=True` for logistic regression, while 18_1 passed them to XGBoost natively. Why does logistic regression need the encoding when XGBoost does not?

3. After fitting, some odds ratios for rare categories have enormous confidence intervals. What does this tell you about how much data supports those coefficients? Should they appear in an adverse action notice?

4. Your forest plot will likely show `checking_status` categories among the strongest predictors, which is consistent with XGBoost's feature importance in 18_1_1. Why is agreement between two very different models reassuring evidence about the data rather than about either model?

5. If XGBoost achieves higher AUC than your logistic regression on this dataset, write the two-or-three sentence recommendation you would give the bank. Which model do you recommend, and what is the deciding factor?
