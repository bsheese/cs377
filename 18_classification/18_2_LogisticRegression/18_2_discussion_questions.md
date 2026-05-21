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

## 18_2_1_1: Logistic Regression with Titanic

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

16. AUC = 0.836. Explain the probabilistic interpretation of AUC in plain English using a concrete example from the Titanic context.

### Threshold Tuning

17. At t=0.3 (lower threshold), recall increases for survivors but precision decreases. Explain why lowering the threshold produces this tradeoff.

18. A medical screening application needs high recall (catch as many true positives as possible). A fraud detection application might prioritize precision (avoid false alarms). What threshold direction would each context favor?

19. The threshold t=0.4 happens to maximize accuracy. Why might you NOT want to use this threshold for a specific business decision?

---

## 18_2_1_2: Gender-Stratified Titanic Models

### Segmentation and Interaction

1. Training separate models for men and women is one way to capture gender effects. What's the difference between this approach and adding a gender interaction term to a single model?

2. The female model and male model will likely show very different coefficient patterns. If 1st class matters strongly for women but not for men, what does this tell you about the nature of the interaction?

3. What is a risk of training on smaller segmented datasets? How does reducing sample size affect coefficient stability?

4. The baseline survival rate differs between men and women (more women survived). How does this affect the interpretation of precision and recall for each model?

5. Comparing odds ratios side-by-side across gender models can reveal which features are uniquely predictive for each group. Why might a feature that's important for women be irrelevant for men in the Titanic context?

---

## 18_2_9: OpenIntro Exercises (Possums, Challenger, Odds Ratios)

### Why Not Linear Regression (Revisited)

1. The exercise asks whether "logistic regression fits a line." Why is this FALSE? What shape does the fitted function produce?

2. Why are residuals not expected to be normally distributed in logistic regression? What does the outcome variable look like compared to linear regression?

### Multicollinearity in the Possum Model

3. `head_l` (head length) and `skull_w` (skull width) have a correlation of 0.71. This is described as multicollinearity. What does multicollinearity do to individual coefficient estimates and their p-values?

4. When `head_l` is removed from the model, `skull_w` becomes more statistically significant. Explain why removing one of two correlated features makes the other one's p-value smaller.

5. AIC is used to compare the full and reduced possum models. Why is AIC more appropriate here than R² or accuracy?

### The Challenger Disaster

6. Engineers had data from previous launches showing O-ring failures at various temperatures. The accident occurred at 31°F — the coldest launch ever attempted. The logistic model predicts ~99.3% probability of failure at 31°F. Why was this trend hard to see from the raw data?

7. The probability curve for O-ring failure shows probability near zero for warm temperatures and near 1 for cold temperatures. Why does the logistic curve naturally capture this threshold behavior?

8. This is a case where the model's output could have been used to prevent a tragedy. What does this example say about the responsibility of data scientists when their models have safety implications?

### Odds Ratios

9. The spam filter coefficient for the word "Winner" is 1.63. The odds ratio is $e^{1.63} \approx 5.10$. What does this mean in plain English?

10. If a coefficient is negative, the odds ratio will be between 0 and 1. What does an odds ratio of 0.3 tell you about the relationship between that feature and the outcome?

---

## 18_2_9: Red-Tailed Hawks and LOOCV

### Small Sample Challenges

1. The hawk dataset has only 65 birds. Why is Leave-One-Out Cross-Validation (LOOCV) preferred over a simple 75/25 train-test split for a dataset this size?

2. LOOCV trains 65 separate models and averages their scores. What computational tradeoff are you making compared to k-fold CV? When would this tradeoff not be worth making?

3. All juvenile (HY) hawks in the dataset happen to be female. How does this age-sex confound affect the model's ability to separate the effects of age and sex?

### Comparing Models to Domain Knowledge

4. Washburn et al. (2022) derived equations from domain expertise for hawk sex classification. A machine learning model trained on local Illinois data achieves similar performance. What does this suggest about the role of ML vs. domain knowledge?

5. The ML logistic regression uses all features together, while the Washburn equations use different variables for different age classes. What are the tradeoffs of each approach?

6. SVM often leads on this dataset despite being more opaque than logistic regression. When would you choose a more interpretable model (logistic regression) over a potentially more accurate model (SVM)?

### Odds Ratios in Biology

7. The odds ratio visualization for the hawk model uses a log scale. Why is a log scale appropriate for comparing odds ratios both above and below 1?

8. If mass has the highest odds ratio for the "female" class, what does this tell you biologically? Does a high odds ratio mean mass is the single best predictor? What might complicate this interpretation?

9. High multicollinearity exists among the body measurements (all are correlated). How does this affect your interpretation of individual feature odds ratios in the logistic regression?

