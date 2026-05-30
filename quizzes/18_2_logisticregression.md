# 18.2 · Logistic Regression

## 18_2_0: Foundations

### Why can't linear regression be used directly for binary classification?
- [ ] Linear regression is too slow to train on typical classification datasets
- [x] Linear regression can predict values below 0 and above 1, which are invalid probabilities
- [ ] Linear regression requires normally distributed targets rather than binary labels
- [ ] Binary classification requires at least three predictor variables to work properly

### In logistic regression, a positive coefficient means the feature increases the log-odds. Is it correct to say it 'increases the probability'?
- [ ] Yes — a positive log-odds value always maps to a probability above 0.5
- [x] Partly — it increases probability, but not by a fixed amount; the effect depends on the current probability level
- [ ] No — positive log-odds corresponds to a decrease in probability for the baseline class
- [ ] Yes — the logistic function is linear, so the probability increase is constant across all values

### Why does logistic regression use log-likelihood (sum of logs) instead of the raw likelihood (product of probabilities)?
- [ ] Log-likelihood is numerically larger than raw likelihood for valid probability values
- [x] The product of many small probabilities underflows to zero; sums of logs are numerically stable
- [ ] Log-likelihood is required in order to use gradient descent for optimization
- [ ] The log function eliminates the need to compute exponentials in the sigmoid

### A logistic regression coefficient β = 0. What is the odds ratio, and what does it mean?
- [ ] OR = 0 — the feature completely eliminates the probability of the outcome
- [x] OR = 1 — the feature has no effect on the odds of the outcome
- [ ] OR = ∞ — the feature perfectly predicts the outcome with certainty
- [ ] OR = e⁰ = 0 — the feature is therefore statistically insignificant

### In what specific sense is logistic regression 'a linear model'?
- [ ] The sigmoid function is linear when the output is plotted on a log scale
- [x] The log-odds (logit) is a linear function of the predictors
- [ ] Logistic regression minimizes a linear loss function during training
- [ ] The decision boundary is always a straight line in any 2D feature space

## 18_2_1: Titanic Workflow

### Fare is log-transformed before fitting the Titanic model. What distribution problem motivates this?
- [ ] Fare has a bimodal distribution that causes problems for gradient descent
- [x] Fare is right-skewed with a few very high values; log compression improves the linear relationship
- [ ] Log-transformation converts fare from a continuous to a categorical variable
- [ ] Fare must be log-transformed to prevent data leakage inside the pipeline

### StandardScaler is placed inside a Pipeline rather than applied before splitting. What data leakage would occur otherwise?
- [ ] The test set would be scaled on different statistics, causing evaluation errors
- [x] The scaler's mean and std would be computed from test data, leaking test information into training
- [ ] The pipeline would fail to process unseen data at deployment time
- [ ] Scaling before the split changes the proportion of the train-test ratio

### The sex (female=1) odds ratio is ~12.28. How would you explain this to a non-statistician?
- [ ] Female passengers were 12 times more likely to survive than male passengers
- [x] Female passengers had 12 times higher odds of surviving compared to male passengers
- [ ] 12.28% of female passengers survived compared to 1% of male passengers
- [ ] The model added 12.28 extra probability points to female passengers

### The pclass_3 odds ratio is ~0.17. What does an odds ratio below 1 mean?
- [ ] 3rd class passengers were predicted to survive 17% of the time
- [x] 3rd class status reduces the odds of survival to 17% of 1st class odds
- [ ] The negative coefficient means 3rd class is incorrectly encoded in the model
- [ ] The model produces unreliable predictions for 3rd class passengers

### 5-fold CV scores are [0.797, 0.782, 0.842, 0.827, 0.744]. The std is 0.069. What does this standard deviation tell you?
- [ ] The model has a 6.9% error rate on average across all predictions
- [x] Performance varies by about 7 percentage points across different data splits
- [ ] 69% of predictions fall within one fold's range of the true value
- [ ] The model is unstable and should be retrained with different hyperparameters

### The model has recall = 0.64 for the survived class. What does this mean in the Titanic context?
- [ ] 64% of the model's survival predictions were correct
- [x] The model correctly identified 64% of the actual survivors
- [ ] 64% of passengers in the test set survived
- [ ] The model made 64 correct predictions on the test set

### AUC = 0.836. What is the correct probabilistic interpretation?
- [ ] The model correctly classifies 83.6% of all passengers in the test set
- [x] Given a random survivor and a random non-survivor, the model ranks the survivor higher 83.6% of the time
- [ ] The model assigns a predicted probability above 0.5 to 83.6% of actual survivors
- [ ] The false positive rate is 16.4% when the threshold is set to 0.836

### Lowering the decision threshold from 0.5 to 0.3 increases recall but decreases precision. Why?
- [ ] A lower threshold means fewer passengers are classified as positive
- [x] A lower threshold catches more true positives but also flags more false positives
- [ ] Precision and recall move in the same direction as the threshold changes
- [ ] A threshold of 0.3 is below the statistical baseline and should not be used

### The model predicts deaths better (recall 0.85) than survivals (recall 0.64). Why might this happen?
- [ ] The model was trained exclusively on passengers who died
- [x] The majority class (died) provides more training examples, so the model is better calibrated for it
- [ ] Survivals are less predictable because they depended on random chance
- [ ] The features encode death status more explicitly than survival status

## 18_2_9: Possums, Challenger & LOOCV

### Logistic regression is called a classification algorithm. Is the statement 'it fits a line' TRUE or FALSE?
- [ ] TRUE — the log-odds equation is linear so the fitted function is a straight line
- [x] FALSE — it fits an S-shaped sigmoid curve, not a straight line, for probability predictions
- [ ] TRUE — the decision boundary is always a straight line in 2D feature space
- [ ] FALSE — logistic regression fits a parabola to model binary outcome probabilities

### In the possum model, head_l and skull_w are correlated at 0.71. When head_l is removed, skull_w becomes more significant. Why?
- [ ] Removing head_l increases the effective sample size available for estimating skull_w
- [x] Both features competed to explain the same variance; removing one lets the other show its full signal
- [ ] skull_w becomes significant because its coefficient changes sign after the removal
- [ ] The AIC decreases only when statistically significant features are removed from the model

### The Challenger logistic model predicts ~99.3% probability of O-ring failure at 31°F. Engineers had data but missed this. Why?
- [ ] The engineers did not have access to the temperature measurements before the launch
- [x] Prior launches were mostly in warm weather where risk was near zero; the cold-temperature trend was invisible without modeling
- [ ] The logistic model was not available in 1986 and had to be derived post-hoc
- [ ] The O-ring failure data was classified and not shared with the engineering team

### A spam filter coefficient for 'Winner' is 1.63. The odds ratio is exp(1.63) ≈ 5.10. What does this mean?
- [ ] Emails containing 'Winner' have a 5.10% probability of being spam
- [x] Emails containing 'Winner' have 5.1 times higher odds of being spam
- [ ] The word 'Winner' appears 5.1 times more often in spam than in non-spam emails
- [ ] 'Winner' increases the predicted spam probability by 5.10 percentage points

### The hawk dataset has 65 birds. Why is Leave-One-Out Cross-Validation (LOOCV) preferred over a 75/25 split?
- [ ] LOOCV trains 65 models, which is computationally more efficient than a single split
- [x] A 75/25 split would leave only ~16 test birds; LOOCV uses every bird as the test case exactly once
- [ ] LOOCV automatically accounts for the age-sex confound in the hawk data
- [ ] 75/25 splits are only valid for datasets with more than 500 observations

### The hawk SVM model often outperforms logistic regression. Under what circumstance would you choose logistic regression anyway?
- [ ] When the dataset has more than 1,000 observations
- [x] When interpretable coefficients and odds ratios are needed for biological or legal explanation
- [ ] When the model must achieve above 95% recall on the minority class
- [ ] When all features are continuous and no categorical encoding is needed
