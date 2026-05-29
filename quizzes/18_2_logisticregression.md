# 18.2 · Logistic Regression

## 18_2_0: Foundations

### Why can't linear regression be used directly for binary classification?
- [ ] Linear regression is too slow to train on classification datasets
- [x] Linear regression can predict probabilities below 0 and above 1, which are undefined
- [ ] Linear regression requires normally distributed targets, not binary ones
- [ ] Binary classification requires at least three predictors to work properly

### In logistic regression, a positive coefficient means the feature increases the log-odds. Is it correct to say it 'increases the probability'?
- [ ] Yes — a positive log-odds always produces a probability above 0.5
- [x] Partly — it increases probability, but not by a fixed amount; the effect depends on the current probability
- [ ] No — positive log-odds corresponds to a decrease in probability for the baseline class
- [ ] Yes — the logistic function is linear, so the probability increase is constant

### Why does logistic regression use log-likelihood (sum of logs) instead of the raw likelihood (product of probabilities)?
- [ ] Log-likelihood is always higher than raw likelihood for valid probability values
- [x] The product of many small probabilities underflows to zero; sums of logs are numerically stable
- [ ] Log-likelihood is required to use gradient descent optimization
- [ ] The log function removes the need to compute exponentials in the sigmoid

### A logistic regression coefficient β = 0. What is the odds ratio, and what does it mean?
- [ ] OR = 0 — the feature completely eliminates the probability of the outcome
- [x] OR = 1 — the feature has no effect on the odds of the outcome
- [ ] OR = ∞ — the feature perfectly predicts the outcome
- [ ] OR = e⁰ = 0 — the feature is statistically insignificant

### In what specific sense is logistic regression 'a linear model'?
- [ ] The sigmoid function is linear when plotted on a log scale
- [x] The log-odds (logit) is a linear function of the predictors
- [ ] Logistic regression minimizes a linear loss function
- [ ] The decision boundary is always a straight line in 2D feature space

## 18_2_1: Titanic Workflow

### Fare is log-transformed before fitting the Titanic model. What distribution problem motivates this?
- [ ] Fare has a bimodal distribution that confuses gradient descent
- [x] Fare is right-skewed with a few extremely high values; log compression improves the linear relationship
- [ ] Log-transformation converts fare from continuous to categorical
- [ ] Fare must be log-transformed to prevent data leakage in the pipeline

### StandardScaler is placed inside a Pipeline rather than applied before splitting. What data leakage would occur otherwise?
- [ ] The test set would be scaled differently, causing evaluation errors
- [x] The scaler's mean and std would be computed from test data, leaking test information into training
- [ ] The pipeline would be unable to process unseen data after deployment
- [ ] Scaling before splitting changes the train-test ratio

### The sex (female=1) odds ratio is ~12.28. How would you explain this to a non-statistician?
- [ ] Female passengers were 12 times more likely to survive than male passengers
- [x] Female passengers had 12 times higher odds of surviving compared to male passengers
- [ ] 12.28% of female passengers survived compared to 1% of males
- [ ] The model assigned 12.28 extra probability points to female passengers

### The pclass_3 odds ratio is ~0.17. What does an odds ratio below 1 mean?
- [ ] 3rd class passengers were predicted to survive 17% of the time
- [x] 3rd class status reduces the odds of survival to 17% of 1st class odds
- [ ] The coefficient is negative, meaning 3rd class is poorly encoded
- [ ] The model is unreliable for 3rd class passengers

### 5-fold CV scores are [0.797, 0.782, 0.842, 0.827, 0.744]. The std is 0.069. What does this standard deviation tell you?
- [ ] The model has a 6.9% error rate on average
- [x] Performance varies by about 7 percentage points across different data splits
- [ ] 69% of predictions are within one fold of the true value
- [ ] The model is unstable and should be retrained with different hyperparameters

### The model has recall = 0.64 for the survived class. What does this mean in the Titanic context?
- [ ] 64% of the model's survival predictions were correct
- [x] The model correctly identified 64% of the actual survivors
- [ ] 64% of passengers in the test set survived
- [ ] The model made 64 correct predictions on the test set

### AUC = 0.836. What is the correct probabilistic interpretation?
- [ ] The model correctly classifies 83.6% of all passengers
- [x] Given a random survivor and a random non-survivor, the model ranks the survivor higher 83.6% of the time
- [ ] The model assigns a predicted probability above 0.5 to 83.6% of actual survivors
- [ ] The false positive rate is 16.4% when threshold is set to 0.836

### Lowering the decision threshold from 0.5 to 0.3 increases recall but decreases precision. Why?
- [ ] Lower threshold means fewer patients are classified as positive
- [x] Lower threshold catches more true positives but also more false positives
- [ ] Precision and recall always move in the same direction
- [ ] A threshold of 0.3 is below the baseline and should not be used

### The model predicts deaths better (recall 0.85) than survivals (recall 0.64). Why might this happen?
- [ ] The model was trained only on passengers who died
- [x] The majority class (died) provides more training examples, making the model better calibrated for it
- [ ] Survivals are less predictable because they depend on luck
- [ ] The features encode death more explicitly than survival

## 18_2_9: Possums, Challenger & LOOCV

### Logistic regression is called a classification algorithm. Is the statement 'it fits a line' TRUE or FALSE?
- [ ] TRUE — the log-odds equation is linear so the fitted function is a line
- [x] FALSE — it fits an S-shaped curve, not a straight line, for probability predictions
- [ ] TRUE — the decision boundary is always a line in 2D space
- [ ] FALSE — logistic regression fits a parabola for binary outcomes

### In the possum model, head_l and skull_w are correlated at 0.71. When head_l is removed, skull_w becomes more significant. Why?
- [ ] Removing head_l increases the sample size available for skull_w
- [x] Both features competed to explain the same variance; removing one allows the other to show its full signal
- [ ] skull_w becomes significant because its coefficient changes sign
- [ ] The AIC decreases only when significant features are removed

### The Challenger logistic model predicts ~99.3% probability of O-ring failure at 31°F. Engineers had data but missed this. Why?
- [ ] The engineers did not have access to the temperature data before launch
- [x] Most prior launches were in warm weather where risk was near zero; the cold-temperature trend was hard to see without modeling
- [ ] The logistic model was unavailable in 1986 and had to be derived post-hoc
- [ ] The O-ring data was classified and not shared with the engineering team

### A spam filter coefficient for 'Winner' is 1.63. The odds ratio is exp(1.63) ≈ 5.10. What does this mean?
- [ ] Emails with 'Winner' have a 5.10% probability of being spam
- [x] Emails containing 'Winner' have 5.1 times higher odds of being spam
- [ ] The word 'Winner' appears 5.1 times more often in spam than non-spam
- [ ] 'Winner' increases the spam probability by 5.10 percentage points

### The hawk dataset has 65 birds. Why is Leave-One-Out Cross-Validation (LOOCV) preferred over a 75/25 split?
- [ ] LOOCV trains 65 models, which is computationally more efficient than one split
- [x] A 75/25 split would leave only ~16 test birds; LOOCV uses every bird as test exactly once
- [ ] LOOCV automatically handles the age-sex confound in the data
- [ ] 75/25 splits are only valid for datasets with more than 500 samples

### The hawk SVM model often outperforms logistic regression. Under what circumstance would you choose logistic regression anyway?
- [ ] When the dataset has more than 1,000 observations
- [x] When interpretable coefficients and odds ratios are needed for biological or legal explanation
- [ ] When the model must achieve above 95% recall on the minority class
- [ ] When the features are all continuous and no categorical encoding is needed
