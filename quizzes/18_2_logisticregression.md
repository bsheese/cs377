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

### AUC = 0.842. What is the correct probabilistic interpretation?
- [ ] The model correctly classifies 84.2% of all passengers
- [x] Given a random survivor and a random non-survivor, the model ranks the survivor higher 84.2% of the time
- [ ] The model assigns a predicted probability above 0.5 to 84.2% of actual survivors
- [ ] The false positive rate is 15.8% when threshold is set to 0.842

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

## 18_2_2: Interpretability

### Sklearn's LogisticRegression and statsmodels' Logit fit the same model. What does statsmodels provide that sklearn does not?
- [x] Standard errors, p-values, and confidence intervals on the coefficients
- [ ] Faster training through gradient descent
- [ ] Pipeline integration and cross-validation support
- [ ] Automatic one-hot encoding of categorical features

### On a forest plot, a feature's 95% confidence interval crosses OR = 1. What does this mean?
- [x] The feature's effect is not statistically distinguishable from no effect
- [ ] The feature increases the odds of the outcome
- [ ] The feature should be removed from the model immediately
- [ ] The model is poorly calibrated for that feature

### Why is OR = 1 (rather than 0) the 'no effect' reference line for odds ratios?
- [x] Because multiplying the odds by 1 leaves them unchanged — the coefficient is 0 and e^0 = 1
- [ ] Because odds ratios cannot be smaller than 1
- [ ] Because probabilities are centered at 1
- [ ] Because statsmodels normalizes all odds ratios to start at 1

### In the Titanic model, sibspouse and parentchild have wide confidence intervals that cross OR = 1. What is the careful interpretation?
- [x] The data are not informative enough to be confident about these effects — they could be sizable or near zero
- [ ] Family size has been proven to have no effect on survival
- [ ] The features were encoded incorrectly
- [ ] These features should have been scaled before fitting

### XGBoost's gain-based feature importance and LR's odds ratios both rank sex_f as the top feature. What can the odds ratio tell you that the importance score cannot?
- [x] The direction and magnitude of the effect, plus the uncertainty around the estimate
- [ ] Which feature the model used most often for splits
- [ ] Whether the model is overfitting
- [ ] The feature's correlation with the other predictors

### How does the individual-prediction decomposition for logistic regression work?
- [x] Each feature's contribution to the log-odds is its coefficient times its (scaled) value; summing them with the intercept reproduces the prediction
- [ ] SHAP values are computed for each feature using a game-theoretic approximation
- [ ] The model is refit once per feature with that feature removed
- [ ] The predicted probability is split equally among the significant features

## 18_2_3: The Bridge to Neural Networks

### The notebook verifies that sigmoid(W·x + b) reproduces sklearn's predict_proba exactly. What does this demonstrate?
- [x] Logistic regression is structurally identical to a single artificial neuron
- [ ] Sklearn uses statsmodels internally
- [ ] Neural networks always outperform logistic regression
- [ ] The sigmoid function is only an approximation of the true model

### Logistic regression scores exactly 50% on the XOR problem. Why?
- [x] XOR is not linearly separable, and a single neuron can only draw a straight-line decision boundary
- [ ] XOR has too few samples for the model to converge
- [ ] The learning rate was set too high
- [ ] The XOR labels are random, so 50% is the best any model can do

### Why can a network with one hidden layer solve XOR when logistic regression cannot?
- [x] The hidden layer transforms the inputs into a new space where the classes become linearly separable
- [ ] The hidden layer memorizes all four training points
- [ ] Neural networks use a different loss function that handles non-separable data
- [ ] The hidden layer increases the learning rate adaptively

### A colleague says: 'Use sigmoid in all hidden layers so the outputs are probabilities.' What is wrong with this reasoning?
- [x] Hidden layers don't need probabilistic outputs, and deep sigmoid networks suffer from vanishing gradients — ReLU avoids this
- [ ] Sigmoid is computationally impossible to use in hidden layers
- [ ] Probabilities can only be produced by the input layer
- [ ] Sigmoid outputs are unbounded, so they cannot be probabilities

### The shallow MLP (145 parameters) performs about the same as logistic regression (8 parameters) on the Titanic test set. Why?
- [x] The dominant signals in the data are captured by a linear boundary, so extra expressive power adds nothing
- [ ] The MLP was undertrained
- [ ] MLPs cannot handle binary features
- [ ] The MLP's hidden layer was too small to learn anything

### What is the 'interpretability tax' of neural networks?
- [x] Their expressive power comes from learned transformations distributed across many weights that have no human-readable meaning
- [ ] Training them requires expensive specialized hardware
- [ ] They require more training data than other models
- [ ] Their predictions cannot be converted to probabilities
