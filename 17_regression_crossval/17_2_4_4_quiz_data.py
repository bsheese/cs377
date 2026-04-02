quiz_data = [
    {
        "question": "What does the alpha parameter control in Ridge, Lasso, and ElasticNet regression?",
        "options": ["The number of features used in the model.",
                    "The strength of the penalty applied to large coefficients.",
                    "The learning rate used during gradient descent.",
                    "The number of cross-validation folds to use."],
        "answer": "The strength of the penalty applied to large coefficients."
    },
    {
        "question": "What happens when alpha is set too high in a regularized regression model?",
        "options": ["The model overfits by memorizing noise in the training data.",
                    "The model behaves identically to ordinary least squares regression.",
                    "The model underfits because coefficients are penalized too heavily.",
                    "The model requires significantly more training data to converge."],
        "answer": "The model underfits because coefficients are penalized too heavily."
    },
    {
        "question": "In the bias-variance tradeoff plot, why is the training R² always higher than the cross-validation R²?",
        "options": ["Cross-validation uses a smaller dataset, so it always scores lower.",
                    "The training score is measured on the same data the model learned from, while CV score measures generalization to held-out folds.",
                    "The cross-validation folds contain errors that artificially deflate the score.",
                    "Training scores are calculated using a different metric than cross-validation scores."],
        "answer": "The training score is measured on the same data the model learned from, while CV score measures generalization to held-out folds."
    },
    {
        "question": "Why does the notebook wrap the model and StandardScaler together in a Pipeline before using GridSearchCV?",
        "options": ["To reduce the total number of model parameters that need to be estimated.",
                    "To ensure the scaler is fit only on the training fold within each CV split, preventing data leakage from the validation fold.",
                    "To allow GridSearchCV to search over the scaling method as a hyperparameter.",
                    "To make the code run faster by caching intermediate results."],
        "answer": "To ensure the scaler is fit only on the training fold within each CV split, preventing data leakage from the validation fold."
    },
    {
        "question": "What would happen if you scaled the entire dataset using StandardScaler before passing it to cross-validation?",
        "options": ["The model would fail to converge because the features are already normalized.",
                    "Information from the validation fold (its mean and standard deviation) would leak into the training fold.",
                    "Cross-validation would refuse to run because scaled data is not compatible with it.",
                    "The results would be identical because scaling is a linear transformation."],
        "answer": "Information from the validation fold (its mean and standard deviation) would leak into the training fold."
    },
    {
        "question": "What is \"optimistic bias\" in the context of hyperparameter tuning?",
        "options": ["A bias that occurs when the researcher expects a specific result before running the experiment.",
                    "The tendency for models to perform better on training data than on test data.",
                    "The upward bias in a performance estimate when hyperparameters are selected based on their score on the same validation data used for evaluation.",
                    "A mathematical property of the R-squared metric that always overstates model quality."],
        "answer": "The upward bias in a performance estimate when hyperparameters are selected based on their score on the same validation data used for evaluation."
    },
    {
        "question": "In nested cross-validation, what is the role of the inner loop?",
        "options": ["To evaluate the final model performance on held-out data.",
                    "To tune hyperparameters by searching for the best configuration within each outer training fold.",
                    "To split the data into training and test sets before any modeling begins.",
                    "To refit the model on the entire dataset after outer loop evaluation is complete."],
        "answer": "To tune hyperparameters by searching for the best configuration within each outer training fold."
    },
    {
        "question": "In nested cross-validation, what is the role of the outer loop?",
        "options": ["To find the single best hyperparameter value to use for all models.",
                    "To evaluate the tuned model on data that was never involved in any hyperparameter tuning decision.",
                    "To scale the features before the inner loop begins.",
                    "To compare Ridge, Lasso, and ElasticNet against each other."],
        "answer": "To evaluate the tuned model on data that was never involved in any hyperparameter tuning decision."
    },
    {
        "question": "Why might different outer folds in nested CV select different optimal hyperparameters?",
        "options": ["There is a bug in the code that causes inconsistent results.",
                    "Each outer fold trains on a different subset of the data, so the best hyperparameters for one subset may differ from another.",
                    "The inner loop uses random initialization, which causes different results each time.",
                    "Nested CV deliberately forces different hyperparameters to test model robustness."],
        "answer": "Each outer fold trains on a different subset of the data, so the best hyperparameters for one subset may differ from another."
    },
    {
        "question": "How does nested cross-validation differ from the standard train/test split approach used earlier in the notebook?",
        "options": ["Nested CV uses less data because it requires both inner and outer loops.",
                    "Nested CV separates hyperparameter tuning (inner loop) from performance evaluation (outer loop), providing an unbiased estimate without needing a separate test set.",
                    "Nested CV is faster because it only trains the model once per fold.",
                    "Nested CV can only be used with regularized models, not with ordinary least squares."],
        "answer": "Nested CV separates hyperparameter tuning (inner loop) from performance evaluation (outer loop), providing an unbiased estimate without needing a separate test set."
    },
    {
        "question": "When would a simple train/test split be preferable over nested cross-validation?",
        "options": ["When you need the most reliable performance estimate possible.",
                    "When you have a very large dataset and need to iterate quickly on model development.",
                    "When you are reporting final results for publication.",
                    "When your dataset is small and you cannot afford to waste data on a held-out test set."],
        "answer": "When you have a very large dataset and need to iterate quickly on model development."
    },
    {
        "question": "In a learning curve, what does it indicate when the training score is high but the validation score is much lower?",
        "options": ["The model is underfitting and too simple for the data.",
                    "The model is overfitting — it has memorized the training data but fails to generalize.",
                    "The dataset is too large for the model to learn effectively.",
                    "The validation set contains corrupted or incorrectly labeled data."],
        "answer": "The model is overfitting — it has memorized the training data but fails to generalize."
    },
    {
        "question": "In a learning curve, what does it mean if both the training and validation scores converge at a low value?",
        "options": ["The model is overfitting and needs regularization.",
                    "More training data will likely fix the problem.",
                    "The model is underfitting — it is too simple to capture the underlying patterns in the data.",
                    "The model is well-calibrated and performing optimally."],
        "answer": "The model is underfitting — it is too simple to capture the underlying patterns in the data."
    },
    {
        "question": "What does it mean if a learning curve shows the validation score steadily increasing and has not yet plateaued?",
        "options": ["The model has already reached its maximum potential and adding more data will not help.",
                    "The model is still learning and would likely benefit from additional training data.",
                    "The model is overfitting and should be simplified.",
                    "The learning rate is too high and the model is diverging."],
        "answer": "The model is still learning and would likely benefit from additional training data."
    },
    {
        "question": "Why does the notebook describe nested CV scores as typically being lower than scores from the simpler train/test split with GridSearchCV?",
        "options": ["Nested CV uses fewer training samples because of the additional folds.",
                    "Nested CV accounts for the uncertainty in hyperparameter selection, producing a more honest and less optimistic performance estimate.",
                    "Nested CV uses a different scoring metric that is inherently more conservative.",
                    "Nested CV forces the use of suboptimal hyperparameters to test robustness."],
        "answer": "Nested CV accounts for the uncertainty in hyperparameter selection, producing a more honest and less optimistic performance estimate."
    },
]
