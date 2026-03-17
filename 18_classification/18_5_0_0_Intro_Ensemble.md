# Ensemble Methods

- **First!** Watch this [StatQuest video on Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&t=143s)

Ensemble methods combine multiple trees to create stronger, more accurate models. While the basic concept of combining models is powerful, each method has specific techniques that enhance performance in different ways.

---

## The Building Block: Decision Trees

Before we dive into ensembles, we must understand the individual trees that make them up. Think of a decision tree as a structured way of making a choice—much like deciding whether to spend your evening watching a StatQuest video or scrolling through your favorite bit of "brain rot." 

A decision tree takes a complex problem and breaks it down into a series of simple, binary questions. Depending on your answer (True or False), you are guided down a specific path—exactly like a flowchart—until you reach a final result. This outcome is either a category (for a **Classification Tree**) or a specific number (for a **Regression Tree**).

### How Trees Grow: Recursive Binary Splitting
While they look like simple flowcharts, trees are built using a mathematical process called **Recursive Binary Splitting**. 
- At every step, the tree looks at all your features and finds the single best place to "split" the data to reduce error. 
- For **Regression**, we want to minimize the **Residual Sum of Squares (RSS)**.
- For **Classification**, we use metrics like the **Gini Index** or **Entropy** to maximize "node purity"—essentially trying to group similar items together.

Trees are incredibly flexible. They can handle both categorical data (e.g., "Do you like popcorn?") and numerical data (e.g., "How many hours do you exercise?"). They can even ask about the same feature multiple times at different levels—for example, looking at "Age < 18" and then later "Age < 65"—to capture very intricate patterns.

### The "Tree Paradox": Interpretable but Unstable
The greatest strength of a single tree is its **interpretability**. Of all the models we study, tree results are the easiest to explain to a non-technical audience. You can literally print the flowchart and follow it with a pen.

However, trees have two major weaknesses:
1. **Overfitting:** Because they are so flexible, they often grow too deep and "memorize" the training data rather than learning general trends. We can try to fix this with **Cost-Complexity Pruning** (using a penalty parameter **$\lambda$**) or by limiting the **Tree Depth**, but a single tree often still struggles with accuracy.
2. **Instability:** Trees are highly sensitive. A tiny change in your data can result in a completely different tree structure. This "High Variance" is exactly why we move to **Ensemble Methods**.

> **Practitioner's Tip:** I almost always report the results of a Random Forest or Boosting model because they are more accurate. However, I **always** run and visualize a single decision tree first. It’s the fastest way to get a "feel" for your data and see which features are doing the heavy lifting.

---

## Bagging (Bootstrap Aggregating)

*Quick summary: **Bagging** reduces the instability of single trees by averaging many of them together. If one tree is sensitive to noise, the average of 500 trees is remarkably stable.*

### Introduction to Bagging
On average, a bootstrap sample contains about **63.2%** of the original unique instances. Bagging works through:

1. **Bootstrap Sampling**: We create multiple versions of our dataset by sampling with replacement. Some rows are repeated; others are left out.
2. **Building Deep Trees**: We train a separate tree on each sample. We let these trees grow deep and complex (**High Variance / Low Bias**).
3. **Aggregation**: We average the results (for regression) or take a majority vote (for classification).

**Why Bias Doesn't Increase:**
Bagging doesn't make the model "simpler" or "dumber." Each tree is still a complex learner. By averaging them, we simply smooth out the "noise" and "eccentricities" of individual trees, achieving a massive reduction in variance without sacrificing the ability to learn complex patterns.

---

## Out-of-Bag Error Estimation

Out-of-Bag (OOB) error estimation is a clever way to measure accuracy without needing a separate test set.

- **Out-of-Bag Instances**: For every tree we build, roughly one-third of the data was not used to train it.
- **OOB Prediction**: We can treat those "left-out" rows as a mini-test set for that specific tree.
- **Average OOB Error**: By averaging these errors across the whole forest, we get an unbiased estimate of how the model will perform on brand-new data.

> **Pro-Tip: Why "Out-of-Bag"?**
> Think of the original dataset as a "bag" of samples. Since we sample with replacement to create training sets, some samples never make it into a specific tree's training set. They are literally "out of the bag" for that tree!

---

## Random Forests

*Quick Summary: **Random Forests** are "Bagging plus randomness." They prevent one very strong feature from dominating every tree, leading to a more diverse and "decorrelated" ensemble.*

Random Forests add one crucial step to the bagging process:
1. **Random Feature Selection**: At every single split in every tree, the model is only allowed to look at a **random subset ($m$)** of the available features.
2. **Decorrelation**: If you have one massive predictor (like "Income" in a spending model), basic bagging would put that split at the top of every single tree. This makes the trees too similar. Random Forest forces some trees to ignore "Income" and look at other features instead, uncovering hidden patterns and improving exploration.

---

## Boosting

*Quick summary: **Boosting** builds trees sequentially. Instead of a "democracy" of independent trees, it’s like a team of experts where each person tries to fix the specific mistakes made by the person before them.*

- ***Number of Trees ($B$ / `n_estimators`):*** Adding more trees can lead to overfitting if not controlled.
- ***Shrinkage ($\lambda$ / `learning_rate`):*** Controls the contribution of each new tree. Smaller values slow down learning, potentially improving accuracy.
- ***Tree Depth ($d$ / `max_depth`):*** Sets the complexity of each tree (usually very shallow trees, or "stumps," work best).

> **Bias-Variance Cheat Sheet**
> - **Bagging:** Reduces **Variance**. Use it when your model is too complex and **overfitting**.
> - **Boosting:** Reduces **Bias**. Use it when your model is too simple and **underfitting**.

### Two Main Flavors of Boosting:
1. **AdaBoost (Adaptive Boosting):** Focuses on "hard" cases. It gives more weight to rows that were misclassified by the previous tree.
2. **Gradient Boosting:** Fits the next tree to the **residuals** (the leftover errors). It literally tries to predict the error of the previous trees to cancel it out. This is the logic behind powerful tools like XGBoost and LightGBM.

> **Pro-Tip: Can Boosting Overfit?**
> Yes! Unlike Random Forests, which are generally safe as you add more trees, Boosting can eventually start "modeling the noise" if you add too many trees ($B$). Always use cross-validation to find the optimal $B$.

---

## Bayesian Additive Regression Trees (BART)

***Quick Summary: BART** is the "Bayesian" version of ensemble learning. Instead of building brand-new trees, it "perturbates" (slightly tweaks) the trees from the previous step, allowing for a very flexible model that handles uncertainty exceptionally well.*

BART's unique strength is its ability to quantify uncertainty through **credible intervals**. It doesn't just give you a prediction; it tells you how confident it is. It also requires very little "tuning" compared to Boosting or Random Forests, often working brilliantly right out of the box.

---

## Summary of Tree Ensemble Methods

| Method | Main Goal | Resampling | Tree Growth | Key Feature |
|--------|-----------|------------|-------------|-------------|
| **Single Tree** | Interpretability | None | Top-down | Very easy to visualize |
| **Bagging** | Reduce Variance | Bootstrapped | Independent | Parallel training |
| **Random Forests** | Reduce Variance | Bootstrapped | Independent | Random feature subsets |
| **Boosting** | Reduce Bias | None | Sequential | Fits to residuals/errors |
| **BART** | Reduce both | None | Sequential | Bayesian tree perturbation |
