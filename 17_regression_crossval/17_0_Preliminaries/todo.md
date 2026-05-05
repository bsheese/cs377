


To replace the "mechanics-first" approach with a "concept-first" foundation, this curriculum design splits the five foundational topics into a cohesive sequence of five Jupyter Notebooks. 

This sequence strictly avoids the `scikit-learn` library. It relies entirely on `numpy`, `pandas`, and `matplotlib`/`seaborn`. Furthermore, we will explicitly avoid the deprecated Boston Housing dataset, replacing it with modern, ethically sound datasets like the **Palmer Penguins** and the **Ames Housing Dataset**.

Here is the detailed pedagogical plan for the prerequisite notebook series.

---

### Notebook 0.1: The Baseline of Variance (Univariate Spread)
**Core Concept:** You cannot understand "variance explained" ($R^2$) if you do not know how to measure variance in the first place.
**Dataset:** Palmer Penguins (Focusing solely on `body_mass_g`).

*   **Section 1: The Simplest Model (The Mean)**
    *   Introduce the mean not just as an average, but as a *baseline predictive model*. If you had to guess the weight of a new penguin without any other information, the mean is your best guess.
*   **Section 2: Measuring the Error of the Mean**
    *   Calculate the distance of every penguin's weight from the mean ($y_i - \bar{y}$). 
    *   *Interactive element:* Plot the data points on a 1D horizontal line and draw lines connecting each point to the mean.
*   **Section 3: Total Sum of Squares (TSS)**
    *   Show why summing the raw distances equals zero (positives cancel negatives).
    *   Introduce squaring the distances to penalize larger deviations.
    *   *Coding task:* Have students write a pure Python function to calculate the **TSS**.
*   **Section 4: Variance and Standard Deviation**
    *   Average the TSS (Variance) and take the square root (Standard Deviation) to bring the metric back to the original units (grams).

### Notebook 0.2: The Visual Imperative and The Danger of Summaries
**Core Concept:** Summary statistics blind us to the actual geometry of the data. Always look at the data before applying math.
**Dataset:** Anscombe’s Quartet & The Datasaurus Dozen.

*   **Section 1: Bivariate Introductions**
    *   Introduce scatterplots. Move from 1D (Notebook 1) to 2D space.
*   **Section 2: Anscombe’s Trap**
    *   Provide four datasets (Anscombe's Quartet). Have students calculate the mean of X, mean of Y, and variance for all four. They will see the stats are nearly identical.
    *   *The Reveal:* Have students plot all four datasets. They will see one is a perfect line, one is a parabola, one has a massive outlier, and one is a vertical line with a high-leverage point.
*   **Section 3: Leverage vs. Outliers**
    *   Conceptually define an outlier (extreme Y value) vs. a high-leverage point (extreme X value). Show visually how a high-leverage point can act like a magnet, pulling any future mathematical model toward it.

### Notebook 0.3: Mathematical Co-movement (Covariance to Correlation)
**Core Concept:** How do we mathematically quantify the visual relationships we just plotted?
**Dataset:** Palmer Penguins (`flipper_length_mm` vs. `body_mass_g`).

*   **Section 1: The Quadrant Method**
    *   Plot the scatterplot and draw a vertical line at the mean of X, and a horizontal line at the mean of Y. This creates four quadrants.
    *   Show how points in the top-right and bottom-left mathematically produce *positive* products when multiplying their deviations: $(x_i - \bar{x}) \times (y_i - \bar{y})$.
*   **Section 2: Covariance**
    *   Sum these products and average them. This is Covariance. 
    *   *The Problem:* Explain that covariance is unbounded. A covariance of 4,000 sounds high, but depends entirely on the units used (grams vs. kilograms).
*   **Section 3: Pearson’s $r$ (Correlation)**
    *   *The Solution:* Standardize covariance by dividing it by the product of the standard deviations of X and Y.
    *   *Coding task:* Students code the Pearson $r$ formula from scratch using the standard deviation formulas they learned in Notebook 1. They prove to themselves that the result is bounded between -1 and 1.

### Notebook 0.4: The Anatomy of Error (Residuals)
**Core Concept:** A model is only as good as its mistakes. We must learn to measure those mistakes.
**Dataset:** A tiny, synthetic dataset (e.g., 6 data points of `Study_Hours` vs. `Exam_Score`) to keep math highly transparent.

*   **Section 1: Drawing an Arbitrary Line**
    *   Give students a basic slope-intercept function: $\hat{y} = mx + b$.
    *   Have them pick *random* values for $m$ and $b$ and plot the resulting line over the scatterplot. (It will likely be a terrible fit).
*   **Section 2: Defining the Residual**
    *   Define the residual mathematically: $e_i = y_i - \hat{y}_i$ (Actual minus Predicted).
    *   *Visual task:* Use `matplotlib.pyplot.vlines` to draw the vertical drop from the actual data points to the arbitrary line. 
*   **Section 3: The Danger of "Total Error"**
    *   Show that simply summing the residuals ($e_i$) is a flawed metric for model evaluation because a terrible line that passes straight through the middle of the data might have a sum of zero.

### Notebook 0.5: The Objective Function and True $R^2$
**Core Concept:** We want the line that minimizes squared mistakes.
**Dataset:** Ames Housing Dataset (Reduced: `Gr_Liv_Area` vs `SalePrice`).

*   **Section 1: Residual Sum of Squares (RSS)**
    *   Solve the Notebook 4 problem by squaring the residuals before summing them. Introduce **RSS** as our ultimate Objective Function.
    *   *Interactive element:* Have students run a `for` loop testing 100 different slopes ($m$) for the housing data, calculating the RSS for each. Have them plot the RSS values to visually find the "bottom of the bowl" (the minimum error).
*   **Section 2: The True Definition of $R^2$**
    *   Bring back the **TSS** from Notebook 1. Remind students: TSS is the error of the *mean* (the worst-case baseline).
    *   Bring in the **RSS**. This is the error of our *line*.
    *   Define $R^2$ as: $1 - \frac{RSS}{TSS}$.
    *   *Pedagogical victory:* Students now perfectly understand that $R^2$ literally means: *"How much of the original variance (TSS) did our line eliminate (RSS)?"*
*   **Section 3: The Bridge to Simple Linear Regression**
    *   Reveal the "cheat codes." Instead of using a `for` loop to guess the best slope, they can use exact algebra.
    *   Show them that the optimal slope is simply: $m = r \times (\frac{S_y}{S_x})$. 
    *   Because they have already mastered correlation ($r$) and standard deviation ($S_y, S_x$), the OLS formula requires zero memorization—it is just the logical combination of everything they have learned over the last five notebooks.

---
**Next Step:** *Now* the curriculum can safely move to `17.1_SLR_with_SciKit-Learn.ipynb`. When the students type `model.fit()` and `model.score()`, they will know exactly what the machine is doing under the hood.
