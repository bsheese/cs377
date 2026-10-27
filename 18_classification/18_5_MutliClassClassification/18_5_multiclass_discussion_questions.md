# Discussion Questions: 18_5 Multi-Class Classification

---

## 18_5_1: When Two Classes Are Not Enough

### From Two Classes to Three

1. Everything in this notebook (probabilities, confusion matrix, classification report) also appeared in the binary notebooks of 18_1. For each of the three, state precisely what changes when the number of classes goes from 2 to 3.
2. `predict_proba` returns a row of three numbers for each penguin, and the numbers always sum to 1. Why must they sum to 1? What assumption about the classes does that encode, and can you think of a problem where that assumption would be wrong (hint: can a photo contain both a cat *and* a dog)?
3. One-hot encoding writes Adelie as [1, 0, 0]. Why not just use 0, 1, 2 as the target and treat it as a regression problem? What false relationship between the species would the model then be free to learn?

### Reading the 3×3 Confusion Matrix

4. A 2×2 confusion matrix has two ways to be wrong; a 3×3 matrix has six. Why is knowing *which specific confusion* occurs (Adelie→Gentoo vs. Adelie→Chinstrap) more actionable than knowing the total error count?
5. Suppose the matrix shows Chinstrap and Adelie being confused in both directions, while Gentoo is almost never confused with anything. Based on what you saw in the class-separability plots, what would explain this pattern? What would you do about it?

## 18_5_2: Measuring Performance Class by Class

### Beyond One Number

6. The notebook argues that a single accuracy number is "not enough" for multiclass problems. Construct a concrete example with three classes of sizes 900/80/20 where 92% accuracy hides a completely useless model for one class.
7. Precision and recall are defined for *one class at a time*, treating that class as "positive" and everything else as "negative." Write out, in plain English, precision and recall for the Chinstrap class specifically.
8. Why is F1 the *harmonic* mean of precision and recall rather than the ordinary average? What behavior does the harmonic mean punish that the arithmetic mean would forgive?

### The Averaging Problem

9. Macro averaging treats a 20-row class exactly like a 2,000-row class. Give one scenario where that is exactly right and one where it is misleading.
10. A model's weighted F1 is 0.93 and its macro F1 is 0.71. Without seeing any other output, describe the model's behavior. Which classes are dragging macro down, and how would you confirm it?
11. The notebook calls the macro/weighted gap "a diagnostic." Diagnostic of what, exactly? Once the gap flags a problem, what tool from 18_5_1 tells you *where* the problem lives?
12. In a business report you can present only one of the two averages. For (a) a wildlife camera classifying common species and (b) a hospital triage system where the rare class is the emergency, which average do you present, and why?

## 18_5_3: When One Class Is Rare

### Majority-Class Bias

13. In the fetal-health data, the majority of recordings are Normal. What accuracy does the "always predict Normal" strategy achieve, and why is it a mistake to be satisfied with beating that number without checking per-class recall? Connect this to the accuracy paradox from 18_1.
14. The rare Pathological class is also the most important one to catch. Is this coincidence, or is there a general reason the rare class is so often the high-stakes class in real applications? Give two more examples.

### Sample Weights

15. `compute_sample_weight(class_weight="balanced")` gives rare-class rows larger weights during training. Explain what changes from the model's point of view: what did a missed Pathological case "cost" the model before weighting, and what does it cost after?
16. Sample weighting does not change the data, since no rows are duplicated or deleted. Compare this to oversampling the rare class. What risk of oversampling does weighting avoid?
17. In 18_5_3, adding balanced sample weights raised recall for Suspect (0.70 to 0.86) and Pathological (0.89 to 0.94), but lowered accuracy and also lowered macro F1 (0.87 to 0.84), because many more Normal sessions were flagged. Make the case for each model. Who, concretely, benefits from the balanced model in the fetal-health setting, who bears its costs, and what would you need to know to choose between them?

### Evaluation with Stratified Cross-Validation

18. Ordinary K-fold can hand one fold almost none of the rare class. Walk through what happens to that fold's Pathological recall and why the resulting cross-validation average is untrustworthy. How does stratification fix it?
19. The notebook recommends macro F1 plus the rare class's own recall as the metrics to watch. Why is macro F1 alone not quite enough when one specific class is the one you cannot afford to miss?
20. You are asked to sign off on deploying this model in a clinic. Beyond the metrics in the notebook, name two questions you would insist on answering first (think: threshold choice, cost of false alarms, what happens downstream of a Suspect prediction).
