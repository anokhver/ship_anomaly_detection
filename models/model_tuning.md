# Model tuning description and reasoning

After creating the model training script, the models needed to be properly adjusted to produce the most informative results. 

We selected F1-score, as well as its components - Precision and Recall to focus on during the tuning, since they model the ratios of false positive and negative results, which is crucial in anomaly detection.

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 score=2⋅(Precision+Recall)/(Precision⋅Recall)
​


Different models required different ways of tuning and testing, depending on their behaviour.

## Random forest

### Model description
A Random Forest is an ensemble of many decision trees, where each tree is trained on a random subset of the data. Each tree is also trained on a random subset of features at each split. The final prediction is made by averaging or in our case, classification as either anomaly or non-anomaly.

### Important parameters
n_estimators - number of trees in the forest (typical values 100 and higher)

max_depth - maximum depth of each tree, lower means less expressive with risk of underfitting, higher means more complex rules with risk of overfitting (typical values 10, 20, 30, or None)

min_samples_leaf - minimum number of samples required at a leaf node (typical values 1, 5, 10)

max_features - number of features to consider when looking for best split (sqrt, log2, float, int)

### Testing
The testing was desined to verify how well the model classifies the anomalies and non-anomalies to appropriate categories. A small sample of both was removed from the training set and used to create a test set. After the training, the model is made to classify the entries. By comparing the labels given by the model to the ones created during manual labeling, we can determine the ratios of false positives and negatives, and determine the F1-score.

### Tuning and final values
The tuning consisted of training the model multiple times with a grid of different parameters. The combination of parameters with the highest F1-score was selected for each training route and the scoring threshold was set dynamically to ensure lowest amount of false results. In this model, we also tried using StratifiedKFold with 5 splits for additional cross-validation but the effects were marginal.


Final values of parameters are as follows(with τ being a scoring threshold):

KIEL-GDYNIA route - n_estimators=100, max_depth=20, min_samples_leaf=1,max_features=sqrt, τ = 0.39 with anomaly score F1=0.992


BREMERHAVEN-HAMBURG route - n_estimators=200, max_depth=None, min_samples_leaf=1, max_features=sqrt, τ=0.42 with anomaly score F1=0.972

### Observations and potential improvements


## Logistic regression

### Model description
Logistic Regression is a linear model used for binary classification (and multiclass via extensions).
It models the probability that an input belongs to class 1 using the sigmoid function:

$$
P(y = 1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}
$$

Where:
- $\mathbf{w}$ is the weight vector,
- $\mathbf{x}$ is the input feature vector,
- $b$ is the bias (intercept term)

### Important parameters
C -  controls the amount of regularization applied to the model. Regularization prevents overfitting by penalizing overly complex models. (typical values: 0.01, 0.1, 1, 10, 100)

penalty - type of regularization used for example l2 which penalizes large weights or l1 which pushes some weights to exact zero, useful when expecting sparse features.

solver - algorithm used to fit the model, not all solvers support all types of penalties (example algorithms: liblinear, lbfgs, saga)

class_weight - balances importance of each class (numerical values or "balanced")

### Testing
Since logistic regression is also a classifier, just like with random forest, a small sample of anomalies and non-anomalies was removed from the training set and used to create a test set. The F1 score is calculated based on the results of the model scoring a test set of labeled entries.

### Tuning and final values
The tuning consisted of training the model multiple times with a grid of different parameters. The combination of parameters with the highest F1-score was selected for each training route and the scoring threshold was set dynamically to ensure lowest amount of false results.

Final values of parameters are as follows(with τ being a scoring threshold):

KIEL-GDYNIA route - C=1.0, penalty : l1, class_weight: None, solver : liblinear, τ=0.32 with anomaly score F1=0.859

BREMERHAVEN-HAMBURG route - C=0.01, penalty : l2, class_weight : {0: 1.0, 1: 4.0}, solver : lbfgs, τ=0.54 with anomaly score F1=0.760

### Observations and potential improvements



## Isolation forest

### Model descripiton
Isolation Forest is an unsupervised algorithm designed for anomaly detection. Unlike density-based methods, it works by isolating anomalies rather than profiling normal points. Anomalies are easier to isolate than normal points. The algorithm builds many random decision trees (called isolation trees). At each node, it randomly selects a feature and a split value. The number of splits (i.e., tree depth) needed to isolate a point is a measure of its "normality". Fewer splits → more anomalous.

### Important parameters

n_estimators - Number of isolation trees to build. More trees usually means more stable and accurate anomaly scores at the cost of increased training time. (typical values 100 and higher)

max_samples - Number (or fraction) of samples to draw for each tree. Smaller means more randomness, larger means more accurate but slower (0.5, 1, "auto")

max_features - Number (or fraction) of features to consider when splitting. (0.6, 0.8, 1)

contamination - Expected proportion of anomalies in the data (values data dependent)

### Testing
Since isolation forest only trains on the non-anomalous data, the test set could be designed in a different way then previous two models. All labeled anomalies were included in the test set, together with the fraction of non-anomalies removed from the training set. The calculation of F1-score was done the same way as in previous models, however this time, with more positives to evaluate.

### Tuning and final values
The tuning once again consisted of training the model multiple times with different parameters. The version with the best score was selected. The only exception was the value of contamination parameter, which was manually determined based on the ratio of anomalies found by us during manual labeling, as well as trial and error when searching for the best F1-score.

Final values of parameters are as follows(with τ being a scoring threshold):

KIEL-GDYNIA route - n_estimators=140, τ=-0.070, max_samples=0.75, max_features=1.0 with anomaly score F1=0.884

BREMERHAVEN-HAMBURG route - n_estimators=140, τ=-0.038, max_samples=1.0, max_features=0.8 with anomaly score F1=0.875

### Observations and potential improvements

## One-class SVM

### Model descripiton
One-Class SVM is an unsupervised anomaly detection algorithm. It is designed to model the “normal” class only, and identify observations that deviate from it as anomalies. It finds a decision boundary (a hyperplane or curved surface) in the feature space that encloses most of the training data.Points outside this boundary are labeled as anomalies.

### Important paramerers

nu (v) - controls the upper bound on the fraction of anomalies and the lower bound on the fraction of support vectors. Acts like a regularization parameter. (value range from 0 to 1, typically from 0.01 to 0.1)

gamma - controls the influence of a single training sample. (typically gamma ∈ [1e-4, 1e-3, 1e-2, 0.1, 1.0])

### Testing

Just like with the isolation forest, the model trained only on non-anomalous data, so all the labeled anomalies were put in the testing set, along with small fraction of non-anomalies. The F1-score calculation was performed the same way as in previously mentioned models.

### Tuning and final values
The combination of parameters resulting in the best F1-score was again selected. The scoring threshold was also selected dynamically, to label as little false positives and negatives in the test set as possible.

Final values of parameters are as follows (with τ being a scoring threshold):

KIEL-GDYNIA route - ν=0.03, τ=0.123, γ(gamma)=1.0 with anomaly score F1=0.941

BREMERHAVEN-HAMBURG route - ν=0.001, τ=0.000, γ(gamma)=1.0 with anomaly score F1=0.900

### Observations and potential improvements

