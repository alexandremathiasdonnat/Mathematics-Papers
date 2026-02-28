# Linear Parametric Classifiers for Binary Classification  
**Perceptron, Logistic Regression, Threshold Analysis, ROC–AUC & Model Selection**

## 1. About

This lab studies **linear parametric classifiers** for binary classification, focusing on both
**hard decision models** and **probabilistic models**.

The objectives of this lab are to:

- Understand the geometry of linear decision boundaries  
- Compare a hard-margin model (Perceptron) with a probabilistic model (Logistic Regression)  
- Analyze the effect of data separability and noise  
- Study the role of decision thresholds and classification metrics  
- Use ROC curves and AUC for threshold-independent evaluation  
- Apply the methodology to both synthetic and real-world datasets  

The work is organized in progressive steps:

Controlled synthetic datasets to study linear separability. Perceptron learning and convergence properties. Logistic regression and probabilistic interpretation. Effect of noise on linear classifiers. Threshold variation, ROC curves and AUC. Model selection on a real medical dataset  

## 2. Learning Problem Setup

We consider a binary classification problem with labels:

$$
y \in \{-1, +1\} \quad \text{(Perceptron)}, \qquad y \in \{0,1\} \quad \text{(Logistic Regression)}
$$

Each observation is a feature vector:

$$
x \in \mathbb{R}^d
$$

We use a **linear model** of the form:

$$
f(x) = w^\top x + b
$$

which defines a linear decision boundary (a line in 2D, a hyperplane in higher dimension).

## 3. Synthetic Data Generation and Linear Separability

Synthetic 2D datasets are generated using `make_blobs` with two different levels of dispersion:

- **Low variance (`cluster_std = 0.8`)**  
    → classes are well separated and almost linearly separable

- **High variance (`cluster_std = 3.5`)**  
    → strong overlap between classes, no perfect linear separator

This controlled setup allows us to explicitly observe how **data geometry** affects model behavior.

## 4. Perceptron Algorithm

### 4.1 Model

The perceptron predicts using a hard decision rule:

$$
\hat{y} = \mathrm{sign}(w^\top x + b)
$$

### 4.2 Learning Rule

For a misclassified example $(x_i, y_i)$, parameters are updated as:

$$
w \leftarrow w + \eta (y_i - \hat{y}_i) x_i, \quad
b \leftarrow b + \eta (y_i - \hat{y}_i)
$$

### 4.3 Observations

- The perceptron **converges only if the data are linearly separable**  
- On overlapping data, it never reaches zero training error  
- The learned boundary is sensitive to noisy or ambiguous points  

This illustrates the **limitations of hard decision classifiers**.

## 5. Logistic Regression

### 5.1 Probabilistic Model

Logistic regression models the conditional probability:

$$
p(x) = P(Y=1 \mid X=x) = \sigma(w^\top x + b)
$$

where $\sigma(\cdot)$ is the sigmoid function.

The decision boundary at threshold $0.5$ coincides with:

$$
w^\top x + b = 0
$$

### 5.2 Advantages

- Outputs calibrated probabilities  
- Remains stable even when data are not linearly separable  
- Allows flexible decision-making via threshold selection  

## 6. Threshold, Precision and Recall

Logistic regression outputs probabilities, which must be converted into class labels using a **decision threshold**.

- **Low threshold** → high recall, more false positives  
- **High threshold** → high precision, more false negatives  

Key metrics:

- **Recall**: proportion of true positives correctly detected  
- **Precision**: proportion of predicted positives that are correct  

This highlights that **classification performance depends on the application context**.

## 7. ROC Curve and AUC

To evaluate models independently of a specific threshold, we use:

- **ROC curve**: plots TPR vs FPR for all possible thresholds  
- **AUC**: area under the ROC curve  

AUC = 1 → perfect classifier  
AUC = 0.5 → random guessing  

ROC–AUC measures the **ranking and discrimination ability** of the model, not a single decision rule.

## 8. Effect of Additive Gaussian Noise

Starting from a separable dataset, Gaussian noise is added to the features.

Observations:

- Increasing noise reduces margin stability  
- The perceptron becomes more sensitive to fluctuations  
- Logistic regression produces smoother and more robust boundaries  

Noise degrades stability but does not necessarily destroy linear separability when its amplitude remains moderate.

## 9. Real Dataset: Breast Cancer Classification

The methodology is applied to the **Breast Cancer dataset** from scikit-learn.

Steps:

- Train/test split  
- Feature standardization  
- Model training and evaluation  

This illustrates how linear classifiers behave on real, noisy, high-dimensional medical data.

## 10. Model Selection with Cross-Validation

To select hyperparameters, we use **K-fold cross-validation**:

- Data are split into K folds  
- Each fold is used once as validation  
- Performance is averaged over folds  

Grid search is used to compare:

- Perceptron with different regularizations  
- Logistic regression with L1 and L2 penalties  

Performance is evaluated using **ROC–AUC**.

## 11. Core takeaways

Linear classifiers define geometric decision boundaries. The perceptron is a hard-margin algorithm with strict assumptions. Logistic regression provides probabilistic outputs and greater robustness. Threshold choice controls the precision–recall trade-off. ROC–AUC enables fair model comparison. Cross-validation is essential for reliable model selection  

## 12. Dependencies

- numpy  
- matplotlib  
- pandas  
- scikit-learn  

---
**Alexandre Mathias DONNAT, Sr**


