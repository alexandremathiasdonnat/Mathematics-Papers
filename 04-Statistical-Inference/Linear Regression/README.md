# Introduction to Linear Regression - Mathematical Foundations
**Minimizing OLS criterion, Confidence Intervals, Statistic test**
## About 

This lab builds a full linear-regression pipeline on the classic **Advertising** dataset:

- **Explore** the data (distributions, scatterplots, correlations)
- **Fit** simple and multiple linear regression models
- **Interpret** coefficients in real units (e.g. "+$1000 TV" → ΔSales)
- **Test** statistical significance (t-test / p-values)
- **Quantify uncertainty** (confidence intervals vs prediction intervals)
- **Select models** using **train/test split** and **cross-validation**

## Dataset

**Advertising.csv** contains **N = 200 markets** with:

- Features (regressors):  
    - `TV`, `Radio`, `Newspaper` = advertising spend (in **thousands of dollars**)
- Response:  
    - `Sales` = product sales (in **thousands of units**)

So one "unit" in `TV` means **$1000**, and one "unit" in `Sales` means **1000 units**.

## Mathematical model

### 1) Simple Linear Regression
We model one predictor (e.g. TV):

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i
$$

- $\beta_0$: intercept (baseline sales when $x=0$)
- $\beta_1$: slope (marginal effect of $x$ on sales)
- $\varepsilon_i$: noise (unexplained part)

**Least Squares** chooses $\hat\beta$ to minimize:

$$
\min_{\beta_0,\beta_1}\ \sum_{i=1}^N (y_i - (\beta_0 + \beta_1 x_i))^2
$$

### 2) Multiple Linear Regression
With three predictors:

$$
y_i = \beta_0 + \beta_1\,TV_i + \beta_2\,Radio_i + \beta_3\,Newspaper_i + \varepsilon_i
$$

Matrix form:

$$
Y = \Phi \beta + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0,\sigma^2 I_N)
$$

- $\Phi$ is the $N \times (d+1)$ design matrix (first column = ones)
- $d=3$ here (TV/Radio/Newspaper)

OLS solution:

$$
\hat\beta = (\Phi^\top \Phi)^{-1}\Phi^\top Y
$$

## Inference: t-tests and confidence intervals

### Testing a coefficient $\beta_k$
Null hypothesis (no linear effect):

$$
H_0:\ \beta_k = 0
\quad\text{vs}\quad
H_1:\ \beta_k \neq 0
$$

Test statistic:

$$
t = \frac{\hat\beta_k}{\widehat{\mathrm{SE}}(\hat\beta_k)}
\quad\sim\quad t_{N-d-1}\ \text{under }H_0
$$

If p-value < 0.05 → reject $H_0$ (statistically significant).

### Confidence interval for $\beta_k$ (95%)
$$
\hat\beta_k \pm q\ \widehat{\mathrm{SE}}(\hat\beta_k)
$$

where:

$$
q = t_{1-\alpha/2,\ N-d-1}
\quad\text{(Student critical value)}
$$

and:

$$
\hat\sigma^2 = \frac{\|Y-\hat Y\|^2}{N-d-1}
$$

## Confidence interval vs prediction interval

For a new input $x_0$:

- **Confidence interval**: uncertainty on the **mean** prediction $E[Y|x_0]$  
    (narrower)
- **Prediction interval**: uncertainty for a **new observation** $Y_0$  
    (wider, because it includes irreducible noise)

Both get wider at extreme $x_0$ (far from the mean of the data).

## Learnings

- **"Significant" ≠ "useful"**: a variable can have p < 0.05 but tiny effect / low R².
- In multiple regression, a feature can become **non-significant** due to **collinearity**.
- Linear regression is not "truth"; it is a **best linear approximation** under assumptions
    (linearity, homoscedasticity, independence, ~Gaussian residuals for inference).

## Content

- EDA: histograms, boxplots, scatterplots, correlation heatmap
- Simple regression: fitted line + interpretation of $\hat\beta_0,\hat\beta_1$
- Hypothesis tests for each feature (TV/Radio/Newspaper)
- Confidence vs prediction intervals plot + explanation
- Multiple regression coefficients + practical interpretation
- Model selection using train/test split + CV logic (generalization focus)


