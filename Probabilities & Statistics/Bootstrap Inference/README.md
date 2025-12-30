# Bootstrap Inference for Regression Coefficients

## Objective  
This notebook studies the **variability and uncertainty** of the slope coefficient $\beta_1$ in a **simple linear regression model** using **bootstrap resampling**.  
The goal is to estimate **confidence intervals** without relying on analytical assumptions, and to compare two main approaches:
- **Normal-approximation CI**
- **Percentile CI**

![alt text](figure.png)

---
The bootstrap is a resampling method that estimates the sampling distribution of a statistic by repeatedly drawing samples with replacement from the observed data; it is more robust than the jackknife because it captures both bias and variance more accurately, especially in nonlinear estimators or small samples, where the jackknife’s leave-one-out approach tends to underestimate variability. 

## Mathematical Framework  
We consider the linear model:
$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad \varepsilon_i \sim \text{i.i.d.}(0, \sigma^2)
$$

The OLS estimators are:
$$
\hat{\beta}_1 = 
\frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2},
\quad
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}.
$$

To assess the sampling variability of $\hat{\beta}_1$, we apply the **bootstrap principle**:
1. Resample $(x_i, y_i)$ pairs with replacement.  
2. Recompute $\hat{\beta}_1^*$ for each bootstrap sample.  
3. Repeat $B$ times (e.g. $B = 2000$) to approximate the sampling distribution of $\hat{\beta}_1$.

---

## Inference  
From the empirical distribution of $\hat{\beta}_1^*$:
- **Normal CI:**  
    $\hat{\beta}_1 \pm z_{1-\alpha/2} \cdot \text{SE}_{boot}$
- **Percentile CI:**  
    empirical quantiles of $\hat{\beta}_1^*$.

These intervals provide **non-parametric inference** and capture possible **asymmetry or skewness** in the sampling distribution.

---

## Main Takeaway  
Bootstrap inference allows direct estimation of the **uncertainty of regression coefficients** without analytical variance formulas.  
It highlights the robustness of $\hat{\beta}_1$ and the shape of its empirical distribution.

