# Hypothesis Testing - CaseStudies (Chi-Square, A/B Test, GeoTest Counterfactual)
 
**Chi-Square Independence Test · A/B Proportion Test · GeoTest (Counterfactual Regression)**

## About
This notebook covers three applied statistical inference case studies, each framed as a hypothesis testing problem.  
The goal is to implement core tests **from first principles**, interpret p-values correctly, and understand methodological limits (sample size, confounding, temporal structure).

## Contents

### Part I — Chi-Square Test for Independence (Contingency Tables)
We test whether the distribution of students across specialties depends on their region of origin.

- **Null hypothesis**: specialization is independent of region  
- **Statistic**:
    $$\chi^2 = \sum_{i,j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}}$$
- Expected counts computed via:
    $$E_{ij}=\frac{(\text{row}_i)(\text{col}_j)}{\text{total}}$$
- Manual p-value computation and validation with `scipy.stats.chi2_contingency`.

### Part II — A/B Test on Conversion Rates (Two-Proportion z-test)
We simulate an A/B test to evaluate whether a price increase impacts subscription probability.

- **Null hypothesis**: $p_A = p_B$  
- Pooled estimate under $H_0$:
    $$\hat{p} = \frac{s_A+s_B}{n_A+n_B}$$
- z-statistic:
    $$Z=\frac{\hat{p}_A-\hat{p}_B}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_A}+\frac{1}{n_B}\right)}}$$
- Implementation using `statsmodels.stats.proportion.proportions_ztest`.

### Part III — GeoTest (Causal Impact via Counterfactual Regression)
A bank stops ads in one region (Est) and wants to measure the causal impact on subscriptions while controlling for overall market dynamics.

1. **Naive approach**: post-period comparison (t-test) vs other regions → criticized for ignoring pre-trends/time structure.  
2. **Counterfactual approach**: build a predictive model of Est using other regions (pre-period only), then forecast post-period outcomes:
     $$Y_t = x_t^\top \theta + \varepsilon_t,\quad \varepsilon_t\sim\mathcal{N}(0,\sigma^2)$$
     - Fit on $t \le 24$, predict $t>24$.
     - Compare observed vs counterfactual with:
         - 95% confidence bands around $\hat{Y}_t$
         - pointwise p-values (per month)
         - global tests on the post-period deviation (paired t-test / chi-square aggregation)

## Core takeaways
A p-value answers: **"How surprising is the observed statistic under $H_0$?"**, not "probability $H_0$ is true". Independence tests (chi-square) require sufficient expected counts; small samples weaken validity. A/B tests on proportions rely on CLT approximations; effect size + sample size determine power. GeoTests highlight why **time structure and confounding** matter: post-only comparisons can be misleading; counterfactual modeling is a stronger causal strategy.


