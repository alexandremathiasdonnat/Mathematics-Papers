# Time Series Analysis - Practice & Experiments

**Signal Processing, Spectral Analysis & Autoregressive Modeling**


## About

This notebook presents a comprehensive, mathematically grounded study of time series analysis through three real-world datasets drawn from epidemiology, network traffic, and climate science.

The objective is to develop a unified analytical framework combining time-domain modeling, frequency-domain representations, statistical inference and validation, while maintaining a strong focus on interpretation and theoretical consistency rather than black-box prediction.

Classical models are explicitly linked to their mathematical assumptions and spectral properties, highlighting both their explanatory power and their limitations.

## Mathematical Framework

A time series $(X_t)_{t \in \mathbb{Z}}$ is analyzed through two complementary perspectives:

### Time-domain representation

The series is viewed as a stochastic process governed by dependence structures such as:

$$X_t = \sum_{k=1}^{p} \phi_k X_{t-k} + \varepsilon_t, \quad \varepsilon_t \sim \text{WN}(0, \sigma^2),$$

where $\{\phi_k\}$ characterize temporal persistence and $\varepsilon_t$ represents innovations.

### Frequency-domain representation

Using Fourier analysis, temporal dependence is translated into spectral content. The power spectral density (PSD) is defined as the Fourier transform of the autocovariance function:

$$f(\omega) = \sum_{h=-\infty}^{\infty} \gamma(h) e^{-2i\pi\omega h}, \quad \omega \in [-1/2, 1/2].$$

This dual representation allows periodic structures, long-memory effects, and noise components to be analyzed rigorously.

## Structure of the Notebook

The notebook is organized into three parts, each addressing a specific class of time series and modeling challenges.

### Part I - Influenza-like Illness Incidence (IAS)

**Dataset: Daily epidemiological incidence data.**

This part focuses on a non-stationary, heavy-tailed time series dominated by seasonal and episodic behavior.

**Main methodological steps:**

- Construction of a daily time series $(X_t)$ with missing-value imputation
- Logarithmic transformation $Y_t = \log(X_t)$ for variance stabilization
- Estimation of the periodogram as a non-parametric PSD estimator
- Identification of dominant frequencies corresponding to annual and weekly cycles
- Removal of annual seasonality using the periodic differencing operator:

    $$(\Delta_T X)_t = X_t - X_{t-T}, \quad T = 365$$

- Seasonal modeling via harmonic regression:

    $$X_t \approx \sum_{k=1}^{K} \left[ a_k \cos\left(\frac{2\pi k t}{T}\right) + b_k \sin\left(\frac{2\pi k t}{T}\right) \right]$$

- Evaluation of predictive performance and analysis of overfitting as $K$ increases

**Insight:**
Periodic harmonic models approximate the deterministic seasonal component but cannot reproduce stochastic epidemic peaks, which are intrinsically non-periodic.

### Part II - Network Traffic Data (LBL-TCP-3)

**Dataset: High-frequency TCP packet timestamps aggregated into 10-second intervals.**

This part examines short-memory stochastic dynamics typical of network traffic.

**Main methodological steps:**

- Aggregation of event-based data into a regular time series
- Modeling via autoregressive processes $\text{AR}(p)$
- Model order selection using:
    - Akaike Information Criterion (AIC),
    - Bayesian Information Criterion (BIC),
    - expanding-window cross-validation
- Residual diagnostics and departure from Gaussianity
- Spectral validation via comparison between:
    - empirical periodogram,
    - theoretical AR spectral density:

        $$f(\omega) = \frac{\sigma^2}{\left|1 - \sum_{k=1}^{p} \phi_k e^{-2i\pi k\omega}\right|^2}$$

**Insight:**
Traffic dynamics are dominated by short-term dependence. Low-order AR models capture correlation structure but fail to model burst-driven heavy tails.

### Part III - Southern Oscillation Index (SOI)

**Dataset: Monthly climate index related to ENSO dynamics.**

This part focuses on persistent geophysical processes with strong physical interpretation.

**Main methodological steps:**

- Reshaping a wide-format dataset into a monthly time series
- Analysis of dependence using ACF and PACF
- Selection and estimation of a low-order AR model
- Residual diagnostics and Gaussian comparison
- Frequency-domain validation of the fitted model

**Insight:**
An AR(1) model provides a statistically and physically meaningful description of climate persistence, capturing low-frequency variability while smoothing high-frequency noise.

## My methodological perspective

In this notebook I emphasizes: the equivalence between time-domain and frequency-domain descriptions, explicit links between model parameters and spectral behavior, the role of statistical diagnostics in validating assumptions, the limits of linear and Gaussian models for real-world data. Rather than optimizing predictive accuracy alone, the analysis prioritizes interpretability, robustness, and consistency with underlying physical or behavioral mechanisms.

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `statsmodels`

*Dataset aviables upon request*

---
***Alexandre Mathias DONNAT, Sr***
