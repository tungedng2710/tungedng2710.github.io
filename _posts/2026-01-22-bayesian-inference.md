---
title: "Bayesian Inference: Priors, Likelihoods, and Decisions"
layout: post
post-image: "https://towardsdatascience.com/wp-content/uploads/2020/12/1eWB1B9MCjG7ZNxXumpsblw.png"
description: A practical tour of Bayesian inference that explains Bayes' rule, conjugate priors, approximate methods, and a worked Beta-Binomial example so you can bring belief updates into real projects.
tags:
- Bayesian Statistics
- Machine Learning
- Probability
author-name: Tung Nguyen
author-url: https://github.com/tungedng2710
lang: en
---

# Why Bayesian inference matters
Deterministic pipelines often fall apart when the data distribution shifts or the amount of evidence changes. Bayesian inference keeps a full probability distribution over uncertain quantities, so you can update beliefs as new observations arrive and keep downstream decisions calibrated. The basic recipe is simple: encode what you already believe (the prior), define how data is generated (the likelihood), then combine the two through Bayes' rule to obtain the posterior.

## Bayes' rule refresher
In its most common form, Bayes' rule relates the posterior distribution of a parameter $\theta$ after observing data $\mathcal{D}$:

<p style="text-align: center;">
\[
 p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\,p(\theta)}{p(\mathcal{D})} \propto p(\mathcal{D} \mid \theta)\,p(\theta).
\]
</p>

- **Prior** $p(\theta)$: beliefs before seeing data.
- **Likelihood** $p(\mathcal{D} \mid \theta)$: data-generating model.
- **Evidence** $p(\mathcal{D})$: normalizing constant, usually an integral or sum.
- **Posterior** $p(\theta \mid \mathcal{D})$: updated beliefs that drive predictions and decisions.

The proportional form is common because the evidence is constant with respect to $\theta$. Normalization is handled explicitly or via sampling algorithms.

## Anatomy of a Bayesian model
1. **Structural assumptions**: pick a distribution family (Bernoulli, Poisson, Gaussian process, etc.) that matches the measurement process.
2. **Parameterization**: define latent variables and hyperparameters that encode unknown relationships.
3. **Prior selection**: choose informative priors when domain knowledge exists or diffuse priors to stay agnostic. Sensitivity analysis is crucial--shift hyperparameters and verify posterior stability.
4. **Posterior computation**: run analytical updates, deterministic approximations, or fully stochastic samplers.
5. **Posterior predictive distribution**: integrate over $\theta$ to make predictions and quantify uncertainty for new inputs.

## Conjugate priors in practice
Conjugate priors produce posteriors in the same family, yielding closed-form updates and minimal compute:

| Likelihood | Conjugate prior | Posterior parameters |
| --- | --- | --- |
| Bernoulli/Binomial with rate $p$ | $\text{Beta}(\alpha, \beta)$ | $\text{Beta}(\alpha + k, \beta + n - k)$ |
| Poisson with rate $\lambda$ | $\text{Gamma}(a, b)$ | $\text{Gamma}(a + \sum x_i, b + n)$ |
| Gaussian with known variance | Gaussian prior | Posterior mean-variance update via precision addition |

Conjugacy is ideal for dashboards, streaming analytics, or embedded systems where you need microsecond updates without full inference pipelines.

## Worked example: Beta-Binomial updating
Suppose you are monitoring a click-through rate. You start with a neutral Beta prior:

- Prior: $p(p) = \text{Beta}(1, 1)$ (uniform between 0 and 1)
- Observations: $n = 40$ impressions, $k = 14$ clicks

The posterior parameters are $\alpha' = 1 + 14 = 15$ and $\beta' = 1 + 40 - 14 = 27$, so

<p style="text-align: center;">
\[
 p(p \mid \mathcal{D}) = \text{Beta}(15, 27).
\]
</p>

From this posterior we can compute:

- Posterior mean: $15/(15 + 27) \approx 0.357$, slightly lower than the empirical rate because the prior added pseudo-counts.
- 95% credible interval: evaluate the 2.5% and 97.5% quantiles of the $\text{Beta}(15, 27)$ distribution (approx. $[0.23, 0.49]$).
- Posterior predictive for the next impression: $p(\text{click}) = \frac{15}{15 + 27} \approx 0.357$.

## Beyond closed forms: approximate Bayesian inference
Most real models--hierarchical regressions, Bayesian neural nets, state-space models--lack conjugacy. Common approximation strategies:

- **Laplace approximation**: fit a Gaussian around the maximum a posteriori (MAP) estimate using the Hessian of the log-posterior.
- **Variational inference (VI)**: posit a tractable family $q_\phi(\theta)$ and minimize KL divergence $\mathrm{KL}(q_\phi \Vert p)$. Stochastic VI with mini-batches scales to millions of data points.
- **Markov Chain Monte Carlo (MCMC)**: draw samples whose stationary distribution equals the posterior. Hamiltonian Monte Carlo and its dynamic variant NUTS remain strong defaults for moderately sized models.
- **Sequential Monte Carlo / particle filters**: update a set of weighted samples online, useful for tracking problems and streaming observations.

In production, it is common to combine these: variational inference for initialization followed by short MCMC chains to capture tail behavior.

## Bayesian workflow checklist
1. **Model critique**: simulate synthetic data from the prior predictive distribution. Does it resemble plausible observations?
2. **Inference diagnostics**: monitor effective sample size, $\hat{R}$, gradient norms, or ELBO convergence depending on the method.
3. **Posterior predictive checks**: draw $y^{(rep)}$ and compare summary statistics or discrepancy measures against the real data.
4. **Decision analysis**: integrate utility/loss functions over the posterior. Decisions change when the posterior crosses predefined risk thresholds.
5. **Communication**: summarize posterior distributions with medians + credible intervals, not just MAP, and visualize how priors influenced the outcome.

## When to reach for Bayesian inference
Bayesian methods shine whenever uncertainty matters:

- **Data scarcity**: priors stabilize estimates when only a few observations are available.
- **Hierarchical structure**: pooling across related groups improves estimates for low-signal segments.
- **Sequential decision-making**: real-time belief updates are perfect for adaptive experiments, robotics, and anomaly detection.
- **Regulatory settings**: credible intervals and explicit priors make audit trails transparent.

If computation is a concern, start with conjugate models or VI, then scale up to richer samplers as the application proves valuable.

## Further reading
- "Bayesian Data Analysis" by Gelman et al. for workflows and model checking.
- "Probabilistic Machine Learning" by Murphy for modern VI + MCMC tooling.
- Stan, NumPyro, PyMC, and Bean Machine communities for high-quality examples and diagnostics.
