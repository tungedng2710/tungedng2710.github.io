---
title: "Positive–Unlabeled Learning: A Short Survey"
pubDate: 2026-10-09
image: "/assets/images/heroimage.png"
description: A practical survey of positive–unlabeled learning, covering selection assumptions, probability correction, non-negative risk estimation, graph methods, and evaluation for spam-phone detection.
tags:
- Machine Learning
- Positive-Unlabeled Learning
- Graph Learning
- Spam Detection
authorName: Tung Nguyen
authorUrl: https://github.com/tungedng2710
lang: en
---

A phone number with several confirmed spam reports is a useful positive example. A number with no reports is harder to interpret: it might be benign, or it might be spam that nobody has reported yet. Treating every unreported number as negative turns missing information into a potentially incorrect label.

**Positive–unlabeled (PU) learning** addresses this setting. It trains a binary classifier from confirmed positives and an unlabeled population that contains both positives and negatives.

This article surveys selected foundational approaches, explains the non-negative PU objective, and discusses how graph structure changes the problem. A running example is DIAL, a spam-phone detection framework in which a trusted set of reported spam numbers supplies the initial positive labels. The evaluation ideas below are proposed experiments, rather than measured results.

## 1. Problem and assumptions

Let $V$ be the population of phone numbers and $S$ the trusted spam set. In the DIAL example, the labeled positives are $P=S$ and the remaining phones are $U=V\setminus S$. The task is to find additional spam numbers without assuming that every member of $U$ is benign. For a broad introduction to PU learning, see Bekker and Davis's [2020 survey](https://doi.org/10.1007/s10994-020-05877-5).

Let $Y\in\{0,1\}$ denote the true class and $L=1$ indicate an observed positive label. Standard PU learning assumes $P(L=1\mid Y=0,X)=0$: observed positives are correct. This is a clean-positive assumption: labeled examples may be incomplete, but they are not false positives. The selection mechanism then determines what can be learned:

- **SCAR — selected completely at random:** $P(L=1\mid Y=1,X)=c$. Labeled positives represent the positive population. Elkan and Noto show that $P(L=1\mid X)=cP(Y=1\mid X)$, enabling probability correction when $c$ can be estimated. [Elkan and Noto, 2008](https://cseweb.ucsd.edu/~elkan/posonly.pdf).
- **SAR — selected at random:** labeling propensity $e(X)=P(L=1\mid Y=1,X)$ varies with observed features. Bekker and Davis learn under this mechanism using additional assumptions about propensity attributes. Class probability and propensity are not generally identifiable from their product alone. [Bekker and Davis, 2018](https://proceedings.mlr.press/v94/bekker18a.html).

In the spam-phone example, report exposure, caller volume, and consensus thresholds plausibly favor particular spam phones. SCAR should therefore be tested as a baseline assumption. SAR requires justified propensity features or additional information; naming the mechanism does not resolve selection bias.

## 2. Main approaches

PU methods differ in how they handle the unknown labels in $U$. Some select likely negatives, some correct predicted probabilities, and others estimate a training objective directly from positive and mixture data.

| Approach | Main idea and representative reference | Strength and limitation |
|---|---|---|
| Two-step methods | Identify reliable negatives within $U$, then train a supervised classifier. See [Bekker and Davis, 2020](https://doi.org/10.1007/s10994-020-05877-5). | Simple and interpretable; mistaken negative selection can hide atypical positives. |
| Probability correction | Train a labeled-versus-unlabeled classifier and correct its probabilities under SCAR. [Elkan and Noto, 2008](https://cseweb.ucsd.edu/~elkan/posonly.pdf). | Useful baseline; depends on representative positives and estimating $c$. |
| Bagging PU | Repeatedly train on positives versus random unlabeled subsets, then aggregate predictions. [Mordelet and Vert, 2014, *A bagging SVM to learn from positive and unlabeled examples*](https://doi.org/10.1016/j.patrec.2013.06.010). | Practical ranking baseline; bagging alone does not establish calibrated spam probabilities. |
| Risk estimation | Rewrite supervised risk using positive and mixture data. [du Plessis, Niu, and Sugiyama, 2015, *Convex Formulation for Learning from Positive and Unlabeled Data*](https://proceedings.mlr.press/v37/plessis15.html). | Principled objective; requires an appropriate positive class prior and sampling assumptions. |
| Non-negative PU (nnPU) | Prevent the estimated negative-class risk from becoming negative. [Kiryo et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html). | Controls a source of overfitting in flexible models; does not correct biased or erroneous positive labels. |
| Propensity-aware PU | Model feature-dependent positive selection. [Bekker and Davis, 2018](https://proceedings.mlr.press/v94/bekker18a.html). | Addresses observed selection bias under identifying assumptions; unobserved reporting factors remain difficult. |

### The nnPU objective

For a score function $f$, loss $\ell$, and target positive prevalence $\pi=P(Y=1)$, supervised risk can be rewritten as

$$
R(f)=\pi\,\mathbb{E}_{X\mid Y=1}[\ell(f(X),1)]
+\mathbb{E}_{X}[\ell(f(X),0)]
-\pi\,\mathbb{E}_{X\mid Y=1}[\ell(f(X),0)].
$$

Let $\widehat R_P^+$ denote the average positive-label loss on positive examples, $\widehat R_P^-$ the average negative-label loss on those same examples, and $\widehat R_M^-$ the average negative-label loss on a sample from the target marginal distribution. The empirical non-negative PU objective is

$$
\widehat R_{\mathrm{nnPU}}(f)
=\pi\widehat R_P^+
+\max\{0,\widehat R_M^- -\pi\widehat R_P^-\}.
$$

Here $M$ samples the target marginal distribution, while $P$ samples the positive conditional distribution. The clamp avoids fitting an impossible negative risk; it introduces bias but improves robustness to overfitting. [Kiryo et al., 2017, *Positive-Unlabeled Learning with Non-Negative Risk Estimator*](https://proceedings.neurips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html).

**Sampling matters.** In the DIAL example, the residual set $V\setminus S$ is conditioned on being unselected. It is not automatically the target marginal $M$. Use the full training population for the marginal expectation, or explicitly adjust for the selection mechanism and corresponding prevalence.

The prior $\pi$ is also not the observed labeled fraction. Ramaswamy, Scott, and Tewari estimate mixture proportions through kernel embeddings with convergence guarantees under distributional assumptions. Their work motivates checking identifiability and sweeping plausible priors instead of treating a prevalence estimate as ground truth. [ICML 2016, *Mixture Proportion Estimation via Kernel Embeddings of Distributions*](https://proceedings.mlr.press/v48/ramaswamy16.html).

## 3. Graph PU learning

A GNN can provide $f(X,A)$ while a PU objective supplies supervision. However, graph neighbors need not share labels. Wu et al. show how heterophilic edges can damage class-prior estimation and latent-label inference, and introduce GPL, which combines label propagation loss with graph refinement through bilevel optimization. [ICML 2024, *Unraveling the Impact of Heterophilic Structures on Graph Positive-Unlabeled Learning*](https://proceedings.mlr.press/v235/wu24ad.html).

In the DIAL example, spam callers can connect to benign recipients, and report edges link reporters to reported phones. Edge direction and node roles matter. Compare feature-only PU with graph PU before assuming that neighborhood smoothing helps.

## 4. Evaluating a spam-phone PU system

For the DIAL example, an evaluation should test both the classifier and the assumptions used to expand the initial spam set. A practical experimental plan is:

1. Compare a naive positive-versus-unlabeled baseline, bagging PU, feature-only nnPU, and GNN + nnPU; consider propensity-aware learning after specifying credible selection assumptions.
2. Sweep seed thresholds, labeled-positive fractions, class priors, and random seeds. Retain disagreement information because standard PU assumes clean positives.
3. Build features, reports, and graph edges using a training-time cutoff; evaluate on independently adjudicated future phones. Report PR-AUC, recall at a fixed precision, calibration, and the number of added spam predictions. Treating unlabeled test phones as negative would invalidate ordinary supervised interpretation of these metrics.
4. Expand $S$ using validated high-confidence predictions while retaining confidence and label provenance. Preserve an uncertain group; train any downstream classifier on pseudo-labels with appropriate weighting, and assess it against independent truth rather than agreement with its own teacher.

## 5. Recommended reading

For a path through the literature, start with the general survey, then move from probability correction to risk estimation and selection bias before exploring graphs:

1. [Bekker and Davis (2020): A survey on positive and unlabeled learning](https://doi.org/10.1007/s10994-020-05877-5).
2. [Elkan and Noto (2008): Learning classifiers from only positive and unlabeled data](https://cseweb.ucsd.edu/~elkan/posonly.pdf).
3. [Kiryo et al. (2017): Positive-Unlabeled Learning with Non-Negative Risk Estimator](https://proceedings.neurips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html).
4. [Bekker and Davis (2018): Learning from Positive and Unlabeled Data under the Selected At Random Assumption](https://proceedings.mlr.press/v94/bekker18a.html).
5. [Wu et al. (2024): Unraveling the Impact of Heterophilic Structures on Graph Positive-Unlabeled Learning](https://proceedings.mlr.press/v235/wu24ad.html).
