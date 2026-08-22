# Abstract

Perturbation prediction benchmarks report increasingly rich metric panels, but standard evaluations aggregate performance at the nominal perturbation level. We show that this aggregation hides substantial variation in realized intervention strength within individual perturbations. In X-Atlas/Orion, depth-corrected sgRNA abundance tracks target-gene suppression with a median within-perturbation Spearman correlation of −0.40, while estimated knockdown increases from 65% in the weakest dose quartile to 92% in the strongest. Conditioning ground truth on realized dose changes perturbation-specific reproducible signal from 0.153 to 0.226 across two dose bins in perturbations with measurable effects, an increase eighteen times larger than a within-perturbation permutation null, while near-null perturbations remain flat. We further show that per-cell stratification is itself confounded: against a single global control population, stratification by an unrelated sham variable produces more apparent signal than stratification by dose. Cell-state-matched controls remove this failure mode. Finally, we ask whether current zero-shot predictors contain enough perturbation-specific signal for dose conditioning to affect their evaluation. C2S-Scale is nearly insensitive to removal of its perturbation token under our evaluation (Δcosine = 0.0002), while the publicly released zero-shot STATE checkpoint yields Overlap@N = 0.0 across 9,355 HCT116 perturbations. PCA retains measurable perturbation-specific signal and yields [PCA RESULT]. The per-perturbation dose-associated score gap has standard deviation approximately 0.6, implying that resolving a mean effect of approximately 0.06 requires roughly 500 informative perturbations at our prespecified power threshold. These results identify realized perturbation strength as a hidden within-condition axis of perturbation-prediction evaluation and show both how to estimate it without inducing cell-state confounding and when current predictors are too weak for it to be visible through model scores.

# 1. Introduction

Predicting transcriptional responses to genetic perturbations has become a central benchmark for virtual-cell models. Yet two findings increasingly complicate the interpretation of progress on this task. First, sophisticated perturbation predictors do not reliably outperform simple baselines. Second, aggregate benchmarks mix conditions that differ substantially in how much perturbation-specific signal they contain.

Recent work has begun to address the second problem by stratifying perturbations according to properties of the condition itself. Kalygina et al., for example, show that benchmark conclusions change after separating perturbations according to how much response remains beyond simple baseline structure. Related work has shown that conclusions can also depend strongly on dataset curation and baseline choice; simple gene-level and nearest-neighbor predictors can remain surprisingly competitive under common evaluation settings.

These analyses operate at the level of the nominal perturbation. We study a finer axis that varies *within* a perturbation.

CRISPRi does not silence a target equally in every cell. If (d) denotes realized perturbation strength, the pooled response for perturbation (p) is

[
P(Y\mid p)
==========

\int P(Y\mid p,d),P(d\mid p),dd.
]

The benchmark target therefore depends not only on perturbation identity but also on the empirical distribution of realized doses among the cells assigned that label. When transcriptional response varies with dose, pooling weakly and strongly perturbed cells removes this structure before any model metric is computed.

X-Atlas/Orion makes this axis measurable because its auxiliary guide-call data contain per-cell sgRNA abundance, which can be validated against direct target-gene suppression. We use this information to ask whether realized dose changes the ground-truth perturbation response, whether such conditioning can be performed without introducing cell-state confounding, and whether current predictors contain enough perturbation-specific signal for the distinction to matter in model evaluation.

Our contributions are:

1. **Realized dose is a hidden within-condition evaluation axis.** We validate depth-corrected sgRNA abundance as a proxy for knockdown strength and show that reproducible perturbation-specific signal changes substantially across dose bins.

2. **Dose conditioning requires matched controls.** Per-cell stratification alters cell-state composition as well as perturbation strength; naïve comparisons to a global control population can therefore manufacture apparent signal. We give a cell-state-matched construction and validate it with a sham-axis and permutation control.

3. **Current predictors often cannot expose this axis.** We audit C2S-Scale, STATE, and PCA for perturbation-specific signal, evaluate dose-conditioned performance where possible, and quantify the number of informative perturbations required to resolve the effect through model scores.

# 2. Background and Related Work

## 2.1 Perturbation prediction and simple baselines

Recent work has questioned whether increasingly complex perturbation-prediction architectures consistently improve over simple statistical or similarity-based baselines. Sun et al. show that a gene-wise majority-vote predictor can perform strongly on curated categorical perturbation benchmarks despite ignoring perturbation identity entirely, while its behavior changes sharply on the full unfiltered data. They further show that a weighted nearest-neighbor predictor remains competitive with substantially more complex methods in several settings. These results motivate treating baseline strength and dataset construction as part of the evaluation problem rather than as implementation details.

## 2.2 Stratifying benchmark difficulty

Kalygina et al. formalize a complementary concern: aggregate metrics mix perturbations with different amounts of reproducible residual signal and different degrees of baseline saturation. Their analysis stratifies conditions at the perturbation level and shows that benchmark conclusions depend on this stratification. They additionally identify dose as an unevaluated axis because suitable perturbation-strength measurements are generally unavailable.

Our analysis concerns a different level of variation. Baseline saturation is a property of a perturbation; realized knockdown dose varies from cell to cell *within* the perturbation. The two axes are therefore complementary rather than redundant.

## 2.3 Models evaluated

We consider C2S-Scale, STATE, and a PCA baseline. STATE is a population-level generative model and has previously been evaluated in cross-context perturbation-prediction settings; Sun et al., for example, generate single-cell STATE predictions and evaluate differential expression between generated perturbed and control populations. C2S-Scale and STATE also represent the generation of recent large perturbation models that motivate reassessing whether benchmark improvements remain visible after conditioning on realized intervention strength.

# 3. Methods

## 3.1 Dataset and realized-dose proxy

We use X-Atlas/Orion, with HCT116 as the primary analysis context. For each perturbed cell, we obtain sgRNA abundance from the corresponding `*_filtered_guide_calls_per_cell.csv` release.

Raw guide abundance is strongly coupled to sequencing depth. We therefore construct the dose proxy by residualizing sgRNA UMI against total cellular depth within the relevant analysis population. All dose-conditioned analyses use this depth-corrected quantity rather than raw guide counts.

We validate the proxy independently using expression of the directly targeted gene. For each perturbation, we measure the association between residualized sgRNA abundance and target-gene expression and summarize the resulting within-perturbation correlations across perturbations. We additionally report target knockdown across sgRNA-dose quantiles.

Because finer stratification introduces instability and dose-linked technical structure, the primary evaluation uses two dose bins. Quartiles are used only for validating the monotonic relationship between guide abundance and direct target suppression.

## 3.2 Reliability screen

Dose dependence can only be interpreted for perturbations containing reproducible transcriptional signal. We therefore screen perturbations using the NSR (\lor) split-half-reproducibility criterion adopted from Kalygina et al. A perturbation is retained when either its signal-to-noise statistic or its split-half direction reproducibility exceeds the 95th percentile of the corresponding control-split null.

Selection and evaluation cells are disjoint. The reliability criterion, gene selection, and other quantities used to decide whether a perturbation enters the benchmark are computed only on the selection split. All reported model and ground-truth evaluation scores are computed on the held-out evaluation split.

The final HCT116 screen retains [N RETAINED] of [N TOTAL] perturbations. Of the frozen evaluation perturbations, [N FROZEN RETAINED] pass the reliability criterion and contain enough cells in both dose bins for dose-conditioned scoring.

## 3.3 Cell-state-matched controls

Naïvely partitioning perturbed cells by dose also partitions cells by variables correlated with dose. Comparing each bin to one global control population can therefore introduce a transcriptional difference unrelated to perturbation strength.

For each perturbation and dose bin, we construct a matched control population using cellular covariates measured before scoring. Matching is performed over sequencing depth and mitochondrial read fraction [ADD ANY OTHER MATCHING VARIABLES], producing a separate control distribution for each dose-conditioned perturbed population.

The dose-conditioned perturbation effect is therefore estimated as

[
\Delta_{p,b}
============

## E[X_{\mathrm{pert}}\mid p,b,Z]

E[X_{\mathrm{ctrl}}\mid Z\sim Z_{p,b}],
]

where (b) denotes dose bin and (Z_{p,b}) denotes the empirical covariate distribution of the corresponding perturbed cells.

To validate the construction, we use two negative controls. First, dose assignments are permuted within perturbations while retaining the same bin sizes. Second, we replace dose with a sham per-cell axis, mitochondrial read fraction, which has no causal interpretation as perturbation strength. These controls test whether apparent dose dependence can arise from stratification alone.

Split-half estimates use independent denominators so that the two halves do not share a control-derived noise term.

## 3.4 Evaluation

Ground-truth perturbation signal is measured separately in the lower and higher dose bins using the same metric construction and matched-control procedure.

Model evaluation uses the same frozen perturbation and gene definitions wherever the model output supports them. Missing predicted genes follow the benchmark convention used in the reference evaluation framework: an unavailable prediction is assigned zero perturbation delta rather than silently removing that gene from the evaluation universe.

We audit every model for perturbation-specific signal before interpreting dose-conditioned scores. For C2S-Scale, we compare predictions with and without the perturbation token. For STATE, we measure overlap between predicted and observed differentially expressed genes across the HCT116 perturbations supported by the released checkpoint. PCA serves as a simple predictor that demonstrably retains perturbation-specific variation.

The primary dose statistic is the paired within-perturbation score difference

[
G_{m,p}
=======

## S_{m,p,\mathrm{high}}

S_{m,p,\mathrm{low}},
]

rather than a comparison of aggregate model rankings. Power is calculated from the empirical across-perturbation variance of (G_{m,p}).

# 4. Results

## 4.1 Realized dose tracks knockdown and defines a hidden within-condition axis

Depth-corrected sgRNA abundance is strongly associated with direct target suppression. Across perturbations, the median within-perturbation Spearman correlation between sgRNA dose and target-gene expression is −0.40. The direction of association is negative for 75% of evaluated perturbations.

The relationship is also visible in absolute knockdown. Estimated target suppression increases from 65% in the weakest dose quartile to 92% in the strongest. Residualizing guide abundance against sequencing depth strengthens the median association from −0.21 to −0.40, arguing against sequencing depth as the source of the effect.

Thus cells carrying the same nominal perturbation label can represent substantially different realized interventions.

## 4.2 Dose stratification changes ground truth, but naïve stratification manufactures signal

Among perturbations with measurable transcriptional effects, reproducible perturbation-specific signal increases from 0.153 in the lower-dose bin to 0.226 in the higher-dose bin. The corresponding increase is eighteen times larger than the effect obtained under within-perturbation permutation of dose labels. Near-null perturbations remain approximately flat.

The pooled value lies between the two dose-conditioned estimates. It therefore characterizes the particular mixture of weakly and strongly perturbed cells present in the pooled condition rather than either realized-dose regime individually.

However, this result is only recovered reliably after matching the control population to the cells in each bin. Using a single global control baseline, stratification by the sham mitochondrial-fraction axis produces more apparent perturbation-specific signal than stratification by realized dose itself. The effect therefore arises from sorting cell states rather than from biological dose.

Cell-state matching removes this failure mode while preserving the real dose-associated increase in perturbation signal. Per-cell conditioning is consequently not a neutral post-processing step: the appropriate control population changes with the slice being evaluated.

## 4.3 Current models cannot reliably expose the dose axis

Before evaluating dose dependence, we tested whether each predictor produced measurable perturbation-specific signal.

C2S-Scale is nearly insensitive to perturbation information under our zero-shot evaluation. Removing the perturbation token changes prediction cosine similarity by only 0.0002, indicating that the perturbation input contributes little measurable information to the generated response in this setting.

The publicly released zero-shot STATE checkpoint likewise fails our perturbation-specific DE audit, yielding Overlap@N = 0.0 across 9,355 HCT116 perturbations under the present evaluation.

PCA, in contrast, retains measurable perturbation-specific signal and therefore provides the usable test case for dose-conditioned model evaluation. Its score changes from [LOW-DOSE PCA SCORE] in the lower dose bin to [HIGH-DOSE PCA SCORE] in the higher dose bin, corresponding to a mean paired difference of [PCA GAP] ([CI / P-VALUE]).

Regardless of the mean gap, the per-perturbation dose-associated score difference is highly heterogeneous, with standard deviation approximately 0.6. This variability places a strong limit on what can be resolved with the current benchmark. Under our prespecified power calculation, detecting a mean score difference of approximately 0.06 requires roughly 500 informative perturbations.

The biological dose effect is therefore much easier to establish directly from ground truth than through current model scores. In the present evaluation, two models fail before the dose question can meaningfully be asked, while the remaining predictor is limited by the number of informative perturbations.

# 5. Discussion

Our results separate three issues that are usually conflated in perturbation-prediction evaluation: variation in the biological target, bias introduced by conditioning, and limitations of the predictor itself.

First, a nominal perturbation is not a single experimental condition at cellular resolution. Cells assigned the same guide experience different realized knockdown strengths, and these differences translate into measurably different transcriptional responses. The standard pooled target is therefore mixture-dependent. This does not make pooled evaluation mathematically invalid; it means that the quantity being evaluated depends partly on the realized dose composition of the experimental population.

Second, conditioning on this hidden variable is not sufficient by itself. Dose is measured at the cell level, and stratifying cells changes the distribution of other cell-level properties. Our sham-axis experiment shows that a naïve global-control comparison can generate an apparent effect larger than the biological dose effect it is intended to measure. The matched-control construction is therefore part of the estimand, not merely a variance-reduction procedure.

Third, a shift in the biological evaluation target need not be visible through model metrics. A predictor must first contain sufficient perturbation-specific signal. In our zero-shot evaluation, C2S-Scale is nearly insensitive to its perturbation input and STATE produces no measurable DE overlap under our audit. PCA does retain perturbation-specific signal, but the high variance of per-perturbation dose gaps means that substantially more informative perturbations are needed to resolve modest model-level effects.

This distinction matters when interpreting negative model results. Failure to detect a dose dependence in prediction scores does not imply that dose is biologically irrelevant: the corresponding ground-truth effect is directly measurable and strongly exceeds the permutation null. It instead establishes a requirement on future benchmarks. A sufficiently powered dose-aware comparison requires both predictors that respond reliably to perturbation identity and enough perturbations with reproducible transcriptional effects.

The present study has several limitations. Realized dose is inferred from sgRNA abundance rather than experimentally specified, although its relationship with direct target suppression provides an internal validation. The robust analysis supports two dose bins but not a finely resolved dose-response curve. Only perturbations with measurable transcriptional effects can contribute meaningfully to the conditioned analysis. Our primary results are restricted to HCT116, and the model-level comparison is limited by the behavior of the available zero-shot predictors. Finally, the observed score-gap variance implies that approximately 500 informative perturbations are required to resolve effects of the magnitude considered here, exceeding the effective sample size available in our present evaluation.

These limitations define the scope of the claim: we do not attempt to estimate a universal dose-response function or establish superiority of one model architecture over another. We show that realized perturbation strength is a measurable component of the evaluation target, that naïvely conditioning on it introduces its own confound, and that current model benchmarks may lack either the perturbation-specific signal or the statistical power required to reveal it.

# 6. Conclusion

Perturbation-prediction benchmarks increasingly report multiple views of the same nominal perturbation response. We show that this response can itself be a mixture over substantially different realized intervention strengths.

In X-Atlas/Orion, depth-corrected sgRNA abundance predicts direct target suppression, and conditioning on this axis changes reproducible perturbation-specific signal far beyond a within-perturbation permutation null. Resolving the effect requires cell-state-matched controls: naïve per-cell stratification can otherwise create a larger false signal than the dose effect itself.

The central implication is therefore not that existing metrics are inadequate, but that their target is underspecified. A nominal perturbation label can correspond to a distribution of realized intervention strengths, and benchmark performance depends on the composition of that distribution. Dose-aware evaluation requires conditioning on realized perturbation strength, controls appropriate to that conditioning, and predictors with sufficient perturbation-specific signal for the distinction to be observable.

# Appendix

## A. Dose-proxy construction and residualization

Full definition of guide-count fraction, depth residualization, filtering, zero-count handling, and perturbation-level aggregation.

## B. Reliability-screen calibration

Full NSR and split-half-reproducibility definitions, control-split null construction, 95th-percentile thresholds, retained perturbation counts, and comparison with alternative effect-size screens.

## C. Matched-control construction

Matching variables, binning or nearest-neighbor procedure, tolerances, fallback behavior, effective control sample sizes, and diagnostics for covariate balance.

## D. Permutation and sham-axis controls

Dose-label permutation procedure, mitochondrial-fraction sham construction, number of permutations, null distributions, and independent split-half denominators.

## E. Evaluation metrics

Full definitions of logFC correlation, DE overlap, DE direction agreement, effect-size metrics, distributional metrics, weighting, missing-gene handling, and Tech-dup or equivalent upper controls.

## F. Model-specific audit details

C2S perturbation-token ablation, STATE Overlap@N construction, PCA implementation, gene-space alignment, and all model-specific preprocessing.

## G. Additional results

Full per-perturbation distributions, alternative dose thresholds, all-versus-reliability-filtered sensitivity analysis, per-stratum tables, and supplementary model metrics.
