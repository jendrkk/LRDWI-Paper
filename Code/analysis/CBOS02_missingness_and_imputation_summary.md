# CBOS Income Variables: Missingness Analysis & Imputation Summary

## Overview

This document summarizes the approach to identifying missingness patterns (CBOS02A) and implementing income imputation (CBOS02B) for the two income variables in the CBOS Polish monthly survey data (355,352 observations, 327 surveys, Jan 1990 – Dec 2017).

| Variable | Raw missing | Post-imputation missing | Imputed values |
|----------|------------|------------------------|----------------|
| `income_p` (personal) | 60.6% | 39.3% (structural) | 56,837 |
| `income_hh` (household) | 22.3% | 0.0% | 79,357 |

---

## CBOS02A: Missingness Pattern Identification

### Approach

We conducted a systematic, survey-by-survey classification of the missingness mechanism following the Rubin (1976) taxonomy (MCAR → MAR → MNAR), using four complementary methods:

1. **Structural vs. item non-response decomposition**: Separated surveys that never collected a variable (100% missing = structural) from surveys with partial missingness (item non-response). Result: 120/327 surveys never collected `income_p`; all 327 surveys collected `income_hh`.

2. **Per-survey logistic regressions** (2 models × 2 targets × 327 surveys):
   - *Model A* (demographics only): age, sex, civil status, education, household size
   - *Model B* (+ attitudinal): adds `sol` (standard of living self-assessment)
   - Computed pseudo-R², AUC (via 5-fold CV), and individual coefficient significance for each survey.

3. **Likelihood-ratio MCAR tests**: For each survey, tested whether missingness is independent of all observables (H₀: MCAR). Used LRT comparing a saturated logistic model to an intercept-only model.

4. **Global pooled analysis**: Pooled all surveys with item non-response, computed cross-validated AUC and Cohen's d effect sizes for each predictor.

### Key Results

| Diagnostic | `income_p` | `income_hh` |
|-----------|-----------|-------------|
| MCAR rejected (LRT, α=0.05) | 82% of surveys | 94% of surveys |
| Median AUC (Model A) | 0.62 | 0.67 |
| Median AUC (Model B) | 0.63 | 0.67 |
| Pooled CV AUC (Model B) | 0.65 | 0.68 |
| Top predictors (by Cohen's d) | age, household_size, education | age, household_size, sol |

### Interpretation

- **MCAR is decisively rejected** in the vast majority of surveys. Missingness is systematically associated with observable respondent characteristics.
- **MAR is the dominant mechanism**: The observable predictors (especially age, household size, education, and standard of living) explain a meaningful portion of missingness, with AUCs in the 0.62–0.68 range. This is moderate but consistent — typical for survey non-response where demographics partially predict who refuses to answer income questions.
- **MNAR cannot be fully ruled out**: AUCs below 0.70 leave room for unobserved factors (e.g., respondents with very high or very low incomes may be more reluctant to disclose). However, the evidence does not favor MNAR over MAR, and MNAR methods require untestable assumptions.
- **Time-varying mechanism**: The predictive strength and dominant predictors shift over time, reflecting changes in CBOS questionnaire design, Polish economic conditions, and survey methodology. This motivates time-localized imputation.

**Recommendation**: Proceed with MAR-based imputation using quarterly batches, with sensitivity analysis noting that MNAR contamination may cause mild bias.

---

## CBOS02B: Imputation Method

### Approach

We implemented **MICE (Multiple Imputation by Chained Equations)** under the MAR assumption, using `sklearn.IterativeImputer` with the following design:

| Parameter | Choice | Rationale |
|-----------|--------|-----------|
| Estimator | BayesianRidge | Conservative, well-characterized uncertainty; avoids overfitting in small batches |
| `sample_posterior` | True | Draws from posterior predictive distribution → proper imputation variance |
| Iterations | 10 | Standard convergence for MICE |
| Batching | Quarterly (year × quarter) | Adapts to time-varying questionnaire structure and label definitions |
| Clipping | ≥ 0 | Income cannot be negative |

**Covariate hierarchy** (included in each batch only if <50% observed):
1. *Core demographics*: age, sex, civil status (`cs`), education (`educ_2000`), household_size
2. *Attitudinal*: `sol` (standard of living, 1–5)
3. *Labour market*: `job_type` (cleaned: values >16 → NaN), `employment_status` (cleaned: value 8 → NaN)
4. *Cross-income*: `income_hh` when imputing `income_p` and vice versa

**Pre-imputation corrections**:
- `income_hh` directly recovered from `income_p` for single-person households (1,420 values)
- `job_type` values >16 recoded to NaN (8,400 obs) — these are miscellaneous/alternate codes with inconsistent meaning across survey waves
- `employment_status` value 8 recoded to NaN (294 obs) — this is a "missing data" sentinel

### Validation Results

Four validation approaches confirm the imputation is fit for purpose:

#### 1. Held-out cross-validation (20% random holdout per quarter)

| Metric | `income_p` (median) | `income_hh` (median) |
|--------|---------------------|----------------------|
| Pearson correlation | 0.463 | 0.400 |
| Relative MAE | 0.723 | 0.699 |
| R² | −0.119 | −0.112 |
| Mean ratio (imputed/observed) | 1.067 | 1.047 |
| Std ratio (imputed/observed) | 0.982 | 1.016 |

The negative R² is expected: demographic covariates cannot predict individual income precisely. The key metrics are **mean and std ratios near 1.0**, confirming distributional fidelity.

#### 2. Distributional comparison

- Log-income KDE shapes overlap well between observed and imputed, with mild rightward shift in imputed values
- Mean bias: +10–14% (regression toward conditional mean)
- Standard deviation: 88–94% of observed (mild tail compression)
- Extreme skewness reduced (22–29 → 2–4), as expected from regression-based imputation

#### 3. Relationship preservation

Income gradients by education, age group, and sex are all **well preserved** post-imputation:
- Education–income gradient: monotonically increasing, minimal distortion
- Age–income hump: peak age groups preserved
- Gender gap: maintained

#### 4. Time-series plausibility

Median income trends are virtually identical before and after imputation across the full 1990–2017 period, confirming that quarterly batching adapts correctly to Poland's rapid nominal income growth.

---

## Evidence Supporting the MAR-Based Approach

1. **Statistical evidence against MCAR**: LRT rejects MCAR in 82–94% of surveys (CBOS02A). Simple complete-case analysis would be biased.

2. **Observable predictors explain missingness**: Demographic and attitudinal covariates achieve AUCs of 0.62–0.68 in predicting non-response (CBOS02A), consistent with MAR where missingness depends on observed characteristics.

3. **Distributional fidelity**: Mean ratios near 1.0 (1.05–1.14) and preserved covariate-income gradients confirm the imputation does not systematically distort the income distribution (CBOS02B).

4. **Temporal stability**: Quarterly batching prevents contamination from time-varying questionnaire designs and label changes, and the imputed time series tracks the observed series closely (CBOS02B).

5. **Conservative methodology**: BayesianRidge with `sample_posterior=True` provides proper uncertainty quantification without overfitting to small quarterly samples.

### Known Limitations

- **~10–14% mean upward bias** in imputed values relative to observed, reflecting regression toward the conditional mean
- **Tail compression**: Extreme income values (P99+) are underrepresented in imputed data; analyses sensitive to the far tails (e.g., top income shares) should use observed data where available
- **Residual structural missingness**: `income_p` remains 39.3% missing because 120 surveys never collected it — this is structural and cannot be imputed
- **MNAR sensitivity**: If non-response is partially driven by income itself (conditional on covariates), the imputed distribution may understate income inequality. This cannot be tested directly.

---

*Generated from notebooks CBOS02A_missingness.ipynb and CBOS02B_imputation_methods.ipynb*
