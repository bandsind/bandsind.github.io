---
title: "Correlation and P-Values in Python: A Step-by-Step Tutorial"
date: "2026-11-12"
tags:
  - data science
  - python
  - statistics
  - tutorial
  - hypothesis testing
bookmark: true
thumbnail: "/assets/img/tutorials/correlation-p-values/positive-correlation.png"
---

<style>
/* Code blocks: the theme leaves <code> inline with vertical padding, so each
   line's background paints over the line above it. Make it a block instead. */
main .highlighter-rouge .highlight pre { padding: 0; background: transparent; }
main .highlighter-rouge .highlight pre code { display: block; top: 0; padding: 1rem; line-height: 1.6; overflow-x: auto; }
/* Figures: the theme stretches every image to the full column width.
   Keep the plots near their native size and never wider than the column. */
.inner-content main p img { width: auto; max-width: min(100%, 520px); height: auto; }
</style>

When two variables appear to move together, how do we measure their relationship? And when we calculate a difference from a sample, how can we tell whether it may be due to random sampling? This tutorial uses **Python code, actual notebook outputs, and visualizations** to explore two tools: **Pearson correlation** and the **one-sample t-test**.

We start with simulated data where we control the relationship, then use a telecom customer dataset to demonstrate hypothesis testing on the `Eve Charge` column. I explain the meaning of each result rather than relying on the p-value alone.

**What you will learn:** how to read a scatterplot, interpret Pearson’s *r*, calculate a p-value, visualize a sampling distribution, and make a hypothesis-test decision without overstating the conclusion.

> **About the results:** The numbers and graphs below are the saved outputs of the notebook linked in Section 11. The sample is randomly drawn with a fixed NumPy seed in that run. Rerunning under different library versions or with changed source data may produce different outputs. The customer dataset is loaded from an external GitLab URL, so the notebook needs an internet connection.

## 1. Set up the Python environment

We use NumPy for numerical operations, pandas for tables, Matplotlib for plots, and SciPy for statistical tests. The fixed seed makes the simulated data repeatable.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, t, ttest_1samp

np.random.seed(42)
```

Install the packages if you do not have them:

```bash
python -m pip install numpy pandas matplotlib scipy jupyter
```

## 2. Positive correlation

For our first example, we generate 1,000 values for `x` and create `y_pos` by adding random noise. Because the response increases with `x`, we expect an upward trend. The noise makes the relationship realistic rather than perfectly straight.

```python
x = np.random.randint(0, 50, 1000)
y_pos = x + np.random.normal(0, 10, 1000)

plt.scatter(x, y_pos)
plt.xlabel('x')
plt.ylabel('y_pos')
plt.title('Positive correlation')
plt.show()

r, p_value = pearsonr(x, y_pos)
print('r =', r)
print('p-value =', p_value)
```

![Scatterplot of simulated data showing a positive linear relationship]({{ "/assets/img/tutorials/correlation-p-values/positive-correlation.png" | relative_url }})

**Output from the notebook:**

```text
r = 0.8101435222157062
p-value = 1.0630495901638522e-233
```

**Interpretation:** `r ≈ 0.810` indicates a strong positive linear relationship in this simulated dataset. The very small p-value gives evidence against the null hypothesis of **zero population correlation**, given the test assumptions. It does **not** mean there is an 81% probability of a cause-and-effect relationship.

### View the correlation matrix

The same calculation can be displayed as a two-by-two correlation matrix. The diagonal is 1 because each variable is perfectly correlated with itself.

```python
np.corrcoef(x, y_pos)

corr_df = pd.DataFrame({'x': x, 'y': y_pos})
corr_df.corr()

pd.plotting.scatter_matrix(corr_df, alpha=0.2)
plt.show()
```

**NumPy matrix output:**

```text
array([[1.        , 0.81014352],
       [0.81014352, 1.        ]])
```

![Scatter-matrix output combining individual histograms and a positive correlation scatterplot]({{ "/assets/img/tutorials/correlation-p-values/correlation-scatter-matrix.png" | relative_url }})

The two off-diagonal plots show the association; the diagonal plots show the separate distributions of the variables.

## 3. Negative correlation

Now we reverse the direction by using `-x`. The scatterplot should slope downward as `x` increases.

```python
y_neg = -x + np.random.normal(0, 10, 1000)
plt.scatter(x, y_neg)
plt.xlabel('x')
plt.ylabel('y_neg')
plt.title('Negative correlation')
plt.show()

r, p_value = pearsonr(x, y_neg)
print('r =', r)
print('p-value =', p_value)
```

![Scatterplot showing a negative linear relationship]({{ "/assets/img/tutorials/correlation-p-values/negative-correlation.png" | relative_url }})

**Output from the notebook:**

```text
r = -0.8268637744564672
p-value = 1.0754544324922143e-251
```

**Interpretation:** `r ≈ -0.827` is a strong **negative** linear relationship. The sign tells us the direction, while the absolute value describes its strength. A very small p-value is not the same thing as a large effect; here we check both.

## 4. Approximately zero correlation

For the third example, we generate `x_rand` and `y_rand` independently. We expect a cloud of points rather than an obvious slope.

```python
x_rand = np.random.randint(0, 50, 1000)
y_rand = np.random.normal(0, 50, 1000)

plt.scatter(x_rand, y_rand)
plt.xlabel('x_rand')
plt.ylabel('y_rand')
plt.title('No correlation')
plt.show()

r, p_value = pearsonr(x_rand, y_rand)
print('r =', r)
print('p-value =', p_value)
```

![Scatterplot of independently simulated variables with near-zero Pearson correlation]({{ "/assets/img/tutorials/correlation-p-values/no-correlation.png" | relative_url }})

**Output from the notebook:**

```text
r = -0.005690010402558496
p-value = 0.8573799788609253
```

**Interpretation:** `r ≈ -0.006` is effectively zero in this sample, and `p ≈ 0.857` does not provide evidence against zero population correlation at `α = 0.05`. This does not prove independence or rule out nonlinear relationships.

### Comparing the three patterns

| Example | Pearson r | p-value | Meaning |
|---|---:|---:|---|
| Positive | 0.810 | 1.06 × 10⁻²³³ | Strong positive linear association |
| Negative | −0.827 | 1.08 × 10⁻²⁵¹ | Strong negative linear association |
| Independently generated | −0.00569 | 0.857 | Near-zero observed linear association |

These examples illustrate **association, not causation**. A correlation may also hide subgroup differences, outliers, or nonlinear patterns, so plotting the data is essential.

## 5. Load and explore the telecom customer data

Next, the notebook uses a customer churn dataset. Our focus is the numerical field `Eve Charge`, which represents an evening charge amount. For this exercise, **we treat the entire loaded dataset as the reference population**, then repeatedly draw samples from it. This is a teaching example of how a statistical test behaves, not an independent analysis of customer churn drivers.

```python
na_values = [' ', 'NaN', 'N/A']
churn = pd.read_csv(
    'https://gitlab.com/CEADS/DrKerby/python/raw/master/churn.txt',
    na_values=na_values
)

churn['Eve Charge'].hist(bins=20)
plt.xlabel('Eve Charge')
plt.ylabel('Number of customers')
plt.show()

mu = churn['Eve Charge'].mean()
print(mu)
```

![Histogram of evening charges across all customers in the reference dataset]({{ "/assets/img/tutorials/correlation-p-values/population-evening-charge.png" | relative_url }})

**Output from the notebook:**

```text
17.083540354035403
```

**Interpretation:** The reference mean is approximately **17.08**. The histogram helps us understand the spread of individual evening charges before we study the means of samples.

## 6. Take one sample and state the hypotheses

We draw a sample of 100 rows **with replacement**. That means a customer record could be selected more than once. This matches the sampling approach in the notebook.

```python
n = 100
sample = churn.sample(n=n, replace=True)
x_bar = sample['Eve Charge'].mean()
print(x_bar)

sample['Eve Charge'].hist(bins=20)
plt.xlabel('Eve Charge')
plt.ylabel('Number of customers')
plt.show()
```

![Histogram of evening charges for a randomly selected sample of 100]({{ "/assets/img/tutorials/correlation-p-values/sample-evening-charge.png" | relative_url }})

**Output from the notebook:**

```text
17.740099999999998
```

The sample mean is approximately **17.74**, somewhat above the reference mean of **17.08**. The question is whether that difference is unusual under sampling variability.

For a **two-sided one-sample t-test**, we state:

- **H₀ (null hypothesis):** the mean of the population from which the sample comes equals `μ = 17.08354`.
- **H₁ (alternative):** the population mean differs from `μ`.
- **Significance threshold:** `α = 0.05`.

The test statistic is `t = (x̄ − μ) / (s / √n)`, where `s` is the sample standard deviation. A one-sample t-test compares **one sample mean with a specified reference mean**; two separate groups are not required.

## 7. Simulate 3,000 samples

Instead of looking at one sample alone, the notebook repeats the process 3,000 times. We record the sample mean and calculated t-statistic each time.

```python
x_bars = []
t_values = []
for i in range(3000):
    sample_i = churn.sample(n=n, replace=True)
    sample_mean = sample_i['Eve Charge'].mean()
    sample_std = sample_i['Eve Charge'].std()
    x_bars.append(sample_mean)
    t_values.append((sample_mean - mu) / (sample_std / np.sqrt(n)))

plt.hist(x_bars, bins=20)
plt.xlabel('Sample mean of Eve Charge')
plt.ylabel('Number of samples')
plt.show()
```

![Histogram of the means of 3000 resampled groups]({{ "/assets/img/tutorials/correlation-p-values/distribution-of-sample-means.png" | relative_url }})

**Interpretation:** The means group around **17.08**, with a much narrower spread than the individual customer charges. This illustrates sampling variability and the approximate bell shape of the sampling distribution of the mean for sufficiently large samples.

```python
plt.hist(t_values, bins=20)
plt.xlabel('t-value')
plt.ylabel('Number of samples')
plt.show()
```

![Histogram of t-statistics from 3000 resampled groups]({{ "/assets/img/tutorials/correlation-p-values/distribution-of-t-statistics.png" | relative_url }})

The simulated t-statistics cluster near zero. For samples of size 100, a t reference distribution with **99 degrees of freedom** gives a useful cutoff, although the exact theoretical t-distribution requires additional assumptions about the sampled population.

## 8. Find the critical value and check the error rate

For a two-sided test at the 5% significance level, the critical values are at the 2.5% tails of the reference t-distribution.

```python
alpha = 0.05
p = 1 - alpha / 2
dof = n - 1
t_critical = t.ppf(p, dof)
print(t_critical)

t_values = np.array(t_values)
inside = np.mean(np.abs(t_values) <= t_critical)
print('Fraction of t-values within +/- t_critical:', inside)
print('Fraction outside (wrongly rejecting a true H0):', 1 - inside)
```

**Output from the notebook:**

```text
1.9842169515086827
Fraction of t-values within +/- t_critical: 0.9503333333333334
Fraction outside (wrongly rejecting a true H0): 0.04966666666666664
```

**Interpretation:** About **95.0%** of simulated t-statistics fall inside the cutoff, and about **5.0%** fall outside. The latter approximates the *Type I error rate*: rejecting a true null hypothesis because of sample variability. This simulation deliberately samples from the same reference dataset used to define `μ`.

## 9. Calculate the observed p-value

Now we return to the single sample from Step 6. We calculate its observed t-statistic and the **two-tailed p-value** using SciPy’s t-distribution survival function (`sf`).

```python
s = sample['Eve Charge'].std()
t_stat = (x_bar - mu) / (s / np.sqrt(n))
p_val = 2 * t.sf(abs(t_stat), dof)

print('t statistic =', t_stat)
print('t critical  =', t_critical)
print('p-value     =', p_val)

if p_val < alpha:
    print('p-value < alpha: reject the null hypothesis')
else:
    print('p-value >= alpha: fail to reject the null hypothesis')
```

**Output from the notebook:**

```text
t statistic = 1.570518349158167
t critical  = 1.9842169515086827
p-value     = 0.1194852987162764
p-value >= alpha: fail to reject the null hypothesis
```

**Interpretation:** Because `p ≈ 0.119` is greater than `α = 0.05`, the sample does **not** provide enough evidence to reject the reference mean. Also `|t| = 1.571` is smaller than the critical value `1.984`, so both decision methods agree.

We can check the same result with SciPy’s built-in test:

```python
ttest_1samp(sample['Eve Charge'], mu)
```

**Notebook output:**

```text
TtestResult(statistic=1.5705183491581667,
            pvalue=0.11948529871627647, df=99)
```

Failing to reject the null hypothesis **does not prove the null is true**. It only means the observed sample difference is not statistically significant at the chosen threshold.

## 10. What the analysis shows and what it does not

Here are my main takeaways from running the examples:

1. **Direction and strength are different from significance.** Pearson’s `r` tells us how strongly two variables are linearly associated; the p-value addresses a specific statistical hypothesis.
2. **A graph matters.** Scatterplots reveal the pattern and potential issues that a single correlation number may not reveal.
3. **Samples vary even when the reference population stays the same.** The 3,000 simulated samples demonstrate why different sample means occur.
4. **A p-value is not a probability that H₀ is true.** Our final result of `0.119` is a probability about results at least as extreme as the observed test statistic *assuming H₀ and the model assumptions*.
5. **Study design matters.** The churn exercise uses resampling from the same dataset that supplies the benchmark mean. It illustrates test mechanics, not a discovery that evening charges are or are not different in an external population.

**Limitations:** The simulated correlation examples are artificial; correlation does not establish causation. The t-test uses a numerical field from one externally hosted dataset, and its reference distribution is approximate when the population is not normally distributed. Outcomes can change with sampling decisions, assumptions, and the dataset source. Statistical significance should be interpreted alongside effect size and the question being investigated.

## 11. Full notebook and references

If you want to run every cell, including the additional sample tables and plots, the notebook is available here. The download is provided **inside this post**, not in the site sidebar.

[**Download the complete Jupyter notebook (.ipynb)**]({{ "/notebooks/Sindi-P3-correlation-and-p-values.ipynb" | relative_url }})

**References and tools**

- [SciPy: `pearsonr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
- [SciPy: `ttest_1samp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html)
- [SciPy: Student t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)
- [pandas plotting documentation](https://pandas.pydata.org/docs/reference/api/pandas.plotting.scatter_matrix.html)
- [Customer dataset used in the notebook](https://gitlab.com/CEADS/DrKerby/python/raw/master/churn.txt)

*The tutorial is based on the supplied class notebook, with its recorded outputs and figures retained and reorganized for a general audience. Cite the instructor or original author as appropriate if the notebook was provided as course material.*
