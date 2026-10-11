# API 参考（Phase 1）

## 1. 高层入口

```python
from causalrank import CausalRank

model = CausalRank(
    treatment="route",             # 处理列名（互斥 K 类，K >= 2）
    outcome="converted",           # 结果列名（0/1，v0.1 唯一支持）
    confounders=["age", "income"], # Treatment 前协变量（>= 1 个）
    estimator="iptw",              # 仅 "iptw"
    propensity_model="gbm",        # 仅 "gbm"（联合多分类）
    estimand="population_mean",    # 仅 "population_mean"
    weighting="hajek",             # 仅 "hajek"
    confidence_level=0.95,         # Phase 2/3 才使用
    n_bootstrap=300,               # Phase 2 才使用
    n_jobs=1,                      # Phase 2 才使用
    random_state=2026,
    cross_fit=5,                   # None 或 int >= 2：OOF 倾向得分
    clip_propensity=None,          # None 或 eps in (0, 0.5)：显式下界裁剪
    gbm_params=None,               # 覆盖默认 GBM 超参（禁止覆盖 random_state）
    impute_numeric="median",
    impute_categorical="most_frequent",
    categorical_fill_value="missing",
    max_category_cardinality=25,
    min_group_n=30,                # 质量门控阈值（下同）
    min_ess=30.0,
    min_ess_ratio=0.2,
    min_propensity_warn=0.01,
    max_unsupported_share=0.05,    # 目标样本中 e_a(X) 过低的占比上限（共同支持）
    max_weight_share_warn=0.1,
    smd_threshold=0.1,
    smd_severe_threshold=0.25,
    quantiles=(0.01, 0.05, 0.5, 0.95, 0.99),
)
result = model.fit(df)
```

不受支持的取值会抛出 `CausalRankValidationError`（例如 `estimator="aipw"`、
`weighting="ht"`），并说明 Phase 1 的边界。

## 2. 返回对象 `CausalRankResult`

| 字段 | 类型 | Phase 1 内容 |
|---|---|---|
| `effects` | `EstimatesTable` | 各臂点估计与权重/ESS |
| `covariance` | `np.ndarray \| None` | `None`（Phase 2） |
| `pairwise` | `DataFrame \| None` | `None`（Phase 3） |
| `ranks` | `DataFrame \| None` | `None`（Phase 3） |
| `diagnostics` | `DiagnosticsReport` | 诊断表 + 结构化警告 |
| `metadata` | `dict` | 配置、版本、样本量、公式、免责声明 |
| `inference_available` | `bool` | `False` |

### 2.1 `result.effects.to_dataframe()`

列：`label, n_treated, raw_mean, mu, se, ci_low, ci_high, weight_sum, ess`。
`label` 按**规范序**（排序后的唯一标签）排列；`se / ci_low / ci_high` 为 `NaN`（Phase 1）。

### 2.2 `result.diagnostics`

- `.group_summary`：`label, n, share, raw_outcome_mean, mu_hajek_iptw`；
- `.weights`：`label, n, weight_sum, ess, ess_ratio, w_min, w_max, max_weight_share` + 权重分位列；
- `.overlap`：`label, e_arm_*, e_obs_*, frac_arm_below_threshold, frac_obs_below_threshold`；
  前者是**整个目标样本**中 `e_a(X)` 低于阈值的占比（共同支持检查），后者是被处理行中的占比；
- `.balance`：长表 `feature, level, pair, mean_a/b(, _weighted), sd_ref, smd_unweighted, smd_weighted`；
- `.balance_summary`：每项 `max_abs_smd_*`、超阈值 pair 数；
- `.warnings`：`DiagnosticWarning(code, level, message, details)` 列表；
  关键码：`no_common_support`(critical)、`weak_common_support`、`low_propensity`、
  `low_ess`、`extreme_weight`、`residual_imbalance`、`severe_imbalance`(critical)；
- `.gate`：`pass | caution | fail`；
- `.summary()`：上述内容的文本渲染 + 因果提示。

### 2.3 `result.to_dict()` / `result.to_json()`

机器可读导出；`NaN` 序列化为 `null`。所有数值 API 不依赖绘图库。

## 3. 低层估计器（可单独验证）

```python
from causalrank.estimators import (
    fit_propensity,        # (X, t_idx, labels, num, cat, config) -> PropensityFit
    align_proba_columns,   # predict_proba 列 -> 公开标签顺序
    hajek_estimate,        # (y, t_idx, propensity, labels) -> IPTWEstimate
    effective_sample_size,
)
from causalrank.diagnostics import (
    weight_diagnostics, overlap_diagnostics, standardized_mean_differences,
)
```

- `PropensityFit.propensity`：`(n, K)`，列顺序 = `labels`，行和为 1；
- `IPTWEstimate`：`mu, weights, weight_sums, numerators, ess, n_treated, raw_means,
  observed_propensity, clipped_fraction`；
- `horvitz_thompson_estimate` 为**内部**核对函数，不承诺公共 API 稳定性。

## 4. 合成数据

```python
from causalrank.simulation import (
    make_dataset, make_oracle_frame, run_estimand_benchmark,
    summarize_benchmark, summarize_runs,
)

ds = make_dataset("confounded", n=4000, random_state=2026)
ds.df          # x1, x2, x3, x4, treatment, outcome
ds.mu_true     # pd.Series，索引 = 臂标签
ds.mu_true_mc_se, ds.n_reference, ds.reference_seed
```

`make_oracle_frame` 额外附带真实的 `p_t_*` 与 `p_y_*` 列（测试/oracle 对照用）。
场景名见 `scenario_names()`（8 个预设）。

`run_estimand_benchmark(...)` 返回 `BenchmarkResult`：

- `.replications`：逐（重复 × 臂）行（含 `status`、`mu_hat`、`raw_mean`、`ess`）；
- `.summary`：臂级 `bias / rmse / bias_raw / rmse_raw` + `n_ok / n_failed / failure_rate`
  （只用成功轮计算误差，失败率并列披露）；
- `.runs`：逐重复行的诊断快照（`quality_gate`、`warning_codes`、`min_e_obs`、
  `max_weight_share`、`min_ess_ratio`、`max_abs_smd_weighted`、失败时的 `error`）；
- `.runs_summary`：每个 (scenario, n) 的失败率、门控分布、各警告码触发率、诊断均值。

**拟合失败不会中断整批实验**：失败轮被记录并计入 `failure_rate`，不会被静默丢弃。

## 5. 异常

- `CausalRankValidationError`（兼 `ValueError`）：输入/配置不满足契约；
- `CausalRankError`：估计过程中的数值/内部错误（如 positivity 违背、概率非有限）。

## 6. 稳定性约定

- 数值列的**顺序与名称**在 Phase 2/3 只增不改（`se/ci` 将被填充，`covariance/pairwise/ranks`
  将从 `None` 变为有值）；
- 警告 `code` 为稳定标识，`message` 文本可变；
- `metadata` 的键只增不删。
