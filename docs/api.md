# API 参考（Phase 1–3）

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
    confidence_level=0.95,         # 单项 CI 置信水平
    n_bootstrap=None,              # >= 1 显式开启联合 Bootstrap 推断（每轮重拟合，较慢）
    n_jobs=1,                      # Bootstrap 并行度（结果与串行逐位一致）
    ci_method="normal",            # "normal" 或 "percentile"（后者需保留 draws）
    bootstrap_keep_draws=True,     # 是否保留 B×K draws
    bootstrap_min_success_rate=0.9,# 低于此成功率阻断推断
    bootstrap_min_successes=30,    # 低于此成功轮数阻断推断
    higher_is_better=True,         # False = 结果越低越好（成本/时延型）；只翻转“更好”语义
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

| 字段 | 类型 | 内容 |
|---|---|---|
| `effects` | `EstimatesTable` | 各臂点估计与权重/ESS；`se/ci_*` 在启用推断后填充 |
| `covariance` | `np.ndarray \| None` | `Σ_boot`（K×K，标签规范序）；未启用/被阻断时为 `None` |
| `pairwise` | `PairwiseTable \| None` | Bonferroni 两两同时区间与方向判定（Phase 3） |
| `ranks` | `RankTable \| None` | 点排名 + 保守排名置信区间 `[1+B_i, K-W_i]`（Phase 3） |
| `diagnostics` | `DiagnosticsReport` | 诊断表 + 结构化警告 |
| `metadata` | `dict` | 配置、版本、`bootstrap`、`rank_note`、免责声明 |
| `inference_available` | `bool` | 推断是否产出 |

### 2.1 `result.effects.to_dataframe()`

列：`label, n_treated, raw_mean, mu, se, ci_low, ci_high, weight_sum, ess`。
`label` 按**规范序**（排序后的唯一标签）排列；未启用推断时 `se / ci_low / ci_high` 为 `NaN`。

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

### 2.3 `result.pairwise`（Phase 3）

`.to_dataframe()` 列（每对 `i < j` 一行，标签规范序）：
`label_i, label_j, mu_i, mu_j, diff, se_diff, ci_low, ci_high, z_crit, n_pairs,
confidence_level, decision, confirmed, point_tie, degenerate_se`。

- `diff = mu_i - mu_j`，`se_diff = sqrt(Σ_ii + Σ_jj - 2Σ_ij)`（联合协方差带入相关性）；
- `ci_*` 为 Bonferroni 同时区间 `diff ± z_{1-α/(2M)}·se_diff`，`M = n_pairs = K(K-1)/2`；
- `decision ∈ {"i_better", "j_better", "inconclusive"}`，语义是**更好**方向
  （受 `higher_is_better` 影响）；`inconclusive` **不是**相等的证据；
- `degenerate_se=True` 表示差值方差数值为零：区间退化为点，按 `diff` 的符号判定，
  精确点并列（`point_tie`）保持 `inconclusive`；
- `.confirmed`：仅含已确认优劣关系的子表；
- `.summary()` / `.to_dict()`：含公式、`z_crit`、置信水平（可追溯）。

### 2.4 `result.ranks`（Phase 3）

`.to_dataframe()` 列：
`label, mu, point_rank, rank_low, rank_high, n_confirmed_better, n_confirmed_worse,
point_tie`。

- 保守排名置信区间 `[rank_low, rank_high] = [1 + B_i, K - W_i]`：
  `B_i` = 可确认优于 i 的臂数，`W_i` = 可确认劣于 i 的臂数（由同时区间得出）；
- `point_rank`：点排名，**并列共享名次**（competition ranking），`point_tie` 显式标记；
  绝不按标签字母序伪造确定排序；
- 真实效果并列时严格全序名次不唯一，区间相应变宽（tie convention 见 `.to_dict()`）；
- 结论依赖联合协方差与渐近正态近似；bootstrap 中点排名第一的频次**不是**最优概率。

### 2.5 `result.to_dict()` / `result.to_json()`

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
from causalrank.inference import (
    joint_bootstrap,       # 联合重拟合 Bootstrap -> JointBootstrapResult
    derive_round_seeds, resample_indices,
    validate_covariance,   # 对称/对角/PSD 校验（非 PSD 报错，不修补）
    normal_ci, percentile_ci,
    pairwise_comparisons,  # Bonferroni 同时差值区间（3.6）
    rank_confidence_sets,  # [1+B_i, K-W_i] 保守排名区间
    rank_from_estimates,   # 低层排名内核入口（规格 2.2）-> RankInference
)
```

- `PropensityFit.propensity`：`(n, K)`，列顺序 = `labels`，行和为 1；
- `IPTWEstimate`：`mu, weights, weight_sums, numerators, ess, n_treated, raw_means,
  observed_propensity, clipped_fraction`；
- `JointBootstrapResult`：`draws (B_ok×K)`、`covariance (K×K)`、`failures`
  （`round/seed/reason` 结构化失败报告）、`seeds`、`base_seed`、`usable`、`warnings`；
- `horvitz_thompson_estimate` 为**内部**核对函数，不承诺公共 API 稳定性。

低层排名内核（规格 2.2，可单独验证）：

```python
from causalrank.inference import rank_from_estimates

ranking = rank_from_estimates(
    estimates={"A": 0.25, "B": 0.24, "C": 0.17},  # 插入序 = 协方差行/列序
    covariance=sigma,
    confidence_level=0.95,
    method="bonferroni",   # v0.1 仅支持 Bonferroni；max-T/stepdown 为后期扩展
    higher_is_better=True,
)
ranking.to_dataframe()     # 排名表（同 result.ranks.to_dataframe()）
ranking.pairwise           # PairwiseTable
```

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

`run_se_benchmark(...)` 返回 `SEBenchmarkResult`（Phase 2 校准）：

- `.replications`：逐（重复 × 臂）行（`mu_hat / se_hat / ci_low / ci_high / covers`）；
- `.summary`：`empirical_sd`（$\hat\mu$ 的经验标准差）、`mean_se`、`se_ratio = mean_se/empirical_sd`、
  `coverage` 与 `coverage_mc_se = sqrt(p(1-p)/R)`。

`run_rank_benchmark(...)` 返回 `RankBenchmarkResult`（Phase 3 校准）：

- `.replications`：逐（重复 × 臂）行（`true_rank_min/max`（并列感知真值）、
  `rank_low/high`、`covers`、`point_rank`）；
- `.summary`：臂级 `coverage / coverage_mc_se / mean_rank_width`；
- `.run_summary`：`any_confirmed_rate`（真并列场景下的假确认率）、
  `mean_n_confirmed_pairs`、`all_rank_sets_full_rate`、`blocked_rate`。

研究脚本：`scripts/run_simulation.py`（Bias/RMSE/失败与门控统计）、
`scripts/run_se_validation.py`（SE 校准与覆盖率）、
`scripts/run_rank_validation.py`（排名覆盖率与假确认率）。

## 5. 异常

- `CausalRankValidationError`（兼 `ValueError`）：输入/配置不满足契约；
- `CausalRankError`：估计过程中的数值/内部错误（如 positivity 违背、概率非有限）。

## 6. 稳定性约定

- 数值列的**顺序与名称**只增不改（`pairwise/ranks` 已在 Phase 3 填充：
  `pairwise` 为 `PairwiseTable`、`ranks` 为 `RankTable`，列集合自此固定）；
- 警告 `code` 为稳定标识，`message` 文本可变；
- `metadata` 的键只增不删；
- 同数据 + 同 `random_state` + 同配置 → 逐位一致的结果（含 Bootstrap draws 与 Σ），
  与 `n_jobs` 无关。
