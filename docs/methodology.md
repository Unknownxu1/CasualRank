# CausalRank 方法说明（Phase 1）

> 对应实现：`src/causalrank/{estimators,diagnostics,validation,simulation}`。
> 本文给出**估计目标、公式、适用条件、假设与数值边界**；仅“跑通”不构成正确性依据。

## 1. 估计目标（estimand）

对 Treatment `a = 1..K`：

$$\mu_a = E_{\mathcal P}[Y(a)]$$

- $Y(a)$：接受 Treatment `a` 时的潜在结果（v0.1 为二元 0/1）；
- $\mathcal P$：输入样本所代表的**样本总体**（ATE/平均潜在结果目标，`estimand="population_mean"`）；
- 关键点：**所有 Treatment 的 $\mu_a$ 都在同一目标总体上定义**，因此点估计之间的比较是可比的。
  各 Treatment 独立计算“相对各自对照组的 uplift”会得到不可比的目标，本项目不这样做。

结果 $Y$ 的定义与观察窗口必须在数据中保持一致；一致性（consistency）由使用者保证。

## 2. 倾向得分：多分类 GBM（预测模型，不是因果方法）

$$e_a(x) = P(T = a \mid X = x), \qquad \sum_{a=1}^K e_a(x) = 1.$$

**模型选择**：sklearn `HistGradientBoostingClassifier`（联合多分类概率模型，非 one-vs-rest）。
与 McCaffrey et al. (2013) 中广义提升模型可能采用的一对其余构造不同，这里明确采用
**单一联合多分类模型**；两者不必相同，本文实现与该文不宣称完全一致。

**预处理管线**（全部在 `sklearn.Pipeline` 内，杜绝泄露）：

| 变量 | 处理 |
|---|---|
| 数值 | 统一转 `float64`（可空 `Int64` 等会被 `SimpleImputer` 拒绝）；`SimpleImputer(strategy="median"\|"mean"\|"constant")`（训练折内 fit） |
| 类别 | 统一转 `object`，所有缺失标记（`pd.NA`/`NaN`/`None`/`pd.NaT`）归一为 `np.nan`；`SimpleImputer(strategy="most_frequent"\|"constant")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` |

**dtype 约定**（pandas 2/3 兼容）：`object` / `StringDtype`（pandas 3 默认字符串类型）/
`category` / `bool` / `BooleanDtype` 一律按**类别**处理并归一表示；日期时间等其余类型
显式报错。归一化只改容器表示不改取值——原因是 sklearn 的 imputer/encoder 遇到裸 `pd.NA`
会抛 `TypeError: boolean value of NA is ambiguous`，`bool` 列会被 `SimpleImputer` 拒绝。
回归测试覆盖 bool / 字符串 / 含 `pd.NA` 的 object / 可空 `Int64` 四类协变量的端到端分析。

规则：
- 插补器、编码器、GBM 只在**训练折 / Bootstrap 重抽样**内 fit，绝不在整份样本上预 fit；
- `predict_proba` 输出列按 `model.classes_` 显式映射回**公开标签规范序**（排序后的唯一标签），
  实现见 `align_proba_columns()`；禁止假设列位置；
- 严禁用 `decision_function` 或未归一化的分数当概率。

**GBM 默认超参**（`config.py::default_gbm_params`，可经 `gbm_params` 覆盖）：

```
max_iter=400, learning_rate=0.02, max_depth=4, min_samples_leaf=60, l2_regularization=2.0
```

选择依据（在已知真值倾向面的合成 DGP 上调参，见 §7）：深度/低叶节点的树会追逐标签噪声、
给出过度自信的概率。这有两个后果：(a) 样本内拟合的 $\hat e_{T_i}$ 系统性偏大；
(b) out-of-fold 的 $\hat e_{T_i}$ 偶发极小值 → 权重爆炸。适度正则后倾向误差显著下降
（实验中 MAE 由 0.106 降至 0.059），OOF 权重 ESS 比由 ~0.10 回升至 ~0.60。
**AUC / 准确率不作为因果有效性的证据**，本包不将其作为质量指标输出。

**Out-of-fold（`cross_fit=m`）**：`StratifiedKFold(m, shuffle=True, random_state=...)`，
每折训练集必须包含全部 K 类（样本量不足时训练前报错）。OOF 倾向得分用于计算权重，
降低“同一份数据拟合与评估”的过拟合偏差；全样本拟合模型的用途仅限于描述与复用。
局限：OOF 倾向得分噪声更大，有限样本下方差可能上升；这是偏差—方差权衡，不是免费午餐。

**概率裁剪（`clip_propensity=eps`，默认关闭）**：
- 只对**实际 Treatment 概率** $e_{T_i}$ 做**下界**裁剪 $e \leftarrow \max(e, \varepsilon)$
  （$e\to 1$ 时权重 $1/e$ 有界，无需上界裁剪）；
- 裁剪比例随结果输出（`propensity_clipped` 警告）；
- 裁剪会改变有限样本偏差与权重含义，不能默默执行；
- 概率恰为 0 是 positivity 违背，直接报错，不被裁剪掩盖。

## 3. IPTW：Hájek 形式（主估计器）

权重与估计量：

$$w_i = \frac{1}{\hat e_{T_i}(X_i)}, \qquad
\hat\mu_a^{H} = \frac{\sum_{i=1}^n 1\{T_i=a\}\, w_i Y_i}{\sum_{i=1}^n 1\{T_i=a\}\, w_i}.$$

性质与边界：
- 二元 outcome 下组内正权重平均必落在 $[0,1]$；
- 对组内权重乘同一常数不变（测试覆盖）；
- 相较 HT 形式，有限样本通常更稳定（不依赖 $n$ 与权重整体尺度的匹配）；
- **不保证**无偏或平衡：偏差由倾向估计误差与识别假设共同决定。

内部核对用 Horvitz–Thompson 形式（不对外暴露）：

$$\hat\mu_a^{HT} = \frac1n \sum_{i=1}^n 1\{T_i=a\}\frac{Y_i}{\hat e_a(X_i)}.$$

**禁止**的混合算法：用组内归一化权重却在公式中除以整体 $n$。实现中 HT 与 Hájek
是两个独立函数（`horvitz_thompson_estimate` / `hajek_estimate`），测试分别手算核对。

## 4. 诊断定义

### 4.1 权重与有效样本量

$$\mathrm{ESS}_a = \frac{\left(\sum_i w_{ia}\right)^2}{\sum_i w_{ia}^2}, \qquad w_{ia} = 1\{T_i=a\} w_i .$$

输出：组内 $n_a$、权重和、ESS、ESS 比（$\mathrm{ESS}_a/n_a$）、权重分位数
（min / 1% / 5% / 50% / 95% / 99% / max，可配置）、最大单条权重占组内权重和的比例。

### 4.2 Overlap（共同支持的经验诊断）

对每个 Treatment `a` 输出：
- $e_a(X)$ 在**全部样本**上的分位数（该臂在协变量空间中的可达范围，positivity 需要它为正）；
- `frac_arm_below_threshold`：**整个目标样本**中 $e_a(X) <$ `min_propensity_warn` 的占比；
- $e_{T_i}(X_i)$ 在被分配到 `a` 的行上的分位数（IPTW 权重分母）；
- $e_{T_i}(X_i) < $ `min_propensity_warn` 的比例；
- 全局：$\max_a e_a(X)$ 接近 1 的行占比（近确定性分配）。

**共同支持检查覆盖目标样本，而不只是被处理行**（回归用例：三行概率
(0.5,0.5,0) / (0,0.5,0.5) / (0.5,0,0.5)——每臂在自己被处理的行上概率都正常，
但在 1/3 的目标样本上完全不可达）：

| 触发条件 | 警告码 | 级别 |
|---|---|---|
| 某行 $e_a(X)=0$ | `no_common_support` | critical（$E[Y(a)]$ 在该区域不可识别） |
| `frac_arm_below_threshold` > `max_unsupported_share`（默认 0.05） | `weak_common_support` | warning（人群均值依赖外推） |
| 被处理行 $e_{T_i}$ < `min_propensity_warn` | `low_propensity` | warning（权重极端） |
| 近确定性分配行占比 > 5% | `low_propensity` | warning |

### 4.3 标准化均值差 SMD

- **项（term）**：数值协变量本身；类别协变量按每个 level 的 0/1 指示变量；
  任何含缺失值的特征额外增加 `__missing__` 指示项（缺失模式也要平衡）。
- 对每个无序 Treatment 对 $(a,b)$：

$$\mathrm{SMD}_{ab} = \frac{m_a - m_b}{s_{\mathrm{ref}}}, \qquad
s_{\mathrm{ref}} = \sqrt{\frac{s_a^2 + s_b^2}{2}}$$

  - $m_g$：未加权（`smd_unweighted`）或 Hájek 加权组内均值
    $m_a^w = \sum_{T_i=a} w_i x_i / \sum_{T_i=a} w_i$；
  - $s_g$：组内**未加权**样本标准差（ddof=1，缺失项逐项剔除）。
    分母使用与权重无关的 pooled reference SD（Austin 2009 建议）：若用加权 SD，
    极端权重会缩小分母以外的差异，从而掩盖失衡。
- 汇报：每个项在所有 pair 上的 `max|SMD|`（加权/未加权）、超过 0.1 / 0.25 的 pair 数。
- 边界：$s_{\mathrm{ref}}=0$ 且均值差为 0 → SMD=0；$s_{\mathrm{ref}}=0$ 且均值差非 0 → $\pm\infty$
  并触发 `degenerate_smd`；有效观测数 < 2 → SMD=NaN。
- **数值稳定性**：SMD 是平移不变量，实现先对每个项按全体均值做列中心化，方差用两遍
  公式 $\sum(x-\bar x)^2$ 计算，**不**使用 $\sum x^2 - n\bar x^2$（后者在数据带大偏移量时
  灾难性相消，实测 `[1,3] vs [2,6]` 平移 `+1e9` 会把 SMD 变成 `-inf`）；
  回归测试覆盖 `+1e9` 与 `+1e12` 平移不变性。

**解释限制**：`max|SMD| <= 0.1` 只是筛查惯例，不是充分条件；弱混杂变量平衡不能掩盖
未测量强混杂的存在。加权后 SMD 变差（> 未加权）是模型失配/极端权重的信号，应如实报告。

### 4.4 质量门控

`quality_gate = pass | caution | fail`：`info` 不降级，`warning` → `caution`，
`critical` → `fail`。警告码：`small_group`、`rare_treatment`、`low_propensity`、
`weak_common_support`、`no_common_support`、`extreme_weight`、`low_ess`、
`residual_imbalance`、`severe_imbalance`、`duplicate_rows`、
`high_cardinality_categorical`、`missing_covariates`、`propensity_clipped`、
`degenerate_smd`、`label_dtype_cast`。门控只陈述经验诊断结果，
**不会**输出“可靠因果排序”类结论。

## 5. 数值与工程边界

| 情形 | 行为 |
|---|---|
| $e_{T_i}=0$ | 报错（positivity 违背），不静默处理 |
| 倾向非有限 / 行和≠1 | 报错（`CausalRankError`） |
| 某臂观测数 < 2 | 报错 |
| 某臂观测数 < `min_group_n` | 警告 `small_group` |
| `cross_fit` > 最小组样本数 | 训练前报错 |
| outcome 非 0/1 / 含缺失 | 报错 |
| treatment 含缺失 / 单类 | 报错 |
| 标签混合类型 / bool | 报错；整数浮点标签显式转换为 int 并记 info |
| 协变量 dtype 不支持（日期等） | 报错并提示转换方式 |
| 重复行 | 警告 `duplicate_rows`（iid 假设提示） |

## 6. 联合 Bootstrap 协方差（Phase 2）

### 6.1 协议

令原样本估计向量 $\hat{\boldsymbol\mu} = (\hat\mu_1,\ldots,\hat\mu_K)^\top$。第 $b$ 轮：

1. 用**同一个**行索引向量（有放回，单位=观测行）重抽样全部行；
2. 在重抽样样本上**重新执行全部协议**：校验、插补/编码、GBM 倾向拟合
   （含 `cross_fit` 的重新折划分）、权重与全部 K 组 Hájek 均值；
3. 得到 $\hat{\boldsymbol\mu}^{*(b)}$（列序=公开标签规范序）。

$$\hat\Sigma_{\mathrm{boot}} = \frac{1}{B-1}\sum_{b=1}^B
(\hat{\boldsymbol\mu}^{*(b)} - \bar{\boldsymbol\mu}^*)(\hat{\boldsymbol\mu}^{*(b)} - \bar{\boldsymbol\mu}^*)^\top
\;\equiv\; \texttt{np.cov(draws, rowvar=False, ddof=1)}$$

**绝对不能再除以 $n$**：每行是完整估计量向量，不是单条客户的贡献值（有测试显式断言）。

### 6.2 随机数与可复现性

- 第 $b$ 轮种子由 `SeedSequence(base_seed).spawn(B)` 派生，同时驱动重抽样与模型随机性
  （折划分、GBM）；
- 固定 `random_state` → 结果逐位可复现；**与执行顺序、`n_jobs` 并行度无关**（有测试）；
- `random_state=None` 时从系统熵生成 `base_seed` 并记录在 `result.metadata["bootstrap"]`。

### 6.3 失败轮与推断阻断

- 某轮若漏抽稀有臂、或数值失败：记录 `{round, seed, reason}` 并从 draw 矩阵排除，
  **绝不静默丢弃**；`bootstrap_failures` 警告给出失败数与原因分类；
- 阻断门槛（可配置）：成功轮数 < `bootstrap_min_successes`（默认 30）或成功率 <
  `bootstrap_min_success_rate`（默认 0.9）→ **不产出 Σ/SE/CI**，
  `bootstrap_inference_blocked`（critical）披露全部失败原因；
- `bootstrap_keep_draws=False` 可不保存 draws（仍计算 Σ）。

### 6.4 单项置信区间

- `ci_method="normal"`（默认）：$\hat\mu_a \pm z_{1-\alpha/2}\sqrt{\hat\Sigma_{aa}}$；
- `ci_method="percentile"`：保留 draws 的分位数区间，**适用条件**是 bootstrap 分布能近似
  抽样分布（需在报告中说明）；
- 排名置信集与两两同时区间属 **Phase 3**（Bonferroni 多重比较校正），单项 CI
  **不能**代替两两显著性检验。

### 6.5 校验与局限

- `validate_covariance()` 强制：方阵、有限、对称、对角非负、PSD（相对容差），
  **非 PSD 明确报错、不悄悄修补**；
- Monte Carlo 校准（`run_se_benchmark` / `scripts/run_se_validation.py`）：
  比较平均 bootstrap SE 与 $\hat\mu$ 的经验标准差（`se_ratio`），并报告单项 CI 覆盖率与
  MC 误差 $\sqrt{p(1-p)/R}$；
- **理论局限**：每轮重拟合 ML nuisance 的非参数 bootstrap，其推断质量依赖算法稳定性与
  正则条件；覆盖率必须用模拟检验（见 `docs/simulation_protocol.md`），不作通用推断保证；
  CI 覆盖的是**估计量抽样波动**，不覆盖识别假设违背带来的偏差；
- 重抽样单位为观测行（iid 假设）；重复客户/聚类结构需要 cluster bootstrap（未实现）。

## 7. 本版本**不**提供（Phase 3 及以后）

- Bonferroni 同时差值区间、可确认优劣关系、排名置信集（Phase 3）；
- AIPW / double robustness、Overlap Weighting（不同的目标总体！）、PSM、DML；
- 灵敏度分析（未观测混杂）、cluster bootstrap、max-T/stepdown 排名推断。

未启用推断时（`n_bootstrap=None`）：`result.se / ci_low / ci_high` 为 `NaN`，
`result.covariance / pairwise / ranks` 为 `None`，`result.inference_available = False`；
启用后 `se/ci/covariance` 填充，`pairwise / ranks` 仍为 `None`（Phase 3）。

## 8. 验证方法（为什么可信）

1. **手算例**：2 Treatment × 4 样本的 Hájek 与 HT 结果与实现逐位吻合（`tests/test_iptw.py`）；
   已知 draws 的手算协方差与正态 CI（`tests/test_covariance.py`）；
2. **不变性**（Hypothesis）：组内权重缩放不变、行置换不变、列对齐映射正确；
   Bootstrap：同种子逐位一致、`n_jobs` 并行与串行一致、每轮重拟合被调用计数验证；
3. **Oracle 对照**：在已知真值 $e_a(x)$ 的 DGP 上，IPTW 权重使加权 SMD 随 $n$ 下降、
   $\hat\mu_a \to \mu_a$；
4. **重复实验**：`tests/test_benchmark.py` 与 `scripts/run_simulation.py` 在多种子、
   多样本量下比较 Bias/RMSE（调整前 vs 调整后），偏差随 $n$ 收敛；
   `run_se_benchmark` / `scripts/run_se_validation.py` 校准 SE 与覆盖率；
5. **诊断有效性**：弱重叠/强混杂场景触发对应警告与门控降级，而非误报“可靠”。

Monte Carlo 误差按 $\sqrt{\hat p(1-\hat p)/R}$ 报告；有限模拟中 0.946 与 0.95 的差别
不构成“方法失效”的证据。
