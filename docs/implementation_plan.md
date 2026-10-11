# CausalRank — 实施计划（Phase 0 + Phase 1）

> 依据 `PROJECT_SPEC.md`（原文件名 `CausalRank_AGENT_DEVELOPMENT_SPEC.md`）第 0–8 节。
> 本文件为执行契约 §0.2 要求的“先计划、后实施”文档；实现中若有偏离，会在本文件与
> `docs/agent_progress.md` 中显式记录。

## 0. 范围与非目标

**本轮范围（Phase 0 + Phase 1）**

1. 可安装的 `src/` 布局 Python 包（Python ≥ 3.11，uv + pyproject，MIT 许可证暂定）。
2. 输入 schema 校验、缺失值/编码预处理 Pipeline（全部在训练折内 fit，杜绝泄露）。
3. 多分类 GBM 倾向得分（sklearn `HistGradientBoostingClassifier` + `ColumnTransformer`），
   含 `classes_` 列对齐与可选 out-of-fold（`cross_fit`）估计。
4. Hájek 形式多 Treatment IPTW 估计 `mu_a = E[Y(a)]`（同一目标总体）。
5. 权重分布 / ESS / overlap / 加权前后 SMD 诊断 + 可配置质量门控警告。
6. 已知 `mu_true` 的可配置合成 DGP（独立大规模 Monte Carlo 参考），含 8 个场景预设。
7. 单元测试（含手算例、不变性测试、端到端 DGP 检查）、示例 Notebook、理论文档、CI。

**本轮不做（明确排除，见规格 §6 Later）**：联合 Bootstrap、协方差/SE/CI、排名置信集、
AIPW、Overlap Weighting、PSM、DML、可视化报告、前后端、云服务、数据库。
架构上为 Phase 2/3 预留接口（`result.covariance` / `pairwise` / `ranks` 位置、
`inference/` 包、`n_bootstrap` 配置项），但本轮一律不计算推断量，返回对象显式标注
`inference_available=False`。

## 1. 模块结构

```text
src/causalrank/
├── __init__.py          # 公共导出：CausalRank, CausalRankConfig, 结果类型, 异常
├── api.py               # CausalRank 高层入口（fit(df) -> CausalRankResult）
├── config.py            # CausalRankConfig：全部可配置阈值与选项 + 校验
├── exceptions.py        # CausalRankError / CausalRankValidationError
├── results.py           # EstimatesTable / DiagnosticsReport / CausalRankResult / DiagnosticWarning
├── validation.py        # schema、标签规范化、缺失、分组、dtype 判定、警告
├── estimators/
│   ├── propensity.py    # 预处理 Pipeline、多分类 GBM、列对齐、OOF 交叉拟合
│   └── iptw.py          # Hájek 权重与 mu_hat；HT 仅作内部理论核对
├── diagnostics/
│   ├── weights.py       # 权重分位数、ESS、top-weight 占比、裁剪统计
│   ├── overlap.py       # 实际 Treatment 概率分布、低重叠警告
│   └── balance.py       # 数值/类别项的 unweighted & weighted SMD
└── simulation/
    ├── dgp.py           # 可配置 DGP + 独立 MC 参考 mu_true
    └── benchmark.py     # 重复实验：Bias / RMSE / 收敛检查（研究脚本 + 慢测试）
tests/                   # 手算例、不变性（Hypothesis）、诊断、DGP、端到端
examples/01_gbm_iptw_from_raw_data.ipynb
docs/{implementation_plan,agent_progress,methodology,assumptions,simulation_protocol,api,references}.md
.github/workflows/tests.yml
```

`inference/` 与 `visualization/` 在本轮不创建占位空包（规格允许按阶段落地）；
Phase 2 起补 `inference/bootstrap.py`。

## 2. 精确定义（与 docs/methodology.md 一致）

### 2.1 估计目标

- `mu_a = E_P[Y(a)]`，`a = 1..K`，`P` = 输入样本代表的样本总体（ATE/平均潜在结果目标）。
- 前提：一致性、条件可交换性、positivity、SUTVA、iid 抽样；软件只能检查部分经验表现。

### 2.2 倾向得分（预测模型，不是因果方法）

- `e_a(x) = P(T=a | X=x)`，联合多分类模型（**非** one-vs-rest），
  sklearn `HistGradientBoostingClassifier` 处理多类 `predict_proba`。
- 预处理：数值 `SimpleImputer(median)`；类别 `SimpleImputer(most_frequent|constant)` +
  `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`；整体包在 `Pipeline` 中，
  **只在训练折 / Bootstrap 重抽样内 fit**。
- 类别对齐：`align_proba_columns()` 显式按 `model.classes_` → 公开标签规范序（排序后的
  唯一标签）重排列，禁止按列位置假设。
- `cross_fit=m`：`StratifiedKFold(m, shuffle=True, random_state=...)`，保证每折训练集含全部
  K 类；OOF 倾向得分用于权重，以降低同一数据拟合与评估造成的过拟合偏差。
- 概率裁剪 `clip_propensity=eps` 仅在显式配置时执行，只对**实际 Treatment 概率**
  （权重分母）做下界裁剪 `e <- max(e, eps)`，并报告被裁剪比例；默认不裁剪；
  概率恰为 0 属于 positivity 违背，仍直接报错而不是被裁剪掩盖。

### 2.3 Hájek IPTW（主估计器）

- `w_i = 1 / e_{T_i}(X_i)`；
- `mu_a^H = Σ_i 1{T_i=a} w_i Y_i / Σ_i 1{T_i=a} w_i`；
- `ESS_a = (Σ_i w_ia)^2 / Σ_i w_ia^2`，`w_ia = 1{T_i=a} w_i`；
- HT 版本 `mu_a^HT = (1/n) Σ_i 1{T_i=a} Y_i / e_a(X_i)` 仅内部理论核对，不对外暴露；
- 二元 outcome 下 Hájek 结果必落在 [0,1]；对组内权重乘常数不变。

### 2.4 SMD（平衡诊断）

- 数值特征：项即特征本身；类别特征：每个 level 的 0/1 指示变量为一项。
- 对每个无序 Treatment 对 `(a,b)`：
  `SMD_ab = (m_a - m_b) / s_ref`，其中 `m_g` 为组内（加权或未加权）均值，
  `s_ref = sqrt((s_a^2 + s_b^2)/2)`，`s_g^2` 为**未加权**组内样本方差（ddof=1），
  即分母使用与权重无关的 pooled reference SD（Austin 2009 建议），避免“加权改变分母”。
- 加权均值用 Hájek 组内归一化：`m_a^w = Σ_{T=a} w_i x_i / Σ_{T=a} w_i`。
- 汇报 `max|SMD|`（weighted 与 unweighted）、超过阈值（0.1 / 0.25）的项数。
- `s_ref = 0` 且均值不同 → `±inf` 并发 `degenerate_smd` 警告；`s_ref = 0` 且均值相同 → 0。

### 2.5 质量门控（warnings）

`quality_gate ∈ {pass, caution, fail}`，由警告级别决定；**不**输出“可靠因果排序”类文案。
警告码：`small_group`、`rare_treatment`、`low_propensity`、`extreme_weight`、`low_ess`、
`residual_imbalance`、`severe_imbalance`、`duplicate_rows`、`high_cardinality_categorical`、
`missing_covariates`、`propensity_clipped`、`degenerate_smd`、`label_dtype_cast`。

## 3. DGP 设计（simulation/dgp.py）

- 协变量：`x1,x2 ~ N(0,1)`、`x3 ~ Bernoulli(p)`、`x4 ~ Categorical(3 levels)`；可选 MCAR 缺失。
- Treatment：多项 softmax，logit = `intercept_a + coef_a·z + overlap_strength·ov_coef_a·x1`；
  `confounding_strength` 缩放 `coef_a`；`rare_treatment` 通过基线概率实现。
- 潜在结果：`P(Y(a)=1|X) = sigmoid(beta0 + beta·z + tau_a + interaction·x1·tau_a)`；
  观测 `Y = Y(T)`。
- `mu_true[a] = E_X[P(Y(a)=1|X)]` 用**独立种子**的大规模 MC 参考样本（默认 200k）估计，
  同时给出 MC 标准误 `sd/sqrt(n_ref)` 与参考 seed（记录在数据集对象与文档中）。
- 场景预设（8 个）：randomized、confounded、strong_confounding、weak_overlap、
  rare_treatment、equal_effects、close_effects、mixed_missing。
- `simulation/benchmark.py`：重复实验计算 Bias / RMSE / 与 `mu_true` 的差距，并比较
  调整前后偏差（原始组均值 vs IPTW）。

## 4. 测试策略

| 层 | 内容 |
|---|---|
| 校验 | 7.1 全部条目：缺列、非 0/1 Y、缺失 T、单类 T、标签稳定排序、稀有组、缺失 X |
| 倾向 | 行和=1、非负有限、`classes_` 对齐（含乱序 classes_ 的桩测试）、OOF 折含全类、固定种子可复现 |
| IPTW | 2 Treatment × 4 样本手算 Hájek 与 HT；组内权重缩放不变性（Hypothesis）；行置换不变性 |
| 诊断 | 手算 SMD（数值+类别）；ESS 公式；极端权重/低 ESS/低重叠触发警告；quality_gate 分级 |
| DGP | `mu_true` 可重复；场景预设正确性；缺失值；MC 参考误差量级 |
| 端到端 | randomized 场景 IPTW ≈ 原始均值 ≈ `mu_true`；confounded 场景调整后偏差 < 原始偏差（多种子平均，确定性种子）；收敛：n 增大偏差下降（slow） |
| 工程 | ruff、mypy、pytest、`python -m build`、`nbconvert --execute` 全绿 |

Monte Carlo 误差按 `sqrt(p(1-p)/R)` 报告；快测试与研究基准分开（`slow`/`research` marker）。

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| GBM 过拟合倾向得分导致权重极端 | 提供 `cross_fit` OOF；诊断 min/分位数/ESS；显式裁剪选项（默认关） |
| 稀有 Treatment / 折内缺类 | 训练前硬校验 `n_a >= cross_fit`；逐折防御性检查 |
| 类别高基数 one-hot 爆炸 | `max_category_cardinality` 警告；文档建议合并水平 |
| all-NaN 协变量列导致 sklearn 丢列 | 校验阶段警告缺失率；文档说明 |
| 把 AUC 当因果有效证据 | 文档与 summary 显式拒绝；不输出模型 AUC 作为质量指标 |
| 推断被误认为已完成 | `inference_available=False`、`se/ci` 为 NA、summary 显式声明 Phase 2 才有联合 Bootstrap |
| Bootstrap（Phase 2）失败率高 | 本轮只预留结构与 `n_bootstrap` 配置；协议写入 docs |

## 6. 里程碑

1. **M0**：包骨架 + `uv sync` + smoke test 通过（Phase 0 验收）。
2. **M1**：validation + propensity + IPTW + 手算测试通过。
3. **M2**：诊断 + 质量门控 + DGP + 场景验证测试通过。
4. **M3**：Notebook 可执行、文档齐全、lint/type/build 全绿、CI 配置完成、进度报告。
5. **M4（Phase 2）**：联合 Bootstrap 协方差 + SE/CI + 失败轮披露/推断阻断 +
   SE/覆盖率 Monte Carlo 校准 + Notebook 02。

## 7. Phase 2 实施增补（已完成）

- **模块**：`src/causalrank/inference/bootstrap.py`（联合重抽样、每轮重拟合、
  种子派生、失败报告、Σ 校验、normal/percentile CI）；
- **集成**：`CausalRank.fit` 在 `n_bootstrap >= 1` 时产出 `covariance / se / ci_*`；
- **关键取舍**：
  1. `n_bootstrap` 默认 `None`（显式开启推断）。原因：每轮重拟合全部协议，
     300 轮会把秒级 fit 变成分钟级；无提示地慢 300 倍比显式开关更糟。
     规格示例本身显式传 `n_bootstrap=300`，不受默认值影响。
  2. 逐轮种子用 `SeedSequence(base_seed).spawn(B)`，同时驱动重抽样与模型随机性
     → 与执行顺序、`n_jobs` 无关（有测试断言 n_jobs=2 与串行逐位一致）。
  3. 失败轮（稀有臂漏抽等）记录 `{round, seed, reason}` 并排除；成功轮数/成功率
     低于门槛时**阻断推断**（critical 警告 + 原因清单），而非输出可疑协方差。
  4. `validate_covariance()` 对非对称/负对角/非 PSD **明确报错**，不做数值修补。
  5. joblib 仅用于轮间并行（新增显式依赖），单线程路径不依赖它。
