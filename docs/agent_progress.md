# Agent 进度报告 — Phase 0 + Phase 1 + Phase 2 完成

> 日期：2026-10-12 · 规格：`PROJECT_SPEC.md` v0.2（§0–§9 权威）
> 结论：**Phase 0、Phase 1 与 Phase 2 已完成并通过全部质量检查**；排名推断（Phase 3）
> 按指令未实施，接口已预留。

---

## 1. 交付物：包目录树

```text
causalrank/
├── pyproject.toml               # uv + hatchling，src layout，依赖下限与 dev 组
├── uv.lock                      # 解析锁定的依赖集（可复现环境）
├── README.md                    # 产品定位、Quickstart、识别假设警示、仅研究用途
├── LICENSE                      # MIT（暂定，可按需改 Apache-2.0）
├── PROJECT_SPEC.md              # 规格副本（原文件 CausalRank_AGENT_DEVELOPMENT_SPEC.md）
├── .gitignore / .python-version
├── .github/workflows/tests.yml  # ruff + mypy + pytest + build + notebook 执行
├── src/causalrank/
│   ├── __init__.py              # 公共导出
│   ├── _version.py              # 单一版本源（hatch 动态版本）
│   ├── api.py                   # CausalRank.fit 全流程编排（含 Phase 2 推断集成）
│   ├── config.py                # CausalRankConfig（含全部门控阈值）+ 校验
│   ├── exceptions.py            # CausalRankError / CausalRankValidationError
│   ├── results.py               # EstimatesTable / DiagnosticsReport / CausalRankResult
│   ├── validation.py            # schema、标签规范化、dtype 归一化、缺失、分组、警告
│   ├── estimators/
│   │   ├── propensity.py        # 预处理 Pipeline、多分类 GBM、classes_ 对齐、OOF
│   │   └── iptw.py              # Hájek 主估计器、ESS、HT（内部核对）
│   ├── inference/
│   │   └── bootstrap.py         # 联合重拟合 Bootstrap、Σ 校验、normal/percentile CI
│   ├── diagnostics/
│   │   ├── weights.py           # 权重分位数、ESS、极端权重、裁剪统计
│   │   ├── overlap.py           # e_a(x)/e_{T}(X) 分布、共同支持检查
│   │   └── balance.py           # 数值/类别项 unweighted & weighted SMD（数值稳定）
│   └── simulation/
│       ├── dgp.py               # 8 场景 DGP + 独立 MC 参考 mu_true
│       └── benchmark.py         # Bias/RMSE/失败与门控统计 + SE/覆盖率校准
├── tests/                       # 119 个测试（详见 §4）
│   ├── conftest.py, __init__.py
│   ├── test_validation.py  test_propensity.py  test_iptw.py
│   ├── test_balance.py     test_diagnostics.py test_dgp.py
│   ├── test_properties.py  test_api.py         test_benchmark.py
│   ├── test_bootstrap.py   test_covariance.py
├── examples/
│   ├── 01_gbm_iptw_from_raw_data.ipynb   # 已执行、含真实输出
│   └── 02_joint_bootstrap.ipynb          # Phase 2：协方差/SE/失败阻断/校准演示
├── scripts/
│   ├── build_notebook.py        # 可复现地生成两个示例 notebook
│   ├── run_simulation.py        # 研究基准（Bias/RMSE/失败与门控统计）
│   ├── run_se_validation.py     # 研究基准（SE 校准与覆盖率）
│   └── repro_issues.py          # 缺陷复现脚本（审查问题的最小复现）
└── docs/
    ├── implementation_plan.md   # 架构/定义/测试策略/风险（先行计划）
    ├── methodology.md           # 估计目标、公式、假设、数值边界
    ├── assumptions.md           # 识别假设与软件红线
    ├── simulation_protocol.md   # DGP、真值、MC 误差、场景矩阵
    ├── api.md                   # API 参考与稳定性约定
    ├── references.md            # 文献与使用约定
    └── agent_progress.md        # 本文件
```

**未创建（按指令刻意排除）**：`inference/bootstrap.py`、`inference/pairwise.py`、
`inference/ranking.py`、`visualization/report.py`——Phase 2/3/4 的内容。

---

## 2. 精确定义与模型选择

### 2.1 估计目标

`mu_a = E_P[Y(a)]`（`estimand="population_mean"`），**所有 arm 在同一目标总体上定义**，
因此 arm 之间的比较可比。

### 2.2 倾向得分（预测模型）

- `e_a(x) = P(T=a|X=x)`，**联合多分类** `HistGradientBoostingClassifier`
  （非 one-vs-rest；与 McCaffrey et al. 2013 的差异已在 docs 声明）；
- 预处理：数值 `SimpleImputer(median)`；类别 `SimpleImputer(most_frequent|constant)` +
  `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`；整体 `Pipeline`，
  只在训练折 / Bootstrap 重抽样内 fit（无泄露）；
- 列对齐：`align_proba_columns()` 按 `model.classes_` 显式映射到排序后的公开标签；
- `cross_fit=m`：`StratifiedKFold(m, shuffle=True)` 产生 OOF 倾向得分，逐折保证全类；
- 默认 GBM 超参（在已知真值倾向面的 DGP 上调参得到）：
  `max_iter=400, learning_rate=0.02, max_depth=4, min_samples_leaf=60, l2_regularization=2.0`。
  **依据**：欠正则的树追逐标签噪声，使 in-sample `e_{T}` 系统性偏大（过拟合），
  使 OOF `e_{T}` 偶发极小值并引爆权重（实验：倾向 MAE 0.106→0.059，
  OOF 最小 ESS 比 0.10→0.60）。`gbm_params` 可覆盖。
- 裁剪：`clip_propensity=eps`（默认关）只对 `e_{T_i}` 做**下界**裁剪并报告比例；
  `e=0` 属 positivity 违背，直接报错。

### 2.3 Hájek IPTW（主估计器）

```
w_i            = 1 / e_{T_i}(X_i)
mu_a^H         = Σ 1{T_i=a} w_i Y_i / Σ 1{T_i=a} w_i
ESS_a          = (Σ w_ia)² / Σ w_ia² ,  w_ia = 1{T_i=a} w_i
```

HT 形式 `(1/n) Σ 1{T_a} Y_i/e_a` 仅内部核对（`horvitz_thompson_estimate`），不对外暴露；
禁止“组内归一化权重 + 除以 n”的混合算法（两个函数独立实现、独立手算测试）。

### 2.4 SMD（平衡）

- 项：数值协变量本身；类别每个 level 的 0/1 指示；含缺失的特征加 `__missing__` 指示项；
- `SMD_ab = (m_a - m_b)/s_ref`，`s_ref = sqrt((s_a²+s_b²)/2)`，`s_g` 为组内**未加权**
  ddof=1 标准差（分母不受权重影响，Austin 2009）；缺失逐项剔除（不污染均值/方差）；
- 输出每项跨 pair 的 `max|SMD|`（加权/未加权）与超阈值 pair 数；`s_ref=0` 的边界处理明确。

### 2.5 联合 Bootstrap 协方差（Phase 2）

- 每轮：**同一个**行索引向量重抽样 → 重新执行校验/插补/编码/GBM（含重新折划分）/权重/全部
  K 组均值，得 `mu^(b)`；
- `Sigma = np.cov(draws, rowvar=False, ddof=1)`，**不**再除以 n（有测试显式断言）；
- 种子：`SeedSequence(base_seed).spawn(B)` 逐轮派生，驱动重抽样+模型随机性 →
  同种子逐位一致、与执行顺序和 `n_jobs` 无关（测试覆盖 n_jobs=2 与串行逐位一致）；
- 失败轮记录 `{round, seed, reason}` 并排除；成功率 < 0.9 或成功轮 < 30 → 阻断推断；
- 单项 CI：`normal`（mu ± z·sqrt(Sigma_aa)）或 `percentile`（保留 draws）；
- `validate_covariance()`：对称/对角非负/PSD 校验，非 PSD 明确报错不修补。

---

## 3. 测试：命令与数值结果

### 3.1 全部工程检查（修订版，含上一版失实更正）

> **更正声明**：上一版报告写“Ruff 全通过”，该结论是在 `scripts/` 目录创建**之前**取得的，
> 而提交前的最终目录树里有 64 项 RUF001（`scripts/build_notebook.py` 的中文全角标点）
> 与 1 项 F401（Notebook 中未使用的 `import numpy`），CI 会在 lint 关卡失败。
> 该表述失实，已修正：中文文本按文件豁免 RUF001（src/tests 保持启用）、
> 清理未使用导入，并在**最终目录树上**重跑全部检查。以下为修订后的实测结果。

| 检查 | 命令 | 结果 |
|---|---|---|
| 单元+端到端测试 | `uv run pytest -q` | **119 passed**（约 4 分钟；含 2 个 `slow` 基准测试） |
| Lint | `uv run ruff check .` | **All checks passed**（对最终目录树实测） |
| 类型 | `uv run mypy` | **Success: no issues found in 19 source files** |
| 打包 | `uv build` | `causalrank-0.1.0.tar.gz` + `py3-none-any.whl` 成功 |
| 干净安装 | 新 venv 安装 wheel 后端到端 `fit` | 通过（`clean install OK 0.1.0`） |
| Python 3.11 兼容 | 3.11.15 + pandas 3.0.6 venv 安装 wheel 运行 `-m "not slow"` | **95 passed**（Phase 1 基线；CI 覆盖两版本全量） |
| Notebook | `uv run jupyter nbconvert --to notebook --execute` | 01/02 全 cell 无错误执行 |
| CI | `.github/workflows/tests.yml` | lint+type+test+build+notebook，3.11/3.12 矩阵（`uv sync --python` 显式指定并打印解释器版本） |

### 3.2 手算例（独立计算 vs 实现）

2 Treatment × 4 样本（`tests/test_iptw.py`）：

| 量 | 手算值 | 实现 |
|---|---|---|
| 权重 | [1.25, 2.5, 10/7, 2.5] | 逐位一致（rel 1e-12） |
| `mu_A` | `1.25/3.75 = 1/3` | 一致 |
| `mu_B` | `(10/7)/(10/7+2.5) = 4/11` | 一致 |
| `ESS_A` | `3.75²/(1.25²+2.5²) = 1.8` | 一致 |
| `mu_A^HT` | `0.25/0.8 = 0.3125` | 一致 |
| `mu_B^HT` | `0.25/0.7` | 一致 |

SMD 手算（`tests/test_balance.py`）：x=[1,3] vs [2,6] → `s_ref=√5`、
`SMD_unw = -2/√5`、加权权重 [1,3]×[1,1] → `SMD_w = -1.5/√5`；类别 level-a 指示 → `SMD=-1.0`。
均与实现一致（rel 1e-12）。

### 3.3 测试清单对照（规格 §7）

| 条目 | 覆盖 |
|---|---|
| 7.1 缺列/非 0/1/缺失 T/单类 T/标签稳定排序/稀有组/缺失 X | `test_validation.py`（19 项，含 int/str/混合/bool/整数浮点标签） |
| 7.2 倾向行和=1、classes_ 对齐、手算 Hájek、缩放不变、极端权重诊断、不用 AUC | `test_propensity.py` + `test_iptw.py` + `test_diagnostics.py` |
| 7.3 Bootstrap（**Phase 2**） | 未实施；协议已写入 docs，`n_bootstrap/n_jobs/confidence_level` 配置位已留 |
| 7.4 排名推断（**Phase 3**） | 未实施；`result.ranks/pairwise/covariance` 占位为 None |
| 7.5 Oracle 随机化对照 / 混杂下 raw vs adjusted / MC 误差与预算 / README 假设 / 合成数据 | `test_dgp.py`、`test_api.py`、`test_benchmark.py`、`docs/simulation_protocol.md`、README |

### 3.4 不变性（Hypothesis，`test_properties.py`）

- Hájek 均值对组内权重乘常数不变（随机 property）；
- 行置换不变；
- `align_proba_columns` 对任意 classes_ 排列正确映射。

---

## 4. DGP 真值 vs 估计 `mu_a`（不是分类指标）

真值：`mu_true[a] = E_X[P(Y(a)=1|X)]`，独立 MC 参考 `n_ref=200,000`、`seed=20260101`，
MC 标准误 ≈ 0.00027–0.00028。

### 4.1 单次端到端（Notebook `01`，confounded, n=4000, cross_fit=5）

| arm | `mu_hat` | `mu_true` | 调整后误差 | `raw_mean` | 原始误差 |
|---|---|---|---|---|---|
| A | 0.4551 | 0.4367 | +0.0184 | 0.4920 | +0.0553 |
| B | 0.5218 | 0.5185 | +0.0033 | 0.4585 | −0.0600 |
| C | 0.3927 | 0.4024 | −0.0097 | 0.4221 | +0.0197 |

平均绝对误差：**调整后 0.0105 vs 原始 0.045**。

### 4.2 重复实验基准（`scripts/run_simulation.py`，R=20，cross_fit=5）

| n | arm | bias(IPTW) | RMSE | bias(raw) | RMSE(raw) | 平均 ESS |
|---|---|---|---|---|---|---|
| 1500 | A | +0.0043 | 0.0137 | +0.0365 | 0.0396 | 468 |
| 1500 | B | −0.0020 | 0.0313 | −0.0720 | 0.0741 | 217 |
| 1500 | C | −0.0103 | 0.0261 | +0.0266 | 0.0329 | 322 |
| 6000 | A | −0.0032 | 0.0096 | +0.0267 | 0.0279 | 1977 |
| 6000 | B | +0.0154 | 0.0227 | −0.0644 | 0.0658 | 887 |
| 6000 | C | −0.0041 | 0.0115 | +0.0286 | 0.0310 | 1376 |

结论（含 MC 误差的谨慎表述）：
- 原始组均值偏差在 n 增大 4 倍后**不消失**（0.027–0.072）——这是混杂偏差；
- IPTW 偏差整体随 n 收敛到 0 附近（0.003–0.015），RMSE 同步下降；
- arm B（最稀有、权重最分散）收敛最慢，**如实保留**：有限样本下 Hájek 估计受
  极端权重影响，属方法内在的方差—偏差权衡，非实现缺陷（oracle 权重对照实验已验证）。
- 快测试（R=6）断言方向性结论：bias 随 n 下降、调整优于未调整、随机化场景双无偏。

**运行级质量统计**（同一次 R=20 基准，`runs_summary`；拟合失败率 0/40）：

| n | gate pass | gate caution | gate fail | 平均警告数 | low_propensity 触发率 | residual_imbalance 触发率 | 平均 min e_obs | 平均 ESS 比最小值 |
|---|---|---|---|---|---|---|---|---|
| 1500 | 50% | 50% | 0% | 0.60 | 10% | 50% | 0.025 | 0.47 |
| 6000 | 65% | 35% | 0% | 0.45 | 25% | 20% | 0.015 | 0.48 |

诊断行为与理论一致：残差失衡触发率随 n 从 50% 降到 20%（加权 SMD 均值 0.111→0.080），
失败率 0%；这些指标正是 Phase 2 Bootstrap 成功率门槛需要的经验依据。

### 4.3 SE 校准与覆盖率（Phase 2，`scripts/run_se_validation.py`）

设置：confounded，n=1200，R=16，B=100，cross_fit=5，`ci_method="normal"`，n_jobs=4
（耗时约 21 分钟）。真值：`mu_true` = (0.43670, 0.51845, 0.40239)。

| arm | bias | 经验 sd($\hat\mu$) | 平均 bootstrap SE | se_ratio | 95% CI 覆盖率 | 覆盖率 MC se |
|---|---|---|---|---|---|---|
| A | −0.0147 | 0.0271 | 0.0253 | **0.93** | 0.875 | 0.083 |
| B | +0.0168 | 0.0348 | 0.0385 | **1.11** | 1.000 | 0.000 |
| C | −0.0136 | 0.0346 | 0.0294 | **0.85** | 0.8125 | 0.098 |

结论（按 MC 误差谨慎表述）：
- `se_ratio` 0.85–1.11：联合 Bootstrap 的 SE 与 $\hat\mu$ 的真实抽样波动吻合在 ±15% 内，
  协方差校准**没有系统性偏小/偏大**；
- 覆盖率 0.81–1.00：R=16 时覆盖率的 MC 误差约 0.08–0.10，最低的 0.8125 距名义 0.95
  约 1.4 个 MC 标准误——**不能**据此断言欠覆盖，也不能宣称精确覆盖；更强的结论需要
  更大的 R（研究脚本支持，CI 内用 R=16/B=60 的慢测试锁定 0.55 < se_ratio < 1.6、
  coverage ≥ 0.80 的回归边界）；
- 快测试与慢测试共同构成 Phase 2 的统计验证门控；`run_se_validation.py` 的更大预算
  运行（如 R=40/B=200）留作发布前的研究级复核。

---

## 5. 诊断输出（Notebook `01`，confounded, n=4000）

**权重 / ESS**：

| arm | n | ESS | ESS 比 | w_min | w_max | 最大权重占比 |
|---|---|---|---|---|---|---|
| A | 1630 | 1374 | 0.843 | 1.25 | 13.75 | 0.3% |
| B | 1228 | 714 | 0.582 | 1.11 | 27.77 | 0.6% |
| C | 1142 | 949 | 0.831 | 1.61 | 19.89 | 0.5% |

**Overlap**（`e_{T}(X)` 最小值）：A 0.073 / B 0.036 / C 0.050，低于阈值比例均为 0。

**SMD**（未加权 → 加权）：x2 0.642→0.051；x1 0.557→0.032；x3 0.178→0.024；
x4 各 level 0.069→0.023 等。**质量门控：PASS（无警告）**——同时打印
“no warnings raised (this does NOT validate the causal assumptions)”。

**弱重叠场景（weak_overlap, n=3000）**：门控 **FAIL**，警告包括
`low_ess`（ESS≈29、ESS 比 0.021）、`extreme_weight`（单条权重占 17.4%）、
`low_propensity` / `weak_common_support`（目标样本中不可达区域占比高）——
这是预期的“诚实失败”形态：给警告、降门控，而非输出可靠排名文案。

---

## 6. 开发过程中发现并修复的工程问题

### 6.1 第一轮实现期间

1. **循环导入**：`api.py` 从包 `__init__` 取版本号 → 独立 `_version.py`（hatch 动态版本）。
2. **类对齐层级错误**：模型按 0..K-1 编码拟合，`classes_` 是编码而非标签 →
   新增 `aligned_predict_proba()`（编码→标签→列对齐两层映射），并用乱序 `classes_` 桩测试。
3. **SMD 被缺失值污染**（真实 bug）：数值协变量 NaN 使均值/方差变 NaN →
   重写为逐项 validity mask 的缺失感知统计，并为含缺失特征增加 `__missing__` 指示项。
4. **GBM 欠正则 → OOF 权重爆炸**：OOF 消除了过拟合偏差但偶发极小 `e_{T}`（ESS 比 0.10）→
   用已知真值倾向面调参确定默认正则强度（MAE 0.106→0.059、ESS 比→0.60），
   并把依据写入 `docs/methodology.md`。
5. **裁剪语义**：对 `e_{T}` 裁剪不应包含上界（`e→1` 权重有界）→ 只做下界裁剪；
   `e=0` 仍 fail fast。
6. `IPTWEstimate` 未携带 `t_idx` 使权重诊断无法按组聚合 → 字段补齐；
   `raw_mean/raw_means`、`weight_sum/weight_sums` 命名不一致 → 统一。

### 6.2 外部复核后的修复（第二轮，全部带回归测试）

| # | 级别 | 问题 | 修复 |
|---|---|---|---|
| 1 | P1 | **共同支持漏报警**：诊断只看被处理行的 `e_{T}` 与近确定性分配，(0.5,0.5,0)/(0,0.5,0.5)/(0.5,0,0.5) 这类“每臂都有不可达区域”的输入无告警、gate=pass | 新增 `frac_arm_below_threshold`（全目标样本低概率占比）；`e_a(X)=0` → critical `no_common_support`；占比 > `max_unsupported_share`(0.05) → `weak_common_support` |
| 2 | P1 | **常见 dtype 无法分析**：bool 列在 SimpleImputer 报错、pandas 3 默认字符串 dtype 被校验拒绝、含 `pd.NA` 的 object 列在拟合时 `TypeError: boolean value of NA is ambiguous` | `_is_categorical_dtype` 统一识别 object/StringDtype/category/bool/BooleanDtype；`_normalize_feature_frame` 把类别列归一为 object+`np.nan`、数值列归一为 float64（含可空 `Int64`）；端到端回归测试覆盖四类 dtype |
| 3 | P1 | **SMD 方差数值不稳定**：`Σx²−n·x̄²` 在大偏移量下灾难性相消，`[1,3] vs [2,6]` 平移 `+1e9` 后 SMD 变 `−inf` | 列中心化 + 两遍方差 `Σ(x−x̄)²`；回归测试覆盖 `+1e9`/`+1e12` 平移不变（加权与未加权） |
| 4 | P2 | **进度报告与实测不一致**：报告称 Ruff 全过，实测 65 项错误（该结论取得于 `scripts/` 创建之前） | 清理未使用导入；`scripts/*` 按文件豁免 RUF001（中文全角标点是有意排版，src/tests 保持启用）；在最终树上重跑并在 §3.1 记录更正 |
| 5 | P2 | **CI 双 Python 矩阵未落实**：只装解释器不同步绑定，`.python-version` 固定 3.12，两 job 可能都跑 3.12 | `uv sync --all-groups --python ${{ matrix.python-version }}` + 打印实际解释器；本地已用 3.11.15 实测 95 项通过 |
| 6 | P2 | **基准缺失败/诊断统计**：仅记 Bias/RMSE/ESS，fit 异常会中断整批 | `run_estimand_benchmark` 捕获失败轮（status/error），新增 `runs`（gate、warning codes、min e_obs、max weight share、min ESS ratio）与 `runs_summary`（失败率、门控分布、各警告触发率）；`summary` 增列 `n_ok/n_failed/failure_rate` |

---

## 7. 未满足 / 有意推迟的需求（如实披露）

| 需求 | 状态 |
|---|---|
| 联合 Bootstrap 协方差、SE/CI（规格 3.5、Phase 2） | **未实施**（按 §9 指令停止于 Phase 1）；`se/ci` 为 NaN，`covariance=None` |
| Bonferroni 排名置信集（规格 3.6、Phase 3） | **未实施**；`ranks/pairwise=None` |
| Bootstrap 失败率/稀有臂协议（3.5） | 协议写入 docs，代码待 Phase 2；基准层已能统计 fit 失败率作对照 |
| 聚类（cluster）bootstrap、重复 ID 检测 | 仅重复行提示（`duplicate_rows`）；无 ID 列接口 |
| 连续 outcome | 架构预留，v0.1 显式拒绝非 0/1 |
| 报告可视化（Phase 4） | 未实施（数值 API 不依赖绘图库，符合规格） |
| Monte Carlo **覆盖率**指标 | 需要推断量，Phase 2/3 后统计；当前只有 Bias/RMSE 与失败/门控/告警率 |
| 真实公开数据示例 | 未使用（避免许可证风险）；全部为合成数据 |
| 许可证选择 | 暂用 MIT，文件内注明可改 Apache-2.0，待决策 |
| **环境怪癖**：Python 3.11 + 非 ASCII 项目路径下 `pip install -e .` 失效 | 非包缺陷：CPython 3.11 的 `site.py` 按 locale 编码读 `.pth`，UTF-8 路径 `D:\workspace\因果排名\src` 被误解码而进不了 `sys.path`（3.12+ 正常）。规避：改用 wheel 安装、或把项目放在 ASCII 路径、或设 `PYTHONUTF8=1`。已在 3.11.15 + wheel 下实测 95 项通过 |

---

## 8. Phase 2 就绪 → 落实情况（本轮完成）

| 准备项 | 现状 |
|---|---|
| 单一可重放的估计协议 | ✅ `joint_bootstrap` 每轮调用 `fit_propensity` + `hajek_estimate` 纯函数链 |
| 每轮重拟合全部协议 | ✅ 校验/插补/编码/GBM/折划分全部在每轮内重跑（调用计数测试验证 B 轮 = B 次拟合） |
| 联合结构 | ✅ `B×K` draws → `np.cov(draws, rowvar=False, ddof=1)`；测试显式断言未除以 n |
| 随机数可复现并行 | ✅ `SeedSequence(base_seed).spawn(B)`；测试断言 n_jobs=2 与串行逐位一致 |
| 失败轮处理 | ✅ `{round, seed, reason}` 结构化报告；成功率/轮数门槛触发**推断阻断**（Notebook 02 实测 35/40 → 阻断） |
| 失败/诊断统计 | ✅ `runs_summary`（失败率、门控分布、告警触发率）+ bootstrap 成功/失败计数 |
| 结果 schema | ✅ `se/ci_*`、`covariance` 填充；`pairwise/ranks` 保留至 Phase 3 |
| 配置位 | ✅ `n_bootstrap/n_jobs/confidence_level/ci_method/bootstrap_*` 全部生效 |
| 验证基础设施 | ✅ 手算 Σ/CI 测试、不变性测试、SE 校准与覆盖率（§4.3） |

**Phase 2 交付摘要**：`inference/bootstrap.py`（重抽样/逐轮种子/失败记录）、
`validate_covariance`（非 PSD 显式报错不修补）、SE/normal+percentile CI 填充
`EstimatesTable`、`run_se_benchmark` SE/覆盖率校准、Notebook `02_joint_bootstrap.ipynb`
（含推断阻断实测演示）。**已知风险**（保留声明）：GBM nuisance + 重拟合 Bootstrap 的
理论表现依赖算法稳定性，覆盖率以模拟检验为准（§4.3），不作通用推断保证。

---

## 9. Phase 3 准备情况评估：**就绪**

| 准备项 | 现状 |
|---|---|
| 联合协方差 | ✅ `result.covariance`（K×K，标签规范序）已可直接用于差值 SE $\sqrt{\Sigma_{ii}+\Sigma_{jj}-2\Sigma_{ij}}$ |
| 相关性结构 | ✅ 非对角元素来自联合估计（Notebook 02 实测相关矩阵；手算差值 SE 测试已覆盖公式） |
| 显著性判定素材 | ✅ `result.effects` 提供 `mu/se/ci`；两两同时区间需加 Bonferroni 因子 $z_{1-\alpha/(2M)}$，$M=K(K-1)/2$ |
| 并列处理 | ⏳ 需定义 tie convention（规格 3.6）：真实并列下严格全序不唯一，点估计并列须显式标记 |
| 多重比较 | ⏳ `inference/pairwise.py`、`inference/ranking.py` 待建；`result.pairwise/ranks` 字段已就位 |
| 验证素材 | ✅ `close_effects`/`equal_effects` 场景与联合覆盖率统计框架可直接复用 |

**Phase 3 建议顺序**：(1) `inference/pairwise.py`：Bonferroni 同时差值区间 + 方向判定；
(2) `inference/ranking.py`：$[1+B_i,\, K-W_i]$ 保守排名区间 + 并列约定；
(3) 不变性测试（Treatment 重排、共同平移）与 K=2 退化检验；
(4) `equal_effects` 场景验证“不得凭噪声宣称确定排名”；(5) Notebook 03 + 最终排名覆盖率基准。
**红线**：不以差异不显著作为效果相等的证据；不把 Bootstrap 第一频次当最优概率；
单项 CI 重叠不代替显著性检验。

---

## 10. 复现方式

```bash
uv sync                              # 安装 runtime + dev 依赖
uv run pytest -q                     # 119 tests
uv run ruff check . && uv run mypy   # 静态检查
uv build                             # 打包检查
uv run jupyter nbconvert --to notebook --execute --inplace examples/01_gbm_iptw_from_raw_data.ipynb
uv run python scripts/run_simulation.py --scenario confounded --n 1500 6000 --replications 20 --cross-fit 5
uv run python scripts/run_se_validation.py --scenario confounded --n 1200 --replications 16 --bootstrap 100 --cross-fit 5 --n-jobs 4
```

环境：Python 3.11/3.12（本地验证 3.12.7）、numpy 2.x、pandas 2.x、scipy 1.x、
scikit-learn ≥1.4（本地 1.9.1）；版本自动记录于 `result.metadata["versions"]`。
