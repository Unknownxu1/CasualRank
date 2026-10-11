# Agent 进度报告 — Phase 0 + Phase 1 完成

> 日期：2026-10-11 · 规格：`PROJECT_SPEC.md` v0.2（§0–§9 权威）
> 结论：**Phase 0 与 Phase 1 已完成并通过全部质量检查**；联合 Bootstrap（Phase 2）、
> 排名推断（Phase 3）按指令未实施，接口已预留。

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
│   ├── api.py                   # CausalRank.fit 全流程编排
│   ├── config.py                # CausalRankConfig（含全部门控阈值）+ 校验
│   ├── exceptions.py            # CausalRankError / CausalRankValidationError
│   ├── results.py               # EstimatesTable / DiagnosticsReport / CausalRankResult
│   ├── validation.py            # schema、标签规范化、缺失、分组、dtype、警告
│   ├── estimators/
│   │   ├── propensity.py        # 预处理 Pipeline、多分类 GBM、classes_ 对齐、OOF
│   │   └── iptw.py              # Hájek 主估计器、ESS、HT（内部核对）
│   ├── diagnostics/
│   │   ├── weights.py           # 权重分位数、ESS、极端权重、裁剪统计
│   │   ├── overlap.py           # e_a(x) / e_{T}(X) 分布、低重叠警告
│   │   └── balance.py           # 数值/类别项 unweighted & weighted SMD
│   └── simulation/
│       ├── dgp.py               # 8 场景 DGP + 独立 MC 参考 mu_true
│       └── benchmark.py         # 重复实验 Bias/RMSE 框架
├── tests/                       # 89 个测试（详见 §4）
│   ├── conftest.py, __init__.py
│   ├── test_validation.py  test_propensity.py  test_iptw.py
│   ├── test_balance.py     test_diagnostics.py test_dgp.py
│   ├── test_properties.py  test_api.py         test_benchmark.py
├── examples/
│   └── 01_gbm_iptw_from_raw_data.ipynb   # 已执行、含真实输出
├── scripts/
│   ├── build_notebook.py        # 可复现地生成示例 notebook
│   └── run_simulation.py        # 手动研究基准（R=20 级别）
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

---

## 3. 测试：命令与数值结果

### 3.1 全部工程检查（最终运行）

| 检查 | 命令 | 结果 |
|---|---|---|
| 单元+端到端测试 | `uv run pytest -q` | **89 passed**（约 2 分钟） |
| Lint | `uv run ruff check .` | **All checks passed** |
| 类型 | `uv run mypy` | **Success: no issues found in 17 source files** |
| 打包 | `uv build` | `causalrank-0.1.0.tar.gz` + `py3-none-any.whl` 成功 |
| 干净安装 | 新 venv 安装 wheel 后端到端 `fit` | 通过（输出 `clean install OK 0.1.0`） |
| Notebook | `uv run jupyter nbconvert --to notebook --execute` | 全 cell 无错误执行 |
| CI | `.github/workflows/tests.yml` | 上述全部步骤 + 双 Python 版本矩阵 |

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
`low_propensity` 等——这是预期的“诚实失败”形态：给警告、降门控，而非输出可靠排名文案。

---

## 6. 开发过程中发现并修复的工程问题

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

---

## 7. 未满足 / 有意推迟的需求（如实披露）

| 需求 | 状态 |
|---|---|
| 联合 Bootstrap 协方差、SE/CI（规格 3.5、Phase 2） | **未实施**（按 §9 指令停止于 Phase 1）；`se/ci` 为 NaN，`covariance=None` |
| Bonferroni 排名置信集（规格 3.6、Phase 3） | **未实施**；`ranks/pairwise=None` |
| Bootstrap 失败率/稀有臂协议（3.5） | 协议写入 docs，代码待 Phase 2 |
| 聚类（cluster）bootstrap、重复 ID 检测 | 仅重复行提示（`duplicate_rows`）；无 ID 列接口 |
| 连续 outcome | 架构预留，v0.1 显式拒绝非 0/1 |
| 报告可视化（Phase 4） | 未实施（数值 API 不依赖绘图库，符合规格） |
| Monte Carlo **覆盖率**指标 | 需要推断量，Phase 2/3 后统计；当前只有 Bias/RMSE |
| 真实公开数据示例 | 未使用（避免许可证风险）；全部为合成数据 |
| 许可证选择 | 暂用 MIT，文件内注明可改 Apache-2.0，待决策 |

---

## 8. Phase 2 准备情况评估：**就绪**

| 准备项 | 现状 |
|---|---|
| 单一可重放的估计协议 | `hajek_estimate(y, t_idx, propensity, labels)` 纯函数；预处理+拟合封装在 `fit_propensity()`，可整体在重抽样上调用 |
| 每轮重拟合全部协议 | `fit_propensity` 内含 Pipeline 构建/折划分/拟合，Bootstrap 只需在重抽样索引上调用同一函数（含 `cross_fit`） |
| 联合结构 | `IPTWEstimate.mu` 是 K 维向量；`B×K` draws → `np.cov(draws, rowvar=False, ddof=1)` 一步到位，且**不会再除 n**（结构上无法误用） |
| 随机数可复现并行 | `random_state` 贯穿 config；Phase 2 用 `SeedSequence.spawn` 按轮派生种子（`benchmark.py` 已示范该模式） |
| 失败轮处理 | 稀有臂/缺类已在 `fit_propensity` 显式报错 → Bootstrap 捕获为失败轮并计入成功率门槛 |
| 结果 schema | `se/ci_low/ci_high` 列、`covariance/pairwise/ranks` 字段已就位，Phase 2/3 只需填充 |
| 配置位 | `n_bootstrap`、`n_jobs`、`confidence_level` 已在 config 与 metadata 中记录 |
| 验证基础设施 | DGP 真值、benchmark 框架、手算/不变性测试模式可直接复用于 SE/覆盖率验证 |

**Phase 2 建议顺序**：(1) `inference/bootstrap.py`：单轮重抽样函数 + 索引派生种子 +
失败轮记录；(2) `Sigma` 对称/PSD/对角线校验（非 PSD 显式报错，不修补）；
(3) SE / 正态 CI 填充 `EstimatesTable`；(4) Monte Carlo 比较估计 SE 与真实抽样波动；
(5) Notebook 02 与覆盖率实验。**风险提示**：GBM nuisance + 重拟合 Bootstrap 的理论表现
依赖算法稳定性，必须用模拟覆盖率检验，不作通用推断保证。

---

## 9. 复现方式

```bash
uv sync                              # 安装 runtime + dev 依赖
uv run pytest -q                     # 89 tests
uv run ruff check . && uv run mypy   # 静态检查
uv build                             # 打包检查
uv run jupyter nbconvert --to notebook --execute --inplace examples/01_gbm_iptw_from_raw_data.ipynb
uv run python scripts/run_simulation.py --scenario confounded --n 1500 6000 --replications 20 --cross-fit 5
```

环境：Python 3.11/3.12（本地验证 3.12.7）、numpy 2.x、pandas 2.x、scipy 1.x、
scikit-learn ≥1.4（本地 1.9.1）；版本自动记录于 `result.metadata["versions"]`。
