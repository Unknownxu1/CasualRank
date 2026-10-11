# CausalRank — 从原始数据到多 Treatment 因果排名的 Python 包

> **版本**：开发规格 v0.2 · 2026-10-11  
> **文档用途**：放入新建 Git 仓库作为 `PROJECT_SPEC.md`，由编码 Agent 按阶段自主实施。  
> **开发路线**：**原始数据 → GBM 多分类倾向得分 → IPTW → 效果估计 + 联合 Bootstrap → 可靠排名**。  
> **首要原则**：统计正确性、可复现性、可诊断性优先于 UI 与功能数量。  
> **数据边界**：只用自行编写的代码、合成数据和许可清晰的公开数据；不得复制公司数据、内部代码、密钥或业务配置。

---

## 0. Agent 必读：执行契约

你是本项目的 **Senior Statistical Software Engineer + Applied Causal Inference Researcher**。目标是独立开发并验证可安装的 Python 包，而不是只写 Notebook 或画一张排序图。

**自主执行规则**：
1. 首先检查仓库现状；如果为空，初始化一个可安装的 `src/` 布局包；如果存在代码，先理解再修改。
2. 先输出一份简短的 `docs/implementation_plan.md`（模块、方法、验证方案、风险），随后立即依计划实施；不要因非关键问题持续反问。
3. 每完成一个模块，写对应单元测试；每一阶段运行测试与静态检查，失败时修复，再进入下一阶段。
4. 所有统计方法必须在 `docs/methodology.md` 写出 **估计目标、公式、适用条件、假设、数值边界**。不能仅凭“跑通”判定正确。
5. 不得把 AUC 高当作因果估计可靠的依据；不通过平衡、重叠、稳定性诊断时必须给出警告，禁止把排名包装成可靠因果结论。
6. 不要在第一阶段实现 AIPW、OW、PSM、DML、LLM、前后端分离、云服务、数据库，也不要为了复杂而设置微服务。
7. 优先使用原生 Python、NumPy、pandas、SciPy、scikit-learn；只在确有理由时增加依赖。
8. 不得伪造实验结果或引用；报告里明确区分“模型估计”“统计推断”和“因果识别条件”。
9. 不要把已经成熟的排名推断方法宣称为原创；可参考 `csranks` 的公开论文做独立实现和对照，但遵守许可证。
10. 记录阶段进度到 `docs/agent_progress.md`（已完成 / 测试 / 遗留问题 / 后续步骤）。

**完成定义**：安装包、端到端 Demo、自动测试、理论文档、质量门控均完成；不能只靠 Agent 自评。

---

## 1. 产品定位

### 1.1 产品一句话

**CausalRank**：End-to-end Python framework for multi-treatment causal outcome estimation, joint uncertainty quantification, and statistically defensible treatment ranking from observational data.

### 1.2 用户输入与输出

用户提供一个 DataFrame：
- `X`：Treatment 发生之前测量的协变量（数值、分类均可）；
- `T`：互斥的 K 类 Treatment（K ≥ 2，V0.1 最多支持 10 个作为测试范围，并非数学上限）；
- `Y`：Outcome，V0.1 先支持二分类 0/1，架构应便于未来扩展连续型；
- 变量语义、目标总体、置信水平、随机种子等配置。

用户得到：
1. 每条 Treatment 的 **同一目标总体上的平均潜在结果** `mu_a = E[Y(a)]` 的估计值；
2. 点估计排名、两两差值；
3. 联合估计量协方差 `Sigma`、SE、单项置信区间；
4. 多重比较校正后的**排名置信区间**和能够确认的优劣关系；
5. 权重、重叠、平衡、有效样本量、Bootstrap 稳定性等诊断；
6. 机器可读 DataFrame/JSON 与可复现的可视化报告（报告在后期）。

### 1.3 不解决的问题

- 自动从数据判定哪些变量是混杂、工具、中介或碰撞变量；
- 自动解决未观测混杂、违反一致性 / 可交换性 / positivity 的问题；
- 直接保证个体层最优 Treatment 或最大化利润；
- 无条件从任何数据得出“因果排序”。

### 1.4 统计目标：同一总体比较的关键

对 Treatment `a`，设潜在结果 `Y(a)`，目标为：

\[\mu_a=E_{\mathcal P}[Y(a)]\quad(a=1,\ldots,K).\]

其中 `P` 为输入样本所代表的目标总体（此项目 v0.1 使用样本总体 ATE/平均潜在结果目标）；输入应具有一致的 Outcome 定义与观察窗口。

只有满足 **一致性（consistency）、条件可交换性（无遗漏的混杂）及 positivity（足够重叠）** 等条件，才可以把 IPTW 结果解释为因果均值。软件可以检查部分经验表现，不能验证所有因果假设。

与各 Treatment 独立计算不同对照组 uplift 的做法不同：本工具始终在统一目标总体上估计各 `mu_a`，避免排名目标不可比。

---

## 2. 用户故事和公开 API

### 2.1 主入口（设计目标，Phase 1 内可先提供基本 summary）

```python
import pandas as pd
from causalrank import CausalRank

model = CausalRank(
    treatment="route",
    outcome="converted",
    confounders=["age", "income", "gender", "previous_visits"],
    estimator="iptw",
    propensity_model="gbm",
    estimand="population_mean",
    weighting="hajek",
    confidence_level=0.95,
    n_bootstrap=300,
    n_jobs=1,
    random_state=2026,
)
result = model.fit(df)
print(result.summary())
print(result.effects.to_dataframe())
print(result.ranks.to_dataframe())
print(result.diagnostics.summary())
# 后续：result.plot(), result.to_html("report.html")
```

**名称必须区分**：`GBM` 是估计倾向得分的预测模型；`IPTW` 是使用倾向得分的因果估计方法。`weighting="hajek"` 是本项目的预设归一化 IPTW 估计器形式，不等同于换成 OW。

### 2.2 低层入口：便于单独验证排名内核

```python
from causalrank.inference import rank_from_estimates
ranking = rank_from_estimates(
    estimates={"A": 0.25, "B": 0.24, "C": 0.17},
    covariance=sigma,
    confidence_level=0.95,
    method="bonferroni",
)
```

### 2.3 返回对象的建议字段

- `result.estimates`: Ordered labels, `mu`, `se`, `ci_low`, `ci_high`;
- `result.covariance`: `K x K` ndarray，标签次序严格一致；
- `result.pairwise`: `K(K-1)/2` 个效果差、SE、同时 CI、方向判断；
- `result.ranks`: 点排名、排名置信区间 `[L,U]`、是否存在真并列时的约定；
- `result.diagnostics`: 倾向得分、极端权重、ESS、加权 SMD、Bootstrap 有效/失败次数、warnings；
- `result.metadata`: estimator、estimand、样本数、模型参数、版本、随机种子、warnings。

所有数值 API 返回不应依赖 Plotly；Plotly 仅用于附加报告功能。

---

## 3. 统计方法精确定义（必须实现）

### 3.1 数据和缺失值约束

- `T` 非空且至少两类；各组样本数有诊断，任何组不足合理最小量时警告/拒绝见配置；
- `Y` v0.1 必须是 `[0,1]` 的二元值，其他结果类型报明确错误；
- `X` 仅允许 Treatment 前特征，文档显著说明不能把 Treatment 后特征盲目纳入；
- 对 `X` 的数值缺失：在训练折内 fit 中位数插补；分类缺失：在训练折内 fit 众数或常量插补；
- 分类编码必须纳入 Pipeline，在 Bootstrap 内重新 fit，避免泄露；
- 不能用整份样本预先 fit scaler / encoder / imputer 再去做交叉验证或抽样重估；
- V0.1 原始结果缺失和 Treatment 缺失可显式拒绝，不能静默删除观察值。

### 3.2 多分类倾向得分 GBM

目标：

\[e_a(x)=P(T=a\mid X=x),\quad \sum_{a=1}^K e_a(x)=1.\]

初期方案：优先使用 **sklearn `HistGradientBoostingClassifier`**，必要时经过 `ColumnTransformer` 数值/类别预处理；也可由 Agent 选择 sklearn 官方稳定的多类 boosting 实现，必须确认其 `predict_proba` 输出具有所有 `K` 类，且列对齐 `classes_`。不允许用未经校准的 `decision_function` 当作概率。

**重要**：McCaffrey et al. (2013) 中的广义提升模型可能采用每个 Treatment 一对其余组的 GBM 构造；这里明确 V0.1 采用**联合多分类概率模型**，两者不必完全相同。模型 AUC 或分类准确率不能代替 overlap/balance 诊断。

策略：
- V0.1 允许全样本拟合 propensity；为减少过拟合偏差，优先支持可选 `cross_fit=5` 产生 out-of-fold 倾向得分；
- 若启用交叉拟合，使用 stratified K-fold，必须保证每折训练集中含所有 Treatment 类；
- **Bootstrap 每轮重新执行与原始估计相同的全部拟合协议**（包含折划分及预处理），而非只在固定倾向得分上重采样结果；
- 原始全样本拟合与 OOF 估计的用途及局限写入文档；
- 如引入 probability clipping，必须显式配置并报告裁剪比例，同时提醒裁剪可能改变估计偏差、影响目标解释；默认不默默裁剪。

### 3.3 IPTW 估计：Hájek 形式（主方法）

对每个观察个体 `i` 的实际 Treatment `T_i`，权重：

\[w_i=\frac{1}{\hat e_{T_i}(X_i)}.\]

对 Treatment `a`，采用组内归一化的加权平均：

\[\hat\mu_a^{H}=\frac{\sum_{i=1}^n 1\{T_i=a\}w_iY_i}{\sum_{i=1}^n 1\{T_i=a\}w_i}.\]

**为什么选 Hájek**：二元 Outcome 下组内正权重平均始终位于 `[0,1]`；对权重整体乘常数不变；相较 Horvitz–Thompson 形式常见的有限样本稳定性改善。但仍依赖倾向估计及因果假设，不能保证平衡或无偏。

可额外写纯内部基准 HT 版本：

\[\hat\mu_a^{HT}=\frac1n\sum_{i=1}^n 1\{T_i=a\}\frac{Y_i}{\hat e_a(X_i)}.\]

HT 仅用于理论核对/实验，不要求 v0.1 对外暴露。**禁止**使用只用 treatment 内重新归一化后却在公式中错误地除以整体 `n` 的混合算法。

### 3.4 权重与数据诊断

最少输出：
1. 各 Treatment 原始频数、比例、Outcome 原始均值；
2. 各组实际 Treatment 概率 `e_{T_i}(X_i)` 的分位数、最小值；
3. IPTW 权重最小/中位/95%、99%、最大分位；
4. 组内有效样本量 `ESS_a=(sum_i w_{ia})^2/(sum_i w_{ia}^2)`，其中 `w_{ia}=1(T_i=a)w_i`；
5. 原始与加权 SMD 的最大绝对值、超过指定门槛的特征数；数值特征用明确的 pooled reference SD，类别变量按每个 level 的指示变量比较，文档说明定义；
6. 关键变量的 Treatment 间倾向分布、可能存在的弱共同支持；
7. 协变量高基数类别、缺失率、极度不平衡和极端权重的警告。

**质量门控**：默认提供 configurable warnings；若 positivity 严重受损或无法拟合则 fail fast。不要将 `max|SMD|<=0.1` 宣称普遍充分，更不能因弱混杂变量平衡而忽略强混杂的失衡。警告以可配置规则展示，因果解释仍由用户负责。

### 3.5 联合 Bootstrap 协方差（Phase 2）

令原样本估计向量：

\[\hat{\boldsymbol\mu}=(\hat\mu_1,\ldots,\hat\mu_K)^\top.\]

第 `b` 次 Bootstrap **共同重抽样观察单位**，并重新执行预处理、GBM 拟合、倾向得分预测、权重生成及全部 K 组加权均值计算，得到：

\[\hat{\boldsymbol\mu}^{*(b)}=(\hat\mu_1^{*(b)},\ldots,\hat\mu_K^{*(b)}).\]

把 B 次估计叠成 `B x K` 矩阵，再计算：

\[\hat\Sigma_{boot}=\frac{1}{B-1}\sum_{b=1}^B(\hat{\boldsymbol\mu}^{*(b)}-\bar{\boldsymbol\mu}^*)(\hat{\boldsymbol\mu}^{*(b)}-\bar{\boldsymbol\mu}^*)^\top.\]

代码等价于：

```python
sigma = np.cov(bootstrap_estimates, rowvar=False, ddof=1)
```

**绝对不能再除以 n**：每行已是完整重抽样上的估计量向量，而非单条客户的贡献值。

**重抽样单位**：基本版本假设客户独立；若一个客户多次记录、家庭/机构聚类，独立客户行 Bootstrap 不再合适，需要独立的 cluster bootstrap（后续）；v0.1 检测重复 ID 时提示当前 iid 假设，不悄悄忽略。

**稀有 Treatment 风险**：重抽样若有组缺失，记录该轮失败原因；不能静默丢弃大量失败轮后仍声称协方差可靠。设置最少成功次数/成功率门槛及对应 warning/error；Bootstrap 成功样本使用的处理规则必须透明。

**潜在问题**：机器学习 nuisance + 重新拟合的 nonparametric bootstrap 的理论表现依赖算法稳定性和所需正则条件；必须通过模拟覆盖率检查，而不是把它当作通用推断保证。推荐在 v0.1 做重新拟合联合 Bootstrap 基线，后续研究改进（sample splitting、m-out-of-n 等视需要）。

**可复现并行**：用 `numpy.random.SeedSequence` 或显式按轮分配独立种子，输出必须在指定种子下与执行顺序无关（在允许浮点误差范围内）。

### 3.6 点估计与排名推断（Phase 3）

排名 1 表示估计转化率最高。默认 `higher_is_better=True`。

对于每对路线 `i<j`：

\[\hat d_{ij}=\hat\mu_i-\hat\mu_j,\qquad \widehat{SE}(\hat d_{ij})=\sqrt{\hat\Sigma_{ii}+\hat\Sigma_{jj}-2\hat\Sigma_{ij}}.\]

设 `M=K(K-1)/2`，构造 simultaneous Bonferroni pairwise intervals：

\[\hat d_{ij}\pm z_{1-\alpha/(2M)}\widehat{SE}(\hat d_{ij}).\]

若 `CI_lower(i-j)>0` 则可确认 `i` 的结果高于 `j`；若 `CI_upper(i-j)<0` 则确认低于 `j`；其他不能确认。

定义 `B_i` 为**可确认优于** `i` 的路线数，`W_i` 为**可确认劣于** `i` 的路线数，则保守排名置信区间：

\[[L_i,U_i]=[1+B_i,K-W_i].\]

必须清楚说明：该方法依赖同时差值区间的覆盖及渐近正态近似；在真实潜在结果并列时，严格全序名次不唯一，v0.1 应明确 tie convention 或限制理论陈述为无真实并列的情形。**观测点估计相同不能通过字母顺序伪造确定的统计排序**，点估计并列必须有显式标记。

第一版先实现 Bonferroni 基线；max-T / stepdown / csranks 交叉验证在后期，且严格遵守许可证。不能用单项 CI 重叠代替显著性检验。

### 3.7 与 Bootstrap CI 区分

- `mu_a` 的单项 95% CI 可以由正态近似 `mu ± 1.96*sqrt(Sigma_aa)` 给出，或者显式指定百分位法并说明适用条件；
- 排名置信集必须使用 **多重比较同时控制**；
- 不能把 bootstrap 出现第一的频次当成“真实最优概率”；
- 不能在 Bootstrap 内每轮独立重抽样每条路线，因为那会破坏联合依赖结构。

---

## 4. 工程结构

```text
causalrank/
├── pyproject.toml
├── README.md
├── LICENSE
├── PROJECT_SPEC.md
├── src/causalrank/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── results.py
│   ├── validation.py
│   ├── estimators/
│   │   ├── propensity.py      # 多分类 GBM + pipeline / OOF
│   │   └── iptw.py            # Hájek / HT 核验
│   ├── inference/
│   │   ├── bootstrap.py       # 联合重估及 Sigma
│   │   ├── pairwise.py
│   │   └── ranking.py
│   ├── diagnostics/
│   │   ├── overlap.py
│   │   ├── balance.py
│   │   └── weights.py
│   ├── simulation/
│   │   ├── dgp.py
│   │   └── benchmark.py
│   └── visualization/
│       └── report.py          # Phase 4 才实现
├── tests/
│   ├── test_validation.py
│   ├── test_propensity.py
│   ├── test_iptw.py
│   ├── test_balance.py
│   ├── test_bootstrap.py
│   ├── test_covariance.py
│   ├── test_ranking.py
│   ├── test_properties.py
│   └── test_dgp.py
├── examples/
│   ├── 01_gbm_iptw_from_raw_data.ipynb
│   ├── 02_joint_bootstrap.ipynb
│   └── 03_causal_ranking.ipynb
├── docs/
│   ├── implementation_plan.md
│   ├── agent_progress.md
│   ├── methodology.md
│   ├── assumptions.md
│   ├── simulation_protocol.md
│   ├── api.md
│   └── references.md
└── .github/workflows/tests.yml
```

文件是目标结构，不要求第一天创建所有空文件；按阶段逐步落地。

### 4.1 技术选型

- Python >=3.11, `uv` + `pyproject.toml`，`src` layout；
- NumPy、pandas、SciPy、scikit-learn；
- pytest + Hypothesis（数值不变性），Ruff + mypy；
- joblib 并行（Phase 2 在单线程正确后再加），Plotly 后期；
- GitHub Actions 每次 commit 运行测试、lint、基本打包验证；
- 不需要 FastAPI、React、数据库、Agent runtime；
- 依赖版本在发布前锁定测试环境，并记录 Python/sklearn 版本。

---

## 5. 合成数据生成器（DGP）与方法验证

这是项目最重要的研究资产。必须实现可配置、**已知真实潜在结果均值**的数据生成过程。

### 5.1 最小 DGP

- `X`: 混合数值/类别混杂变量，如正态 `X1,X2` 与 Bernoulli `X3`；
- Treatment 分配：通过 `X` 的已知多项 Logit/softmax 函数产生多种 `T`；
- 二元潜在结果：`Y(a)~Bernoulli(sigmoid(beta0 + beta_X·X + tau_a + interactions))`，其中条件购买率因 Treatment 与混杂变量变化；
- 只观察 `Y=Y(T)`；
- 使用独立的大规模 Monte Carlo 参考 X 样本或数值积分获得 `mu_true[a]=E_X[P(Y(a)=1|X)]`，误差控制和参考 seed 必须记录。

### 5.2 至少测试的场景

| 场景 | 目的 |
|---|---|
| Randomized / near-randomized | 验证基本公式，和分组原始均值接近 |
| Correctly modeled propensity | 验证 IPTW 在有混杂下恢复 `mu_true` |
| Strong confounding | 检验权重与 SMD 诊断 |
| Weak overlap | 观察 ESS、极端权重、估计不稳定与警告 |
| Rare treatment | 检验 fold 以及 Bootstrap 有效性 |
| Equal treatment effects | 检验不能凭样本噪声宣称确定排名 |
| Close versus separated effects | 验证排名置信集宽度与统计功效 |
| Mixed categorical/numeric X | 验证编码、缺失值与模型管线 |

### 5.3 Monte Carlo 指标

- `Bias(mu_a)`、RMSE、标准误均值；
- 各 `mu_a` 单项置信区间覆盖率；
- Bonferroni **联合排名覆盖率**：所有真实名次同时落入区间；仅在无真实并列或定义好 tie convention 时统计；
- 置信区间平均宽度、错误确认相对优劣概率；
- 协方差矩阵与 Monte Carlo 经验协方差之间的比较；
- 拟合失败率、Bootstrap 成功率、ESS、SMD、极端权重警告触发率；
- 运行时长与可复现性。

**Monte Carlo 误差**：对于覆盖率 `p_hat`、重复次数 `R`，报告 `sqrt(p_hat(1-p_hat)/R)`，不得因为有限模拟 0.946 vs 0.95 就擅自认定方法失效。计算预算有限时区分 fast tests 与 research benchmarks，后者支持手动运行。

---

## 6. 开发阶段：务必按次序执行

### Phase 0 — 初始化 + 设计（立即）

任务：
- 创建 Python 包骨架、Git 忽略、Apache-2.0 或 MIT 许可证（先确认希望用哪种，否则暂用 MIT 并注明可调整）；
- `docs/implementation_plan.md` 提炼当前结构；
- 实现 API 数据结构和配置校验约定；
- 实现 minimal DGP 并确认 `mu_true` 可重复；
- README 包含因果识别假设和“仅用于研究”警示。

验收：`uv sync` / `uv run pytest` 与简短 smoke test 可运行。

### Phase 1 — 原始数据 → GBM 多分类倾向得分 → IPTW（首个可用里程碑）

任务：
- 输入数据/schema 及预处理 Pipeline；
- GBM `predict_proba`、class alignment、可选 OOF 机制；
- Hájek-IPTW `mu_hat`、权重与 ESS；
- overlap、权重、加权前后 SMD 诊断；
- DGP 中进行多场景验证；
- Notebook `01_gbm_iptw_from_raw_data.ipynb`；
- `docs/methodology.md` 说明 IPTW 与共同目标总体。

验收：
- 可从原始 DataFrame 得到所有 Treatment 的调整后均值；
- 与手工小样本加权公式结果严格吻合；
- 理想 DGP 里随着 n 增大，估计值趋向 `mu_true`（通过重复实验验证，不依赖单次随机样本）；
- 强混杂/弱重叠场景给出明确诊断而非误报“可靠”；
- 所有预处理只在合适的数据折/样本中训练；
- 单元测试、lint、打包检查通过。

### Phase 2 — 联合 Bootstrap → 协方差矩阵（统计推断里程碑）

任务：
- 对同一索引重抽样，重新执行 *所有* Treatment 的效果估计；
- 收集 `B x K` bootstrap estimates，计算 `K x K` 协方差；
- 基于 `Sigma` 生成各路线 SE / normal CI；
- 处理稀有 Treatment、数值无效、失败轮和随机数；
- Notebook `02_joint_bootstrap.ipynb`；
- 运行 Monte Carlo 比较 estimated SE 与真实重复抽样波动。

验收：
- Sigma 对称、PSD（允许数值容差）、对角方差非负，列顺序可追溯；
- 所有非对角值来自联合估计；
- 重新拟合（包括 GBM 与预处理）发生在每轮 Bootstrap；
- 固定 seed 可重现；
- 高失败率阻断推断并披露原因；
- 不把 `np.cov(bootstrap_draws)` 再除以 n。

### Phase 3 — 联合排名置信集（MVP 完成）

任务：
- Bonferroni 同时差值 CI、明确的优劣关系、保守排名 CI；
- 点估计并列处理、`higher_is_better` 可选；
- 与人工构造 covariance cases 对照；
- Notebook `03_causal_ranking.ipynb`；
- DGP 的最终 ranking coverage 基准；
- 用户友好摘要 `result.summary()`、`result.ranks.to_dataframe()`。

验收：
- 排名置信集与处理变量标签同时输出，能追溯数据、公式和置信水平；
- Bonferroni 校正 K=2 时退化为常规两方案比较；
- 排名结果对 Treatment 重排、效果加共同常数（在纯排名内核下）保持不变；
- 不以差异不显著作为效果相等的证据；
- 统计测试与 Monte Carlo 按研究设计通过。

### Phase 4 — 分发与可视化（不影响 MVP 的统计验收）

任务：
- 报告显示各路线估计值、SE、CI、排名置信集、两两优势关系；
- Overlap/ESS/SMD/Bootstrap 诊断分开展示；
- 导出静态 HTML，不必搭建 React、Streamlit 服务器；
- 文档与 Quickstart、PyPI 发布准备。

验收：干净环境安装后，用户从 CSV 运行示例并生成 HTML；报告中的数值全部追溯到 result 对象。

### Later — 扩展研究，**不要提前实现**

- AIPW / 多 Treatment DR，带适当 cross-fitting；
- Overlap Weighting（明确新的 target estimand）；
- 灵敏度分析、weak overlap 应对；
- Influence function / sandwich 对照联合 Bootstrap；
- max-T/stepdown rank inference；
- 近似最优集合、MCB、损失决策推断；
- cluster bootstrap、多时期观察及复杂采样结构。

---

## 7. 测试清单（Agent 必须逐项落地）

### 7.1 输入与数据
- [ ] treatment/outcome/confounder 字段不存在时报明确异常。
- [ ] 非 0/1 outcome、缺失 treatment、单一 treatment 拒绝。
- [ ] 标签是字符串或整数时稳定排序，不按模型概率列错误映射。
- [ ] 类别混杂、数值混杂和缺失协变量处理正确。
- [ ] 样本量太少、Treatment 稀有时明确提示。

### 7.2 IPTW 与倾向得分
- [ ] 所有倾向得分有限、非负、行和为 1（数值容差）。
- [ ] sklearn `classes_` 与公开 Treatment 顺序映射完全一致。
- [ ] 手算 2 Treatment × 4 个样本的 Hájek 结果与实现一致。
- [ ] 对同组权重乘相同常数不会改变其加权均值。
- [ ] 权重极端、ESS 低时产生诊断。
- [ ] 不用模型 accuracy/AUC 作为识别有效性的判定。

### 7.3 Bootstrap
- [ ] 每轮共享原始样本重抽样索引，重估全部路线。
- [ ] 每轮重训 GBM，而不是复用原始模型。
- [ ] 100% Bootstrap 成功时返回 `B x K` draws 和 `K x K` Sigma。
- [ ] 含两条路线、已知 draws 的手算 covariance test。
- [ ] `Sigma` 与 `np.cov(draws, rowvar=False, ddof=1)` 一致。
- [ ] 实现保存/不保存 draws 的选项及结构化失败报告。
- [ ] 同一种子与配置重复执行一致；并行策略可复现。

### 7.4 排名推断
- [ ] 2、3、5 组的已知例子得到正确差值 SE 和排名区间。
- [ ] 相关性被正确带入差值标准误。
- [ ] 对称与 PSD 验证；差值方差接近零有明确处理。
- [ ] 同时 CI 仅在经多重校正后宣布某组“确定优于”另一组。
- [ ] 真实并列与点估计并列按指定规则处理。
- [ ] Treatment 次序变化后，结果按标签重排一致。

### 7.5 研究有效性
- [ ] Oracle randomized DGP 与因果模型结果进行对照。
- [ ] Confounding DGP 下原始组均值与调整均值的偏差比较。
- [ ] Coverage 的 Monte Carlo SE 与计算预算记录。
- [ ] README 明确不可验证的因果识别假设。
- [ ] 所有结果都来自合成或可合法使用的公开数据。

---

## 8. 失败模式与安全护栏

| 失败模式 | 要求 |
|---|---|
| 极端倾向得分 | 显示极值、分位数、比例和 ESS，不得静默裁剪 |
| 加权 SMD 大 | 标记强警告，不要输出“可靠因果排序”文案 |
| 缺少 Treatment 类别 | 训练前失败，Bootstrap 单轮失败可记录但高比例时阻断 |
| Bootstrap 不稳定 | 报告有效重复数及失败比例；不要挑好看的轮次 |
| 共同支持不成立 | 明确 estimand 外推不可识别风险；不能用漂亮 CI 遮盖 |
| 同一客户重复行 | 提示 iid 假设不适用，后续改 cluster bootstrap |
| 选择 Treatment 后协变量 | 文档显著警告可能引入偏差 |
| Covariance 非 PSD | 明确错误，不做悄悄修补 |
| 窄 CI 与错误模型 | 不得称“因果效果已验证”；因果识别依赖外部假设 |
| 所有方案差别很小 | 容许宽的排名置信集，这是正确答案，不是算法失败 |

---

## 9. 第一轮 Agent 工作指令（可直接复制）

**Task: Implement CausalRank Phase 0 and Phase 1 in this repository.**

Read `PROJECT_SPEC.md` fully. Treat Sections 0–8 as the authoritative specification.

1. Audit existing files; create `docs/implementation_plan.md` with architecture, exact estimator definitions, DGP design, and testing strategy.
2. Set up Python 3.11+ with a `src/causalrank` installable package, uv, pytest, ruff and mypy. Establish CI.
3. Implement input schema/validation, preprocessing pipeline, deterministic Treatment label alignment, multi-class GBM propensity estimation, and Hájek multi-treatment IPTW.
4. Implement clear ESS, weight-distribution, overlap, and unweighted/weighted SMD diagnostics. Add warning states for weak overlap and serious residual imbalance.
5. Implement a configurable synthetic DGP with known population potential-outcome means (via sufficiently precise independent Monte Carlo reference), with weak/strong overlap cases.
6. Add unit tests with independently calculated hand examples, invariance tests, and an end-to-end randomized/confounded DGP smoke test.
7. Create `examples/01_gbm_iptw_from_raw_data.ipynb` and README Quickstart.
8. Run all test, lint, type, notebook-execution and packaging checks. Fix failures; do not skip tests just to achieve green CI.
9. Write `docs/agent_progress.md` listing implemented pieces, numerical checks, test outputs, limitations, and tasks deferred to Phase 2.
10. **Stop after Phase 1**. Do not implement joint Bootstrap, ranking inference or UI until this foundation is verified. Do not request approval for ordinary implementation choices; document them.

**Required Phase 1 final report**:
- Package directory tree;
- exact formulas and model choices;
- tests performed, including commands and numerical results;
- DGP ground truth versus estimated `mu_a` (not just classification metrics);
- overlap/ESS/SMD diagnostic outputs;
- any unfulfilled requirements;
- Phase 2 readiness assessment.

**Acceptance priority**: A correct, validated estimator with clear assumptions is more valuable than more features. If an assumption cannot be justified, expose it as a limitation rather than concealing it.

---

## 10. 参考研究与基准实现

1. McCaffrey, D. F., Griffin, B. A., Almirall, D., Slaughter, M. E., Ramchand, R., & Burgette, L. F. (2013). **A tutorial on propensity score estimation for multiple treatments using generalized boosted models**. *Statistics in Medicine*, 32, 3388–3414. https://doi.org/10.1002/sim.5753
2. Mogstad, M., Romano, J. P., Shaikh, A. M., & Wilhelm, D. (2024). **Inference for Ranks**. *Review of Economic Studies*. https://doi.org/10.1093/restud/rdad006
3. Chetverikov, D., Mogstad, M., Morgen, P., Romano, J. P., Shaikh, A. M., & Wilhelm, D. (2024). **csranks: An R Package for Estimation and Inference Involving Ranks**. https://arxiv.org/abs/2401.15205
4. `csranks` 文档 / 实现：https://danielwilhelm.github.io/R-CS-ranks/
5. scikit-learn 官方文档：https://scikit-learn.org/stable/

**文献使用约定**：实现前核实论文的模型设定、同一目标总体下的多 Treatment 权重定义、渐近条件和软件接口；不可仅因为某篇文章标题相似就宣称实现与其完全一致。

---

## 11. 项目边界与下一步

当前最重要的是建立可信的 **GBM + IPTW 核心估计器**，而不是马上追求漂亮的排名报告。

Phase 1 成功的标志：对具有已知真实潜在结果均值的模拟数据，能够从原始 DataFrame 进行完整、可测试、可诊断的多 Treatment IPTW 分析，并明确告诉用户何时不应信任结果。

随后才进入 Phase 2 联合 Bootstrap、Phase 3 排名置信集；最终交付的是完整的端到端 CausalRank，而不是 `csranks` 的简单复刻。
