# CausalRank

[![PyPI](https://img.shields.io/pypi/v/causalrank.svg)](https://pypi.org/project/causalrank/)
[![Python](https://img.shields.io/pypi/pyversions/causalrank.svg)](https://pypi.org/project/causalrank/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**End-to-end Python framework for multi-treatment causal outcome estimation, joint
uncertainty quantification, and statistically defensible treatment ranking from
observational data.**

> **当前状态：Phase 0–3（v0.3.0，MVP 完成）。**
> 已实现：原始 DataFrame → 多分类 GBM 倾向得分 → Hájek IPTW → `mu_a = E[Y(a)]` 估计
> + 重叠 / ESS / SMD 诊断与质量门控；**联合 Bootstrap 协方差、SE / 单项 CI**
> （`n_bootstrap` 显式开启）；失败轮披露与推断阻断；**Bonferroni 两两同时区间、
> 可确认优劣关系、保守排名置信集** `[1+B_i, K-W_i]`（`result.pairwise / result.ranks`）。
> **尚未实现（后续）**：max-T/stepdown 排名推断、AIPW/OW、可视化报告（Phase 4）。
> 点估计排序不是统计上确立的排名；`inconclusive` 不是相等的证据。

---

## ⚠️ 因果识别假设（软件无法验证）

只有在以下条件成立时，IPTW 估计值才能解释为因果平均潜在结果 `mu_a = E[Y(a)]`：

1. **一致性（consistency）**：观测结果等于所接受 Treatment 的潜在结果，Treatment 版本定义一致；
2. **条件可交换性（无未观测混杂）**：给定输入的协变量 `X` 后，Treatment 分配与潜在结果独立。
   **软件无法检查这一点**；遗漏混杂（尤其是 Treatment 前的强混杂）会直接使估计有偏；
3. **Positivity（共同支持）**：每个 `x` 上每种 Treatment 的倾向得分都为正。弱重叠会导致极端
   权重与不稳定估计，诊断会给出警告，但**不能修复**该问题；
4. **SUTVA / 无干扰**：个体之间潜在结果互不影响；
5. **iid 抽样**：观测行独立同分布（存在客户/家庭/机构聚类时本版本不适用，需 cluster bootstrap）；
6. **协变量必须测量于 Treatment 之前**。把 Treatment 后变量（中介、后果）当作混杂调整会引入偏差。

模型 AUC / 准确率 **不能** 作为因果估计可靠的证据；平衡、重叠、稳定性诊断不通过时，
软件会降级其质量门控并输出警告，**不会**输出“可靠因果排序”类结论。因果解释始终由使用者负责。

**仅用于研究与方法验证**，不构成任何业务、医疗或政策决策建议。

## 安装

要求 Python ≥ 3.11。

从 [PyPI](https://pypi.org/project/causalrank/) 安装发布版：

```bash
pip install causalrank
# 或 uv pip install causalrank
```

开发环境使用 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync                 # 创建 .venv 并安装 runtime + dev 依赖
uv run pytest           # 运行测试
uv run ruff check .     # lint
uv run mypy             # 类型检查
uv build                # 打包检查
```

或在本地以可编辑模式安装本包本身：

```bash
uv pip install -e .      # 或 pip install -e .
```

## Quickstart

```python
import pandas as pd
from causalrank import CausalRank

# df 需包含：treatment 列（互斥 K 类）、outcome 列（0/1）、Treatment 前协变量列
df = pd.read_csv("your_data.csv")

model = CausalRank(
    treatment="route",
    outcome="converted",
    confounders=["age", "income", "gender", "previous_visits"],
    estimator="iptw",            # Phase 1/2 支持 IPTW
    propensity_model="gbm",      # 倾向得分预测模型（多分类 GBM）
    estimand="population_mean",  # mu_a = E[Y(a)]，同一目标总体
    weighting="hajek",           # 组内归一化 IPTW（不改变估计目标）
    cross_fit=5,                 # 可选：out-of-fold 倾向得分
    confidence_level=0.95,
    n_bootstrap=300,             # >= 1 显式开启联合 Bootstrap 推断（SE/CI）
    random_state=2026,
)
result = model.fit(df)

print(result.summary())              # 估计值、SE/CI、排名置信集、诊断与质量门控
print(result.effects.to_dataframe()) # mu_a、se、ci、原始均值、权重、ESS
print(result.covariance)             # K×K 联合 Bootstrap 协方差
print(result.ranks.to_dataframe())   # 点排名 + 排名置信区间 [1+B_i, K-W_i]
print(result.pairwise.to_dataframe())# 两两同时区间与可确认优劣关系
print(result.diagnostics.summary())  # overlap / 权重 / SMD 诊断
for w in result.diagnostics.warnings:
    print(w.level, w.code, w.message)
```

`n_bootstrap` 是**显式开关**：默认 `None` 只输出点估计与诊断（秒级）；设为 300 会每轮
重新拟合全部协议来估计协方差（分钟级，可 `n_jobs` 并行且结果与串行逐位一致）。

低层排名内核可单独验证（规格 2.2）：

```python
from causalrank.inference import rank_from_estimates

ranking = rank_from_estimates(
    estimates={"A": 0.25, "B": 0.24, "C": 0.17},
    covariance=sigma,
    confidence_level=0.95,
    method="bonferroni",
)
```

## 估计方法（一句话版）

- 倾向得分：`e_a(x) = P(T=a|X=x)`，由**联合多分类** GBM（scikit-learn
  `HistGradientBoostingClassifier`）估计；预处理（缺失插补、one-hot）全部包在 Pipeline 内、
  只在训练折内 fit，避免信息泄露。
- 估计器：Hájek IPTW，`w_i = 1 / e_{T_i}(X_i)`，
  `mu_a = Σ_{T_i=a} w_i Y_i / Σ_{T_i=a} w_i`，所有 Treatment 在**同一目标总体**上比较。
- 诊断：权重分位数、有效样本量 ESS、倾向得分分布（overlap）、加权前后标准化均值差 SMD，
  以及可配置的质量门控警告。

完整定义、假设与数值边界见 [`docs/methodology.md`](docs/methodology.md) 与
[`docs/assumptions.md`](docs/assumptions.md)。

## 合成数据与验证

`causalrank.simulation` 提供**已知真实潜在结果均值** `mu_true` 的可配置 DGP
（独立大规模 Monte Carlo 参考 + MC 标准误），内置 8 个场景（随机化、混杂、强混杂、弱重叠、
稀有 Treatment、效应相等、效应接近、混合类型 + 缺失）。模拟协议见
[`docs/simulation_protocol.md`](docs/simulation_protocol.md)，端到端示例见
[`examples/01_gbm_iptw_from_raw_data.ipynb`](examples/01_gbm_iptw_from_raw_data.ipynb)、
[`examples/02_joint_bootstrap.ipynb`](examples/02_joint_bootstrap.ipynb)、
[`examples/03_causal_ranking.ipynb`](examples/03_causal_ranking.ipynb)。

## 项目结构

```text
src/causalrank/        包源码（api / config / validation / estimators / diagnostics / simulation）
tests/                 pytest 单元与端到端测试
examples/              可执行示例 Notebook
docs/                  方法、假设、模拟协议、API、进度报告
```

## 许可证

MIT（暂定，见 `LICENSE`）。参考文献的使用约定见 [`docs/references.md`](docs/references.md)。
