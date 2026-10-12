"""Generate the examples/*.ipynb notebooks (kept reproducible)."""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[dict[str, object]] = [
    {
        "cell_type": "markdown",
        "source": [
            "# CausalRank — 从原始数据到多 Treatment 因果估计（Phase 1）\n",
            "\n",
            "本 Notebook 演示 **Phase 1** 的完整链路：\n",
            "\n",
            "1. 已知真值的合成数据（含混杂）——`causalrank.simulation` 提供 `mu_true = E[Y(a)]`；\n",
            "2. 原始组均值为什么是有偏的（混杂偏差）；\n",
            "3. 多分类 GBM 倾向得分 + Hájek IPTW 得到同一目标总体上的 `mu_a`；\n",
            "4. 重叠 / ESS / SMD 诊断与质量门控；\n",
            "5. 弱重叠场景下**诚实的失败样貌**：宽警告而非“可靠排名”。\n",
            "\n",
            "> **Phase 1 不提供统计推断**：`se / ci / ranks` 为空，联合 Bootstrap 协方差（Phase 2）\n",
            "> 与排名置信集（Phase 3）尚未实现。请勿把点估计排序当作已确立的统计排名。\n",
            "> 因果解释依赖一致性、无未观测混杂、positivity 与 iid 抽样等**无法用数据验证**的假设。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "import pandas as pd\n",
            "\n",
            "from causalrank import CausalRank\n",
            "from causalrank.simulation import make_dataset, scenario_names\n",
            "\n",
            "pd.set_option(\"display.width\", 120)\n",
            "print(\"available scenarios:\", scenario_names())",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 1. 合成数据：已知真实潜在结果均值\n",
            "\n",
            "`confounded` 场景：多项 softmax 分配 + 二元潜在结果。真值\n",
            "$\\mu_a = E_X[P(Y(a)=1\\mid X)]$ 由**独立**的 20 万条 Monte Carlo 参考协变量样本估计，\n",
            "参考 seed 与 MC 标准误都会记录下来。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "ds = make_dataset(\"confounded\", n=4000, random_state=2026)\n",
            "print(\"reference: n =\", ds.n_reference, \", seed =\", ds.reference_seed)\n",
            "truth = pd.DataFrame({\"mu_true\": ds.mu_true, \"mc_se\": ds.mu_true_mc_se})\n",
            "display(truth.round(5))\n",
            "ds.df.head()",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 2. 朴素对比：原始组均值为什么有偏\n",
            "\n",
            "各 Treatment 组的协变量分布不同（混杂），所以原始组均值比较的是**不同的人群**，\n",
            "不是同一目标总体上的 $E[Y(a)]$。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "raw = ds.df.groupby(\"treatment\")[\"outcome\"].agg([\"mean\", \"count\"])\n",
            "raw.columns = [\"raw_mean\", \"n\"]\n",
            "raw[\"mu_true\"] = ds.mu_true\n",
            "raw[\"raw_bias\"] = raw[\"raw_mean\"] - raw[\"mu_true\"]\n",
            "display(raw.round(4))\n",
            "print(\"mean |raw bias| =\", round(float(raw[\"raw_bias\"].abs().mean()), 4))",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 3. 估计器：多分类 GBM 倾向得分 + Hájek IPTW\n",
            "\n",
            "$$\\hat e_a(x)=\\widehat{P}(T=a\\mid X=x),\\qquad w_i=\\frac{1}{\\hat e_{T_i}(X_i)},\\qquad\n",
            "\\hat\\mu_a^{H}=\\frac{\\sum_i 1\\{T_i=a\\}w_iY_i}{\\sum_i 1\\{T_i=a\\}w_i}$$\n",
            "\n",
            "- 倾向得分是**预测模型**（联合多分类 GBM），IPTW 才是因果估计方法；\n",
            "- 插补 / 编码 / 拟合全部包在 sklearn Pipeline 内，`cross_fit=5` 产生 out-of-fold 倾向得分，\n",
            "  避免“同一份数据既拟合又评估”导致的过拟合权重；\n",
            "- 所有 Treatment 都在**同一目标总体**上估计，排名目标才可比。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "model = CausalRank(\n",
            "    treatment=\"treatment\",\n",
            "    outcome=\"outcome\",\n",
            "    confounders=[\"x1\", \"x2\", \"x3\", \"x4\"],\n",
            "    estimator=\"iptw\",\n",
            "    propensity_model=\"gbm\",\n",
            "    estimand=\"population_mean\",\n",
            "    weighting=\"hajek\",\n",
            "    cross_fit=5,\n",
            "    confidence_level=0.95,\n",
            "    random_state=2026,\n",
            ")\n",
            "result = model.fit(ds.df)\n",
            "result.effects.to_dataframe().round(4)",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "est = result.effects.to_dataframe().set_index(\"label\")\n",
            "compare = pd.DataFrame(\n",
            "    {\n",
            "        \"mu_hat\": est[\"mu\"],\n",
            "        \"mu_true\": ds.mu_true,\n",
            "        \"err_adjusted\": est[\"mu\"] - ds.mu_true,\n",
            "        \"raw_mean\": est[\"raw_mean\"],\n",
            "        \"err_raw\": est[\"raw_mean\"] - ds.mu_true,\n",
            "    }\n",
            ")\n",
            "display(compare.round(4))\n",
            "print(\"mean |err| adjusted =\", round(float(compare[\"err_adjusted\"].abs().mean()), 4))\n",
            "print(\"mean |err| raw      =\", round(float(compare[\"err_raw\"].abs().mean()), 4))",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 4. 诊断：权重、重叠、平衡与质量门控\n",
            "\n",
            "- **ESS**：$\\mathrm{ESS}_a=(\\sum_i w_{ia})^2/\\sum_i w_{ia}^2$，衡量权重把有效信息集中到多少条观测上；\n",
            "- **overlap**：实际 Treatment 概率 $e_{T_i}(X_i)$ 的分布，极小值意味着 positivity 弱；\n",
            "- **SMD**：加权前后协变量在 Treatment 两两之间的标准化均值差（分母用未加权 pooled SD）。\n",
            "\n",
            "诊断只能**暴露经验问题**，不能验证“无未观测混杂”等识别假设。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "print(result.diagnostics.summary())",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "for w in result.warnings:\n",
            "    print(f\"{w.level:>8} | {w.code:<28} | {w.message}\")\n",
            "print(\"\\nquality gate:\", result.diagnostics.gate)",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 5. 弱重叠场景：诚实的失败样貌\n",
            "\n",
            "当 Treatment 分配接近确定性、共同支持不足时，IPTW 权重会爆炸、ESS 崩塌。\n",
            "正确的行为是**给出强警告并降级质量门控**，而不是输出一个看起来精确的排名。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "weak = make_dataset(\"weak_overlap\", n=3000, random_state=7)\n",
            "weak_result = CausalRank(\n",
            "    treatment=\"treatment\",\n",
            "    outcome=\"outcome\",\n",
            "    confounders=[\"x1\", \"x2\", \"x3\", \"x4\"],\n",
            "    cross_fit=5,\n",
            "    random_state=2026,\n",
            ").fit(weak.df)\n",
            "display(weak_result.effects.to_dataframe().round(4))\n",
            "print(\"quality gate:\", weak_result.diagnostics.gate)\n",
            "for w in weak_result.warnings:\n",
            "    print(f\"{w.level:>8} | {w.code:<22} | {w.message[:110]}\")",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 6. 本版本的边界\n",
            "\n",
            "1. **无推断**：`se / ci / ranks` 为空（`result.inference_available is False`）；\n",
            "   Phase 2 引入“每轮重抽样 + 重新拟合全部协议”的联合 Bootstrap 协方差，\n",
            "   Phase 3 引入 Bonferroni 同时差值区间与保守排名置信集；\n",
            "2. **不检验识别假设**：无未观测混杂、一致性、positivity 由使用者负责论证；\n",
            "3. **不把 AUC 当证据**：倾向模型的分类表现不能替代 overlap / balance 诊断；\n",
            "4. **iid 假设**：重复记录同一客户/家庭时行 Bootstrap 不成立（本版本只提示）；\n",
            "5. **不做**：AIPW、Overlap Weighting、PSM、DML、个体层最优决策——明确列为后续研究。\n",
            "\n",
            "详细定义见 `docs/methodology.md` 与 `docs/assumptions.md`。",
        ],
    },
]


CELLS_02: list[dict[str, object]] = [
    {
        "cell_type": "markdown",
        "source": [
            "# CausalRank — 联合 Bootstrap 协方差与不确定性量化（Phase 2）\n",
            "\n",
            "本 Notebook 演示 Phase 2 的统计推断内核：\n",
            "\n",
            "1. **联合重抽样**：每轮对同一行索引有放回重抽样（一个索引向量服务全部 Treatment）；\n",
            "2. **每轮重拟合全部协议**：插补/编码/GBM/折划分/权重/K 组均值全部重跑；\n",
            "3. 协方差 $\\hat\\Sigma = \\mathrm{np.cov}(draws, rowvar=False, ddof=1)$，\n",
            "   **绝不**再除以 $n$（每行是完整估计量，不是单客户贡献值）；\n",
            "4. 单项 CI：正态近似 $\\hat\\mu_a \\pm z_{1-\\alpha/2}\\sqrt{\\hat\\Sigma_{aa}}$；\n",
            "5. 失败轮（稀有臂被漏抽等）被记录并披露；成功率过低时**阻断推断**；\n",
            "6. Monte Carlo 校准：bootstrap SE 与真实抽样波动的比较。\n",
            "\n",
            "> **排名推断属于 Phase 3**：本 Notebook 不做多重比较校正的两两区间或排名置信集，\n",
            "> 点估计排序仍不是统计上确立的排名。不确定性只覆盖估计量的抽样波动，\n",
            "> 不覆盖识别假设被违背带来的偏差。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "import numpy as np\n",
            "import pandas as pd\n",
            "\n",
            "from causalrank import CausalRank\n",
            "from causalrank.simulation import make_dataset\n",
            "\n",
            "pd.set_option(\"display.width\", 130)",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 1. 启用联合 Bootstrap 的完整分析\n",
            "\n",
            "`n_bootstrap=120` 是显式开启推断的开关（默认 `None` 只给点估计，避免无意中跑几分钟）。\n",
            "研究级配置建议 `n_bootstrap=300` 以上；这里为 Notebook 执行时间取小预算。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "ds = make_dataset(\"confounded\", n=2000, random_state=2026)\n",
            "model = CausalRank(\n",
            "    treatment=\"treatment\",\n",
            "    outcome=\"outcome\",\n",
            "    confounders=[\"x1\", \"x2\", \"x3\", \"x4\"],\n",
            "    cross_fit=3,\n",
            "    n_bootstrap=120,\n",
            "    ci_method=\"normal\",\n",
            "    n_jobs=1,\n",
            "    random_state=2026,\n",
            ")\n",
            "result = model.fit(ds.df)\n",
            "print(\"inference_available:\", result.inference_available)\n",
            "display(result.effects.to_dataframe().round(4))\n",
            "print({k: v for k, v in result.metadata[\"bootstrap\"].items() if k != \"failure_reasons\"})",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "est = result.effects.to_dataframe().set_index(\"label\")\n",
            "check = pd.DataFrame(\n",
            "    {\n",
            "        \"mu_hat\": est[\"mu\"],\n",
            "        \"se\": est[\"se\"],\n",
            "        \"ci_low\": est[\"ci_low\"],\n",
            "        \"ci_high\": est[\"ci_high\"],\n",
            "        \"mu_true\": ds.mu_true,\n",
            "        \"covered\": (ds.mu_true >= est[\"ci_low\"]) & (ds.mu_true <= est[\"ci_high\"]),\n",
            "    }\n",
            ")\n",
            "display(check.round(4))",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 2. 协方差矩阵：联合估计的依赖结构\n",
            "\n",
            "非对角元素来自**同一次**重抽样下的联合估计——两臂共享同一批被抽中的客户，\n",
            "因此 $\\hat\\Sigma_{ij}$ 携带真实的估计相关性。检验相关矩阵与 PSD：",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "sigma = result.covariance\n",
            "display(pd.DataFrame(sigma, index=est.index, columns=est.index).round(6))\n",
            "d = np.sqrt(np.diag(sigma))\n",
            "corr = sigma / np.outer(d, d)\n",
            "display(pd.DataFrame(corr, index=est.index, columns=est.index).round(3))\n",
            "print(\"min eigenvalue:\", float(np.linalg.eigvalsh(sigma).min()))\n",
            "print(\"SE == sqrt(diag(Sigma)):\", np.allclose(est[\"se\"], d, rtol=1e-10))",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 3. 可复现性：同种子逐位一致，且与执行顺序无关\n",
            "\n",
            "每轮种子由 `SeedSequence(base_seed).spawn(B)` 派生，同时驱动重抽样与模型随机性；\n",
            "因此 `n_jobs=2` 并行与单线程结果逐位相同。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "small = {\"n_bootstrap\": 12, \"bootstrap_min_successes\": 8}\n",
            "a = CausalRank(treatment=\"treatment\", outcome=\"outcome\", confounders=[\"x1\", \"x2\", \"x3\", \"x4\"],\n",
            "               random_state=7, **small).fit(ds.df)\n",
            "b = CausalRank(treatment=\"treatment\", outcome=\"outcome\", confounders=[\"x1\", \"x2\", \"x3\", \"x4\"],\n",
            "               random_state=7, **small).fit(ds.df)\n",
            "c = CausalRank(treatment=\"treatment\", outcome=\"outcome\", confounders=[\"x1\", \"x2\", \"x3\", \"x4\"],\n",
            "               random_state=7, n_jobs=2, **small).fit(ds.df)\n",
            "print(\"same seed identical   :\", np.array_equal(a.covariance, b.covariance))\n",
            "print(\"n_jobs=2 identical    :\", np.array_equal(a.covariance, c.covariance))",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 4. 失败轮与推断阻断：稀有 Treatment 的诚实处理\n",
            "\n",
            "当某轮重抽样漏掉稀有臂时，该轮**无法**给出完整 K 维向量：记录失败原因并排除，\n",
            "绝不静默丢弃；成功率低于门槛（默认 90% 且至少 30 轮）时**阻断推断**。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "rare = make_dataset(\"rare_treatment\", n=70, random_state=3)\n",
            "rare_result = CausalRank(\n",
            "    treatment=\"treatment\",\n",
            "    outcome=\"outcome\",\n",
            "    confounders=[\"x1\", \"x2\", \"x3\", \"x4\"],\n",
            "    n_bootstrap=40,\n",
            "    bootstrap_min_successes=20,\n",
            "    bootstrap_min_success_rate=1.0,  # 任何失败都阻断\n",
            "    random_state=11,\n",
            ").fit(rare.df)\n",
            "print(\"inference_available:\", rare_result.inference_available)\n",
            "print(\"成功/失败轮:\", rare_result.metadata[\"bootstrap\"][\"n_success\"],\n",
            "      \"/\", rare_result.metadata[\"bootstrap\"][\"n_failed\"])\n",
            "for w in rare_result.warnings:\n",
            "    if \"bootstrap\" in w.code:\n",
            "        print(f\"{w.level:>8} | {w.code:<30} | {w.message[:110]}\")",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 5. Monte Carlo 校准：SE 是否对得上真实抽样波动？\n",
            "\n",
            "`se_ratio = 平均 bootstrap SE / mu_hat 的经验标准差`；覆盖率同时报告 MC 误差\n",
            "$\\sqrt{p(1-p)/R}$。研究级脚本：`scripts/run_se_validation.py`（R=24、B=150 的结果\n",
            "见 `docs/agent_progress.md`）。下面是一个小预算演示：",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "from causalrank.simulation import run_se_benchmark\n",
            "\n",
            "se_res = run_se_benchmark(\"confounded\", n=600, n_replications=6, n_bootstrap=30, random_state=5)\n",
            "display(se_res.summary.round(3))\n",
            "print(\"注意：R=6 的 MC 误差很大（coverage 的 MC se 约 0.10），只演示流程不下结论\")",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 6. 边界与下一步\n",
            "\n",
            "1. CI 覆盖的是**估计量在拟合模型下的抽样波动**，不覆盖未观测混杂等识别假设违背带来的偏差；\n",
            "2. 重新拟合的非参数 bootstrap + ML nuisance 的理论性质依赖算法稳定性，\n",
            "   必须用模拟覆盖率检验（上节）而不是当作通用推断保证；\n",
            "3. 单项 CI **不能**代替两两显著性检验；`ci_method=\"percentile\"` 可选，\n",
            "   但其适用条件是 bootstrap 分布能近似抽样分布；\n",
            "4. **Phase 3**：Bonferroni 同时差值区间、可确认优劣关系、保守排名置信集。\n",
            "\n",
            "定义与协议见 `docs/methodology.md` 与 `docs/simulation_protocol.md`。",
        ],
    },
]


CELLS_03: list[dict[str, object]] = [
    {
        "cell_type": "markdown",
        "source": [
            "# CausalRank — 从估计到排名：同时区间与排名置信集（Phase 3）\n",
            "\n",
            "本 Notebook 演示 Phase 3 的排名推断内核：\n",
            "\n",
            "1. **两两同时区间**：$\\hat d_{ij}=\\hat\\mu_i-\\hat\\mu_j$，\n",
            "   $\\widehat{SE}(\\hat d_{ij})=\\sqrt{\\hat\\Sigma_{ii}+\\hat\\Sigma_{jj}-2\\hat\\Sigma_{ij}}$，\n",
            "   Bonferroni 校正 $\\hat d_{ij}\\pm z_{1-\\alpha/(2M)}\\widehat{SE}$，$M=K(K-1)/2$；\n",
            "2. **可确认优劣关系**：仅当同时区间整体位于 0 的一侧才宣称“确定优于/劣于”；\n",
            "3. **保守排名置信区间**：$[L_i,U_i]=[1+B_i,\\ K-W_i]$，\n",
            "   $B_i$ 为可确认优于 $i$ 的路线数、$W_i$ 为可确认劣于 $i$ 的路线数；\n",
            "4. **K=2 退化检验**：$M=1$ 时 Bonferroni 因子就是常规 $z_{1-\\alpha/2}$；\n",
            "5. **真并列红线**：`equal_effects` 场景下噪声不得产生确定排名。\n",
            "\n",
            "> **红线**：不显著 ≠ 相等（`inconclusive` 不是相等的证据）；\n",
            "> 点估计并列共享名次并显式标记，绝不按标签字母序伪造确定排序；\n",
            "> bootstrap 中点排名第一的频次不是“最优概率”；单项 CI 重叠不能代替显著性检验。\n",
            "> 排名结论依赖联合协方差与渐近正态近似，覆盖率以模拟检验为准（`scripts/run_rank_validation.py`）。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "import numpy as np\n",
            "import pandas as pd\n",
            "\n",
            "from causalrank import CausalRank\n",
            "from causalrank.inference import rank_from_estimates\n",
            "from causalrank.simulation import make_dataset\n",
            "\n",
            "pd.set_option(\"display.width\", 130)",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 1. 完整分析：估计 + 联合 Bootstrap 协方差\n",
            "\n",
            "排名推断消费的是 **Phase 2 的联合协方差** $\\hat\\Sigma$（K×K，标签规范序）。\n",
            "关闭 `n_bootstrap` 或成功率过低时推断被阻断，`pairwise/ranks` 保持 `None`。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "ds = make_dataset(\"confounded\", n=2000, random_state=2026)\n",
            "model = CausalRank(\n",
            "    treatment=\"treatment\",\n",
            "    outcome=\"outcome\",\n",
            "    confounders=[\"x1\", \"x2\", \"x3\", \"x4\"],\n",
            "    cross_fit=3,\n",
            "    n_bootstrap=100,\n",
            "    confidence_level=0.95,\n",
            "    higher_is_better=True,  # rank 1 = 最高转化率；成本型指标用 False\n",
            "    random_state=2026,\n",
            ")\n",
            "result = model.fit(ds.df)\n",
            "print(\"inference_available:\", result.inference_available)\n",
            "display(result.effects.to_dataframe().round(4))",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 2. 两两同时区间（Bonferroni）\n",
            "\n",
            "差值标准误使用**联合**协方差：$\\widehat{SE}(\\hat d_{ij})=\\sqrt{\\hat\\Sigma_{ii}+\\hat\\Sigma_{jj}-2\\hat\\Sigma_{ij}}$，\n",
            "臂间相关性被正确带入。`decision` 的语义是“更好”方向（受 `higher_is_better` 影响）：\n",
            "\n",
            "- `CI_lower(i-j) > 0` → `i_better`（确认 $i$ 优于 $j$）；\n",
            "- `CI_upper(i-j) < 0` → `j_better`；\n",
            "- 其他 → `inconclusive`：**不能**解读为两者相等。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "pairs = result.pairwise.to_dataframe()\n",
            "display(pairs.round(4))\n",
            "print(\"M =\", result.pairwise.n_pairs, \", z_crit =\", round(result.pairwise.z_crit, 4))\n",
            "print(\"\\n可确认关系（confirmed）：\")\n",
            "display(result.pairwise.confirmed[[\"label_i\", \"label_j\", \"diff\", \"ci_low\", \"ci_high\", \"decision\"]].round(4))",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 3. 保守排名置信区间 $[1+B_i,\\ K-W_i]$\n",
            "\n",
            "`point_rank` 为点排名（并列共享名次，`point_tie` 显式标记）；\n",
            "`rank_low/rank_high` 是同时覆盖意义上的保守区间——分不出优劣时区间变宽，\n",
            "这是诚实的答案，而不是按点估计强行给出全序。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "ranks = result.ranks.to_dataframe()\n",
            "display(ranks)\n",
            "print(result.metadata[\"rank_note\"])",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 4. K=2 退化检验与手算对照\n",
            "\n",
            "规格验收：K=2 时 $M=1$，$z_{1-\\alpha/(2M)}=z_{1-\\alpha/2}$，\n",
            "同时区间退化为常规两方案比较。下面用手算数字验证内核。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "from scipy.stats import norm\n",
            "\n",
            "sigma = np.array([[0.0025, 0.0018], [0.0018, 0.0016]])\n",
            "r2 = rank_from_estimates({\"A\": 0.5, \"B\": 0.4}, sigma)\n",
            "display(r2.pairwise.to_dataframe().round(6))\n",
            "display(r2.ranks.to_dataframe())\n",
            "se_diff = np.sqrt(0.0025 + 0.0016 - 2 * 0.0018)\n",
            "print(\"手算 se_diff =\", se_diff)\n",
            "print(\"手算 z_{0.975} =\", norm.ppf(0.975), \" 内核 z_crit =\", r2.pairwise.z_crit)\n",
            "assert abs(r2.pairwise.z_crit - norm.ppf(0.975)) < 1e-12",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 5. 真并列红线：`equal_effects` 场景\n",
            "\n",
            "三个 Treatment 的潜在结果分布完全相同。此时真实名次不唯一（每个臂的\n",
            "真实名次范围都是 $[1,K]$），正确输出是**没有可确认关系、排名区间保持全宽**。\n",
            "若某次抽样里噪声“制造”出确定排名，会以 `confirmed` 对的形式暴露出来，\n",
            "而不会被包装成结论。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "ds_eq = make_dataset(\"equal_effects\", n=1500, random_state=7)\n",
            "result_eq = CausalRank(\n",
            "    treatment=\"treatment\",\n",
            "    outcome=\"outcome\",\n",
            "    confounders=[\"x1\", \"x2\", \"x3\", \"x4\"],\n",
            "    cross_fit=3,\n",
            "    n_bootstrap=80,\n",
            "    bootstrap_min_successes=30,\n",
            "    random_state=7,\n",
            ").fit(ds_eq.df)\n",
            "display(result_eq.ranks.to_dataframe())\n",
            "print(\"confirmed pairs:\", int(result_eq.pairwise.to_dataframe()[\"confirmed\"].sum()))",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 6. 边界与红线\n",
            "\n",
            "1. `inconclusive` **不是**相等的证据；“差异不显著”与“效果相等”是两个命题；\n",
            "2. 排名区间依赖**同时**差值区间与渐近正态近似，不是无条件保证；\n",
            "   用 `scripts/run_rank_validation.py` 做覆盖率与假确认率的模拟检验；\n",
            "3. 不要把 bootstrap 里点排名第一的频次当“最优概率”；\n",
            "4. 单项 CI 重叠不能代替两两显著性检验；\n",
            "5. 不变性（规格验收）：Treatment 重排只置换行；效果加共同常数不改变排名结论；\n",
            "   `higher_is_better=False` 时优劣方向与名次同步翻转。\n",
            "\n",
            "定义与协议见 `docs/methodology.md` 与 `docs/simulation_protocol.md`。",
        ],
    },
]


CELLS_04: list[dict[str, object]] = [
    {
        "cell_type": "markdown",
        "source": [
            "# CausalRank — 从 CSV 到静态 HTML 报告（Phase 4）\n",
            "\n",
            "本 Notebook 演示 Phase 4 的分发与可视化：\n",
            "\n",
            "1. **CSV 往返**：从 CSV 读入数据 → 完整分析（干净环境安装后的典型用法）；\n",
            "2. `result.to_html(path)`：**静态** HTML 报告（无服务器、无 JS、无外部资源、\n",
            "   零新增依赖，图表为内联 SVG）；\n",
            "3. `result.plot(kind)`：5 种可单独保存的图（estimates / ranks / balance / ess / overlap）；\n",
            "4. **可追溯性**：报告里的每个数字都来自 `result` 对象（同一舍入规则），\n",
            "   渲染器不重新拟合、不重新计算任何统计量；\n",
            "5. **诚实形态**：质量门控横幅、结构化警告、推断阻断状态、公式、tie 约定、\n",
            "   因果识别免责声明全部随报告输出——报告不会比 result 对象更“干净”。\n",
            "\n",
            "> **红线不变**：`inconclusive` 不是相等证据；点排名不是最优概率；\n",
            "> 诊断不能验证无未观测混杂；仅用于研究。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "from pathlib import Path\n",
            "\n",
            "import pandas as pd\n",
            "\n",
            "from causalrank import CausalRank\n",
            "from causalrank.simulation import make_dataset\n",
            "\n",
            "pd.set_option(\"display.width\", 120)\n",
            "out_dir = Path(\"artifacts\")\n",
            "out_dir.mkdir(exist_ok=True)",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 1. CSV 往返：模拟数据落盘再读入\n",
            "\n",
            "真实用户通常从 CSV/Parquet 开始。先把合成数据写到磁盘再读回，\n",
            "模拟“干净环境 + 用户自己的文件”的用法。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "ds = make_dataset(\"confounded\", n=2000, random_state=2026)\n",
            "csv_path = out_dir / \"demo_data.csv\"\n",
            "ds.df.to_csv(csv_path, index=False)\n",
            "df = pd.read_csv(csv_path)\n",
            "print(\"loaded from\", csv_path, \"->\", df.shape)\n",
            "df.head(3)",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 2. 完整分析（估计 + 联合 Bootstrap + 排名推断）",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "model = CausalRank(\n",
            "    treatment=\"treatment\",\n",
            "    outcome=\"outcome\",\n",
            "    confounders=[\"x1\", \"x2\", \"x3\", \"x4\"],\n",
            "    cross_fit=3,\n",
            "    n_bootstrap=80,\n",
            "    confidence_level=0.95,\n",
            "    random_state=2026,\n",
            ")\n",
            "result = model.fit(df)\n",
            "print(\"inference_available:\", result.inference_available, \"| gate:\", result.diagnostics.gate)\n",
            "result.effects.to_dataframe().round(4)",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 3. 生成静态 HTML 报告\n",
            "\n",
            "`result.to_html(path)` 返回 HTML 字符串并写盘。报告自包含：\n",
            "内联 CSS + 内联 SVG，双击即可用浏览器打开，不含脚本与外部资源。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "report_path = out_dir / \"causalrank_report.html\"\n",
            "html = result.to_html(report_path)\n",
            "print(f\"wrote {report_path} ({len(html):,} bytes)\")\n",
            "print(\"script tags:\", html.lower().count(\"<script\"))\n",
            "# 可追溯性自检：报告包含 result 的每个 mu 值（同一舍入）\n",
            "est = result.effects.to_dataframe()\n",
            "assert all(f\"{m:.4f}\" in html for m in est[\"mu\"])\n",
            "assert all(f\"{s:.4f}\" in html for s in est[\"se\"])\n",
            "print(\"traceability check: every mu/se value present in the report ✓\")",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 4. 单独的图：`result.plot(kind)`\n",
            "\n",
            "5 种图都返回内联 SVG 的 `Figure`（Jupyter 里直接内联显示；`fig.save(path)` 可存\n",
            "独立 `.svg` 或 `.html`）。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "for kind in (\"estimates\", \"ranks\", \"balance\", \"ess\", \"overlap\"):\n",
            "    fig = result.plot(kind)\n",
            "    p = fig.save(out_dir / f\"fig_{kind}.svg\")\n",
            "    print(f\"{kind:10s} -> {p} ({len(fig.svg):,} bytes)\")\n",
            "result.plot(\"estimates\")",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 5. 推断未启用 / 被阻断时的诚实报告\n",
            "\n",
            "报告不会伪造不确定性：`n_bootstrap=None` 或成功率门槛未满足时，\n",
            "排名/两两区间章节会退化为显式说明（“Not available”），\n",
            "而不是省略或给出错误数字。",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "plain = CausalRank(\n",
            "    treatment=\"treatment\",\n",
            "    outcome=\"outcome\",\n",
            "    confounders=[\"x1\", \"x2\", \"x3\", \"x4\"],\n",
            "    cross_fit=3,\n",
            "    random_state=2026,\n",
            ").fit(df)\n",
            "plain_path = out_dir / \"causalrank_report_no_inference.html\"\n",
            "plain_html = plain.to_html(plain_path)\n",
            "print(\"wrote\", plain_path)\n",
            "print(\"mentions blocked/absent inference:\", \"Not available in this result\" in plain_html)",
        ],
    },
    {
        "cell_type": "markdown",
        "source": [
            "## 6. 边界\n",
            "\n",
            "1. 报告只是 result 对象的**视图**：它不计算任何新统计量，也不验证识别假设；\n",
            "2. HTML 里的置信区间/排名结论依赖联合 Bootstrap 协方差与渐近正态近似，\n",
            "   覆盖率以模拟检验为准（`scripts/run_se_validation.py` / `run_rank_validation.py`）；\n",
            "3. 质量门控 `pass` 只说明本样本没有触发经验警告，**不**说明因果假设成立；\n",
            "4. 同一 result 两次渲染逐字节一致（无渲染时刻时钟，时间戳取 `metadata['created_at']`）。",
        ],
    },
]


def _notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": [
            {
                "id": f"cell-{i:02d}",
                "cell_type": cell["cell_type"],
                "metadata": {},
                "source": cell["source"],
                **(
                    {"execution_count": None, "outputs": []}
                    if cell["cell_type"] == "code"
                    else {}
                ),
            }
            for i, cell in enumerate(cells)
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build() -> None:
    root = Path(__file__).resolve().parents[1] / "examples"
    root.mkdir(parents=True, exist_ok=True)
    for filename, cells in (
        ("01_gbm_iptw_from_raw_data.ipynb", CELLS),
        ("02_joint_bootstrap.ipynb", CELLS_02),
        ("03_causal_ranking.ipynb", CELLS_03),
        ("04_html_report.ipynb", CELLS_04),
    ):
        out = root / filename
        out.write_text(
            json.dumps(_notebook(cells), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print("wrote", out)


if __name__ == "__main__":
    build()
