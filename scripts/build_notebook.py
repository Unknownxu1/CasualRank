"""Generate examples/01_gbm_iptw_from_raw_data.ipynb (kept reproducible)."""

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
    ):
        out = root / filename
        out.write_text(
            json.dumps(_notebook(cells), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print("wrote", out)


if __name__ == "__main__":
    build()
