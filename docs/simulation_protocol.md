# 模拟验证协议（DGP 与 Monte Carlo 基准）

> 实现：`src/causalrank/simulation/{dgp.py,benchmark.py}`，
> 研究脚本：`scripts/run_simulation.py`，快测试：`tests/test_dgp.py`、`tests/test_benchmark.py`。

## 1. DGP 结构

协变量（$z = [x_1, x_2, x_3, x_{4,\mathrm{code}}]$）：

- $x_1, x_2 \sim N(0,1)$；$x_3 \sim \mathrm{Bernoulli}(p)$；$x_4 \sim \mathrm{Categorical}(\pi)$（3 水平）；
- 可选 MCAR 缺失（`missing_rate`）施加于 $x_2$ 与 $x_4$，用于验证插补管线。

Treatment 分配（多项 softmax）：

$$P(T=a\mid X) = \mathrm{softmax}_a\Big(\gamma_{0a} + \gamma_a^\top z + s_{\mathrm{ov}} \cdot \omega_a \cdot x_1\Big)$$

- `confounding_strength` 由 $\gamma_a$ 的量级体现（场景预设中给出）；
- `overlap_strength` $s_{\mathrm{ov}}$ 与 $\omega_a$ 用于构造弱共同支持。

潜在结果（只观测 $Y = Y(T)$）：

$$P(Y(a)=1\mid X) = \sigma\big(\beta_0 + \beta^\top z + \tau_a + \kappa \cdot x_1 \cdot \tau_a\big)$$

- $\tau_a$：Treatment 效应；`equal_effects` 场景令 $\tau_a$ 全相等（真实并列）；
- $\kappa$：效应异质性（`mixed_missing` 启用）。

## 2. 真值与 Monte Carlo 误差

$$\mu_{\mathrm{true}}[a] = E_X\big[P(Y(a)=1\mid X)\big] \approx \frac{1}{N_{\mathrm{ref}}}\sum_{j=1}^{N_{\mathrm{ref}}} P(Y(a)=1\mid X_j^{\mathrm{ref}})$$

- 参考协变量样本 $X^{\mathrm{ref}}$ **独立于**估计样本，默认 $N_{\mathrm{ref}} = 200{,}000$，
  固定 `reference_seed = 20260101`；两者均记录在 `SyntheticDataset` 与基准输出中；
- MC 标准误 $\mathrm{mc\_se} = \mathrm{sd}(P)/\sqrt{N_{\mathrm{ref}}}$（默认约 $3\times10^{-4}$），
  远小于任何有意义的效应差异；
- `population_mean()` 对 $(N_{\mathrm{ref}}, \mathrm{seed})$ 缓存，同一配置的所有重复实验共享同一真值；
- $N_{\mathrm{ref}} < 10{,}000$ 会被拒绝（误差控制下限）。

## 3. 场景矩阵（8 个预设）

| 场景 | 目的 | 关键参数 |
|---|---|---|
| `randomized` | 基本公式正确性；原始均值近似无偏 | $\gamma_a = 0$ |
| `confounded` | 有混杂且倾向可估：IPTW 恢复 `mu_true` | 中等 $\gamma_a$，重叠良好 |
| `strong_confounding` | 权重与 SMD 诊断压力测试 | $\gamma_a$ 约 3 倍 |
| `weak_overlap` | ESS 崩塌、极端权重、警告触发 | $s_{\mathrm{ov}} = 1.8$ |
| `rare_treatment` | 稀有臂：折划分与后续 Bootstrap 有效性 | 臂 C 基线 logit $-2.4$ |
| `equal_effects` | 真实并列：不得凭噪声宣称确定排名 | $\tau_a = 0.2$ 全相等 |
| `close_effects` | 排名置信集宽度与功效（Phase 3 用） | $\tau = (0, 0.08, -0.06)$ |
| `mixed_missing` | 编码/缺失/管线；4 臂 + 效应异质性 | `missing_rate=0.10`, $\kappa=0.15$ |

## 4. Monte Carlo 指标

- `Bias(mu_a)`、RMSE、跨重复实验标准差；原始组均值的同名指标（对照）；
- 平均 ESS、SMD、极端权重警告触发率（诊断有效性）；
- 拟合失败率、Bootstrap 成功率（逐轮 `status/error` 记录）；
- **SE 校准**（`run_se_benchmark`）：`se_ratio = 平均 bootstrap SE / mu_hat 经验标准差`；
  单项 CI **覆盖率**与 MC 误差 $\sqrt{\hat p(1-\hat p)/R}$；
- **排名推断校准**（`run_rank_benchmark`，Phase 3）：排名区间覆盖率（并列感知真值
  `[1+#严格更好, K-#严格更差]`，要求区间覆盖全部可行名次）、区间宽度、
  以及真并列场景（`equal_effects`）下的**假确认率** `any_confirmed_rate`；
- 运行时长与可复现性（固定种子逐位一致）。

Bonferroni 联合排名覆盖率由 `run_rank_benchmark` / `scripts/run_rank_validation.py`
统计；同样必须报告 Monte Carlo 误差，不得因 0.946 vs 0.95 就断定方法失效。

## 5. 快测试 vs 研究基准

- **快测试**（CI 内，`@pytest.mark.slow`）：$R \in \{2, 6\}$、$n \le 4{,}000$，
  只断言方向性结论（偏差随 $n$ 下降、调整优于未调整、随机化场景无偏）；
- **研究基准**（手动）：

```bash
uv run python scripts/run_simulation.py --scenario confounded --n 1500 6000 \
    --replications 20 --cross-fit 5 --out artifacts/benchmark_confounded.csv

uv run python scripts/run_se_validation.py --scenario confounded --n 1200 \
    --replications 24 --bootstrap 150 --cross-fit 5 --n-jobs 4 \
    --out artifacts/se_validation.csv

uv run python scripts/run_rank_validation.py --scenario equal_effects --n 1200 \
    --replications 24 --bootstrap 150 --cross-fit 5 --n-jobs 4 \
    --out artifacts/rank_validation.csv
```

  前者输出逐重复表、臂级 Bias/RMSE 与运行级失败率/门控/告警触发率；
  中者输出 `se_ratio`（bootstrap SE vs 经验标准差）与 CI 覆盖率（含 MC 误差）；
  后者输出排名区间覆盖率（并列感知真值）与真并列下的假确认率。

## 6. 已知的验证局限

- GBM 是非参数 nuisance 估计器，重新拟合的 Bootstrap 理论性质依赖算法稳定性与正则条件；
  本项目用模拟覆盖率检验，而不把任何推断当作通用保证；
- DGP 的函数形式（softmax + logistic）是“可正确设定”的理想化设定；真实数据的
  模型误设风险只能通过稳健性/灵敏度分析评估（后续阶段）；
- 单一 DGP 家族的通过不等于对所有数据生成机制的保证。
