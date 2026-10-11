# 参考文献与使用约定

1. McCaffrey, D. F., Griffin, B. A., Almirall, D., Slaughter, M. E., Ramchand, R., &
   Burgette, L. F. (2013). **A tutorial on propensity score estimation for multiple
   treatments using generalized boosted models**. *Statistics in Medicine*, 32,
   3388–3414. https://doi.org/10.1002/sim.5753
   → 多 Treatment 倾向得分的 GBM 思路来源。**差异声明**：该文的广义提升模型常采用
   one-vs-rest 构造；CausalRank v0.1 明确采用**联合多分类**概率模型，
   不宣称与该文实现一致。

2. Mogstad, M., Romano, J. P., Shaikh, A. M., & Wilhelm, D. (2024). **Inference for
   Ranks**. *Review of Economic Studies*. https://doi.org/10.1093/restud/rdad006
   → 排名推断理论（Phase 3 参考）。CausalRank 的 Bonferroni 保守排名区间是**独立实现**，
   不宣称原创，也不宣称等同于该文的 max-T / stepdown 方法。

3. Chetverikov, D., Mogstad, M., Morgen, P., Romano, J. P., Shaikh, A. M., & Wilhelm, D.
   (2024). **csranks: An R Package for Estimation and Inference Involving Ranks**.
   https://arxiv.org/abs/2401.15205
   → 参考实现对照对象（Phase 3+）。仅在遵守其许可证的前提下做独立实现与行为对照。

4. `csranks` 文档 / 实现：https://danielwilhelm.github.io/R-CS-ranks/

5. scikit-learn 官方文档：https://scikit-learn.org/stable/
   （`HistGradientBoostingClassifier`、`ColumnTransformer`、`Pipeline`、`StratifiedKFold`）

6. Austin, P. C. (2009). **Balance diagnostics for comparing the distribution of baseline
   covariates between treatment groups in propensity-score matched samples**.
   *Statistics in Medicine*, 28, 3083–3107. → SMD 分母使用未加权 pooled SD 的建议来源
   （本项目在多 Treatment 加权语境下的定义见 `docs/methodology.md` §4.3）。

## 使用约定

- 实现前核实论文的模型设定、权重定义、渐近条件与软件接口；不因标题相似就宣称等同实现；
- 不把成熟的排名推断方法宣称为原创；
- 所有数据为自行生成的合成数据（`causalrank.simulation`），不包含任何公司数据、
  内部代码、密钥或业务配置；
- 第三方包（numpy/pandas/scipy/scikit-learn）按其各自许可证使用（BSD/Apache-2.0 等）。
