# 贡献约定

## Git 提交信息

- **提交信息使用中文**（标题一行概括，正文按需要补充细节）。
- 标题建议使用祈使语气，说明“做了什么”，例如：
  - `修复共同支持漏报警：新增目标样本低概率占比检查`
  - `为 SMD 方差计算增加列中心化与两遍公式`
  - `补充 dtype 归一化的端到端回归测试`
- 涉及缺陷修复的提交，正文注明复现方式与回归测试位置。

## 开发流程

```bash
uv sync                 # 安装 runtime + dev 依赖
uv run pytest -q        # 全部测试须通过（不要为变绿而跳过测试）
uv run ruff check .     # lint
uv run mypy             # 类型检查
uv build                # 打包检查
```

- 统计方法的任何改动都要同步更新 `docs/methodology.md`（估计目标、公式、假设、数值边界）；
- 缺陷修复必须附带能复现该缺陷的回归测试；
- 不提前实现 AIPW、Overlap Weighting、排名推断、可视化报告等后续阶段内容；
- 阶段进展记录在 `docs/agent_progress.md`。
