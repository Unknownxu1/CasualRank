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

## 发布

发布由 `.github/workflows/workflow.yml` 在推送 `v*` 标签时自动完成，使用 PyPI Trusted Publishing（OIDC，无需 API token）。

1. 更新 `src/causalrank/_version.py` 中的版本号（唯一版本来源），提交并推送；
2. 打标签并推送，标签必须与版本号一致：

   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```

3. CI 会先跑 lint / 类型检查 / 测试、校验标签与版本号一致、构建并通过 `twine check`，然后上传 PyPI；
4. 到 https://pypi.org/project/causalrank/ 确认新版本，可 `uv pip install causalrank==<版本号>` 验证安装。

注意：

- **PyPI 版本号不可覆盖**，发错只能改版本号重新发；拿不准时先在本地 `uv build` + `uvx twine check --strict dist/*` 验证；
- 首次发布前需在 PyPI 配置 Trusted Publisher（见下）；若修改了 workflow 文件名或路径，需同步更新 PyPI 上的配置。
