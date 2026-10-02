# 🚀 daily-arXiv-ai-enhanced

> [!CAUTION]
> 若您所在法域对学术数据有审查要求，谨慎运行本代码；任何二次分发版本必须履行合规审查（包括但不限于原始论文合规性、AI合规性）义务，否则一切法律后果由下游自行承担。

> [!CAUTION]
> If your jurisdiction has censorship requirements for academic data, run this code with caution; any secondary distribution version must remove the entrance accessible to China and fulfill the content review obligations, otherwise all legal consequences will be borne by the downstream.


This innovative tool transforms how you stay updated with arXiv papers by combining automated crawling with AI-powered summarization.


## ✨ Key Features

🎯 **Zero Infrastructure Required**
- Leverages GitHub Actions and Pages - no server needed
- Completely free to deploy and use

🤖 **Smart AI Summarization**
- Daily paper crawling with DeepSeek-powered summaries
- Cost-effective: Only ~0.2 CNY per day

💫 **Smart Reading Experience**
- Personalized paper highlighting based on your interests
- Cross-device compatibility (desktop & mobile)
- Local preference storage for privacy
- Flexible date range filtering

🧩 **SKILL System**
- Plug-and-play skill modules for customizing paper filtering

⚙️ **Easy Preference Export & Integration**
- One-click copy in Settings to export your keywords and authors configuration
- Seamlessly combine exported preferences with SKILL for reproducible and shareable setups

👉 **[Try it now!](https://dw-dengwei.github.io/daily-arXiv-ai-enhanced/)** - No installation required



https://github.com/user-attachments/assets/b25712a4-fb8d-484f-863d-e8da6922f9d7




# How to use
This repo will daily crawl arXiv papers about **cs.CV, cs.GR, cs.CL and cs.AI**, and use **DeepSeek** to summarize the papers in **Chinese**.
If you wish to crawl other arXiv categories, use other LLMs, or other languages, please follow the instructions.
Otherwise, you can directly use this repo in https://dw-dengwei.github.io/daily-arXiv-ai-enhanced/. Please star it if you like :)

**Instructions:**
1. Fork this repo to your own account and delete my own information in [buy-me-a-coffee](./buy-me-a-coffee/README.md).
2. Go to: your-own-repo -> Settings -> Secrets and variables -> Actions
3. Go to Secrets. Secrets are encrypted and used for sensitive data
4. Create two repository secrets named `OPENAI_API_KEY` and `OPENAI_BASE_URL`, and input corresponding values.
5. [Optional] Set a password in `secrets.ACCESS_PASSWORD` if you do not wish others to access your page. (see https://github.com/dw-dengwei/daily-arXiv-ai-enhanced/pull/64)
6. Go to Variables. Variables are shown as plain text and are used for non-sensitive data
7. Create the following repository variables:
   1. `CATEGORIES`: separate the categories with ",", such as "cs.CL, cs.CV"
   2. `LANGUAGE`: such as "Chinese" or "English"
   3. `MODEL_NAME`: such as "deepseek-chat"
   4. `EMAIL`: your email for push to GitHub
   5. `NAME`: your name for push to GitHub
8. Go to your-own-repo -> Actions -> arXiv-daily-ai-enhanced
9. You can manually click **Run workflow** to test if it works well (it may take about one hour). By default, this action will automatically run every day. You can modify it in `.github/workflows/run.yml`
10. Set up GitHub pages: Go to your own repo -> Settings -> Pages. In `Build and deployment`, set `Source="Deploy from a branch"`, `Branch="main", "/(root)"`. Wait for a few minutes, go to https://\<username\>.github.io/daily-arXiv-ai-enhanced/. Please see this [issue](https://github.com/dw-dengwei/daily-arXiv-ai-enhanced/issues/14) for more precise instructions.

### 个人研究日报与 DeepSeek 配置

本 fork 在 `config/directions.yaml` 集中设置五个兴趣方向、主次层级、稳定子主题、权重和名额。
主要方向为世界模型（1.6）、视频生成（1.5）、音视频联合生成（1.5）；辅助方向为统一理解与生成（1.1）、连续语言模型（1.0）。视频世界模型是世界模型内最高优先子主题，方向权重用于排序，不是收录比例。
默认 `SELECTION_MODE=semantic`：全量抓取并去重 → 免费读取标题和完整摘要做本地预筛 → 最多 150 篇模型轻筛 → 最多 50 篇详细摘要（辅助合计最多 10 篇，各预留 5） → 未取得详细名额的相关／不确定论文保留简讯 → 综合重要性排序 → 3 篇 PDF 精读。主要方向候选充足时精读至少 2 篇。
两个辅助方向各预留 5 篇，不足的名额转给另一方向；跨方向论文按主要归属计入一次。详细处理失败或判为不相关不自动补位。筛选失败或无法确定归属的论文另外进入详细阶段兜底，不占辅助名额。
每天精读先取主要方向最高的 2 篇，再从剩余详细论文中取最高的 1 篇；主要方向不足时补足。简讯与处理失败论文不参与精读。
本地候选上限在首次模型调用前生效。未被选中的论文保留“未进入模型审阅”记录；模型明确判为不相关的论文不生成详细摘要。
轻量筛选同时给出基于摘要的初步相关性与研究价值分，默认排序采用模型分 70%＋个人启发式分 30%；这不是全文质量认证。
内部保留未封顶的本地分数用于同分排序，ELF 等关键词按词边界匹配。
`SELECTION_MODE=legacy` 使用旧的预筛流程；当前推荐使用带本地候选限制的 semantic 模式。
精读保留动机、方法、启发、实验、局限与个人收获等栏目，英文概述每节最多一句。

纯图像生成、普通 VLM 理解、纯语音或音乐生成，必须在摘要中体现对目标方向的明确迁移价值才收录，并记录依据。统一理解与生成要求共享模型、表征或训练目标，不能泛化为所有 VLM。
抓取分类保留 `cs.CV/cs.CL/cs.AI/cs.LG/cs.MM/cs.RO`，增加 `cs.SD/eess.AS`；所有方向均参与本地匹配与名额分配，选中的候选再接受轻量判断；辅助权重低不等于不相关。

本次先做不调用模型的离线验证，再部署代码与字段；日报保持暂停，在线诊断需单独授权。
只有修改 `MODEL_NAME` 不够：现有各阶段的 `*_MODEL_NAME` 会覆盖它。

进入仓库 **Settings → Secrets and variables → Actions**。
[Secrets 设置](https://github.com/luokairo/daily-arXiv-ai-enhanced/settings/secrets/actions)：

| Secret | 值 |
| --- | --- |
| `OPENAI_API_KEY` | DeepSeek 官方 API Key，在 GitHub 中手动填写 |
| `OPENAI_BASE_URL` | `https://api.deepseek.com` |

[Variables 设置](https://github.com/luokairo/daily-arXiv-ai-enhanced/settings/variables/actions)：

| Variable | 推荐值／默认值 | 说明 |
| --- | --- | --- |
| `MODEL_NAME` | `deepseek-flash` | 统一默认模型 |
| `DETAIL_MODEL_NAME` | `deepseek-flash` | 将原 GLM 值替换；未设置则继承 `MODEL_NAME` |
| `FILTER_MODEL_NAME` | `deepseek-flash` | 轻量筛选模型；未设置则继承 `MODEL_NAME` |
| `IMPORTANCE_MODEL_NAME` | `deepseek-flash` | 可选模型评分；未设置则继承 `MODEL_NAME` |
| `DEEP_READ_MODEL_NAME` | `deepseek-flash` | PDF 精读；未设置则继承 `MODEL_NAME` |
| `DEEPSEEK_THINKING` | `false` | 常规阶段显式关闭思考 |
| `DEEP_READ_THINKING` | `false` | 将原 `true` 改为 `false`；全文分析仍保留 |
| `ENABLE_DEEP_READ` | `true` | 本次默认启用精读；`false` 可暂停 |
| `DAILY_DEEP_READ_TOP_K` | `3` | 每日精读数量 |
| `SELECTION_MODE` | `semantic` | 先做本地候选限制，再轻量语义筛选；`legacy` 为旧流程 |
| `SECONDARY_DETAIL_LIMIT` | `10` | 辅助方向详细摘要总额；两方向按配置各预留 5，未入选仍有简讯 |
| `MAX_AI_CANDIDATES` | `150` | 首次模型调用前的候选总上限；必须为正整数 |
| `MAX_DETAIL_ITEMS` | `50` | 两种模式均生效的详细请求总上限；必须为正整数 |
| `USE_MODEL_FILTER` | `false` | 仅 legacy 模式生效；semantic 模式始终筛选已选候选 |
| `USE_MODEL_IMPORTANCE` | `false` | 复用轻量筛选中的初步价值判断；true 则额外调用评分模型 |
| `FILTER_MAX_OUTPUT_TOKENS` | `512` | 简短筛选决定、理由与评分的单次输出上限 |
| `DETAIL_MAX_OUTPUT_TOKENS` | `2500` | 每次摘要输出上限 |
| `DEEP_READ_MAX_OUTPUT_TOKENS` | `6000` | 每次精读输出上限 |
| `DEEP_READ_MAX_CONTEXT_CHARS` | `40000` | PDF 上下文字符数，含章节标题；不等于 token 数 |
| `LANGUAGE` | `Chinese` | 摘要语言与输出文件名 |
| `NAME` | `luokairo` | 自动提交身份；未设置时使用仓库 owner |

保留现有 `EMAIL`、`CATEGORIES`。`FILTER_MAX_WORKERS=4`、`DETAIL_MAX_WORKERS=2`、
`IMPORTANCE_MAX_WORKERS=1`、`DEEP_READ_MAX_WORKERS=1`；并发影响速度，不会直接减少 token。
兴趣配置存在时，抓取分类从兴趣方向配置合并得到，优先于 `CATEGORIES`。
`deepseek-v4*` 旧名称仍识别思考开关，推荐官方当前名称 `deepseek-flash`。
可选开启思考时，`DEEPSEEK_REASONING_EFFORT` 默认 `low`，仅允许 `low/high/max`。
原 GLM 配置仍兼容；切换至 DeepSeek 后无需设置 `GLM_REASONING_EFFORT`。

**如何节省 token**

- `config/local_recall.yaml` 的扩展词表只在本地读取，不发给模型。本地规则按概念去重、标题优先，条件组合限制为同一句、40 词以内；分数代表兴趣匹配，不认证论文质量。
- 模型轻量筛选只发送研究方向说明、子主题名称、标题、分类和完整摘要，不发送动态 taxonomy、作者列表或重复关键词。
- 相关／不确定论文再调用详细摘要；默认复用筛选评分，不额外调用评分模型。
- 主要或辅助方向超出详细名额时直接复用轻量筛选的一句话简讯、理由和评分，不追加摘要请求。
- 保留全部预设子主题，每方向最多传入 5 个常用动态子主题；不重复发送计数和关键词。
- PDF 上下文去重并按章节分配额度，方法、实验获得较多额度，局限等章节不被前文挤掉。
- 摘要和精读的成功结果保存在 `data/ai_cache/`，随 `data` 分支恢复；相同论文、输入、模型、
  参数和实际提示上下文可复用。输入或配置变化会失效。失败与仅摘要降级的精读不缓存。
- Actions Step Summary 显示实际模型调用（含重试）、缓存命中、失败条目和输入／输出 token；
  API 未返回用量的调用标注“未知”。成功运行的用量另存 `data/run_metrics/`，不进入网页文件列表。

上下文上限和输出上限是单次预算，不是每日账单硬上限；重试也可能计费。不承诺固定节省比例。
缓存未命中、不含重试、关闭额外评分时，默认最多 150＋50＋3＝203 次模型调用。数量上限不是固定 token 总量；修改候选／详细数量不会清空实际输入相同的缓存。
显式设置 `SELECTION_MODE=semantic`、`MAX_AI_CANDIDATES=150`、`MAX_DETAIL_ITEMS=50`。日报当前保持暂停，部署配置不会自动恢复定时运行。

本地候选预留为世界模型 40、视频生成 40、音视频生成 30、统一理解生成 10、连续语言模型 10、探索 20；下调上限按最大余数法缩放。探索由剩余弱信号优先和公告日期＋ID 稳定哈希各占一半。不足名额转给剩余有效方向候选，不强行凑满。
详细名额先保留辅助 10（各 5，缺额在辅助间转移），主要／兜底最多使用剩余 40；辅助未使用的名额转给主要／兜底。处理失败或判为不相关不补位。

[9 月 30 日离线比较](docs/local-recall-validation.md)只检查代表样例与分配，没有人工全量标签，不能保证零遗漏。复现方式：

```bash
uv run python scripts/evaluate_local_recall.py --data <原始日期.jsonl> --output local_reports/diagnostics/local-budget
```

**失败与发布**

鉴权、模型和参数错误立即停止；限流、连接超时与服务器错误最多重试两次，不自动提高输出预算。
个别轻量筛选失败进入有上限的详细兜底池；没有兜底名额时进入待复核清单，不伪造简讯。全体筛选失败则终止。
每篇论文的筛选决定、理由、评分及详细处理状态保存在 `data/run_metrics/<日期>-selection.json`。
记录包含全量本地匹配证据，并明确区分未进入模型审阅（`not_reviewed`）、模型判定不相关、详细名额不足（`quota_deferred`）、处理失败和待复核（`awaiting_review`）。Actions 摘要同时显示各方向轻量判断、详细请求、成功摘要、简讯与精读候选数量。
每阶段开始输出总数，每完成 5 篇或 15 秒输出进度，等待请求时每 30 秒输出状态；缓存、审计和用量逐篇原子保存。
失败运行的筛选记录可从 Actions `selection-audit-<run_id>` artifact 下载。工作流完成或正常取消时用 `always()` 尽力上传 `ai-recovery-<日期>-<run_id>` 恢复包（保留 14 天）。后续运行合并最近三个匹配恢复包的成功缓存及已验证原始快照，随后仍校验真实模型请求签名；不恢复半成品日报用于发布。强制终止前未上传的部分不保证保留。部分摘要失败标记 failed，不能当作不相关；重新运行原始数据时失败请求会重试，成功请求可命中缓存。
个别摘要解析失败不进入正常报告，全部请求失败则终止；正常筛选后无相关论文允许空结果。
可选评分的个别失败使用本地评分，整个评分阶段失败则终止。精读全部生成失败时也终止，
PDF 提取失败的摘要分析明确标为 `abstract_fallback`，不能当作全文精读。
工作流和 `run.sh` 将日报与 taxonomy 写入暂存目录，通过全部阶段后才替换正式产物。
缓存文件不含 API Key。直接运行 `ai/enhance.py` 只保证增强阶段成功后再写入；完整发布保护请使用流水线。
方向范围变化时旧动态子主题清除，保留有效预设分类，并将迁移前 taxonomy 保存到 `data/run_metrics/taxonomy-before-<摘要值>.json`。旧日报保持原样，历史缺少 `report_level` 时按详细摘要显示。
日报与网页分为主要方向详细摘要、辅助方向精选摘要、相关论文简讯（含主次标签），组内按重要性排序；简讯明确标注“未做详细分析”，不显示空的分析栏目。

**小规模验证**

Actions → **DeepSeek Pipeline Diagnostic** → Run workflow：选择 `data` 分支中存在的原始论文日期，
可指定逗号分隔的论文 ID；未指定时选择本地优先级最高的候选。默认只处理 1 篇、最多 1 篇精读；可设置 `sample_limit`（1–20）、`secondary_detail_limit`（0–20）及 `deep_read_top_k`（1–3）验证主次分配，
不执行额外模型评分，不提交任何分支、不发布 Pages；结果和用量下载自 `deepseek-diagnostic` artifact。
若摘要模型判定该论文不相关，精读会正常跳过。首次部署后应检查摘要字段、精读来源标记及实际用量。
该入口固定使用 `deepseek-flash`，仍需先更新官方 API 的两个 Secrets。
诊断入口读取已保存原始数据，不进行历史抓取。正式 `arXiv-daily-ai-enhanced` 工作流的 `target_date` 现在表示 **arXiv 公告列表日期**（不是论文提交日期，也不是运行日期），例如 `2026-09-30`。留空时处理列表中最新公告日，周末不会把旧列表重新命名为当天。

日期抓取通过官方 `/catchup/<category>/<日期>?abs=True` 历史补看页核对公告日期、所有分页及各组数量，直接获取完整摘要，保留新投稿与跨分类收录，排除论文替换更新，并进行跨分类去重。网页请求间隔至少 15 秒。原始数据包含 `announcement_date` 和元数据来源口径；来源、分类数量及原始文件 SHA-256 保存在 `data/run_metrics/<日期>-crawl.json`。不再用前七天 ID 删除该日已核对的公告成员。

**支持范围：**官方补看页面提供的最近约 90 天，以及本项目已经保存且校验通过的历史快照。更早日期或官方历史页面不可用时，若没有可信快照，明确失败，不回退 `/new`，也不把提交日期冒充公告日期。旧的无校验记录文件不能作为可信历史快照。元数据与精读全文使用当前可获取版本，不能视作当时版本的重建。

本地可运行 `python scripts/crawl_dated.py --date 2026-09-30` 仅抓取核对后的原始摘要（不调用模型）。完整流程仍可通过 `TARGET_DATE=2026-09-30 bash run.sh` 运行。只有抓取完整、AI 阶段和产物校验通过后，Actions 才会发布到 `data` 分支。

本地无付费 API 调用的回归检查：

```bash
uv sync --locked
.venv/bin/python -m unittest discover -s tests -v
node tests/test_date_range.cjs
node tests/test_report_levels.cjs
```

# Plans
See https://github.com/users/dw-dengwei/projects/3

# Contributors
Thanks to the following special contributors for contributing code, discovering bugs, and sharing useful ideas for this project!!!
<table>
  <tbody>
    <tr>
      <td align="center" valign="top">
        <a href="https://github.com/JianGuanTHU"><img src="https://avatars.githubusercontent.com/u/44895708?v=4" width="100px;" alt="JianGuanTHU"/><br /><sub><b>JianGuanTHU</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/Chi-hong22"><img src="https://avatars.githubusercontent.com/u/75403952?v=4" width="100px;" alt="Chi-hong22"/><br /><sub><b>Chi-hong22</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/chaozg"><img src="https://avatars.githubusercontent.com/u/69794131?v=4" width="100px;" alt="chaozg"/><br /><sub><b>chaozg</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/quantum-ctrl"><img src="https://avatars.githubusercontent.com/u/16505311?v=4" width="100px;" alt="quantum-ctrl"/><br /><sub><b>quantum-ctrl</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/Zhao2z"><img src="https://avatars.githubusercontent.com/u/141019403?v=4" width="100px;" alt="Zhao2z"/><br /><sub><b>Zhao2z</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/eclipse0922"><img src="https://avatars.githubusercontent.com/u/6214316?v=4" width="100px;" alt="eclipse0922"/><br /><sub><b>eclipse0922</b></sub></a><br />
      </td>
    </tr>


  </tbody>
  <tbody>
   <tr>
      <td align="center" valign="top">
        <a href="https://github.com/xuemian168"><img src="https://avatars.githubusercontent.com/u/38741078?v=4" width="100px;" alt="xuemian168"/><br /><sub><b>xuemian168</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/Lrrrr549"><img src="https://avatars.githubusercontent.com/u/71866027?v=4" width="100px;" alt="Lrrrr549"/><br /><sub><b>Lrrrr549</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/AinzRimuru"><img src="https://avatars.githubusercontent.com/u/59441476?v=4" width="100px;" alt="AinzRimuru"/><br /><sub><b>AinzRimuru</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/fengxueguiren"><img src="https://avatars.githubusercontent.com/u/153522370?v=4" width="100px;" alt="fengxueguiren"/><br /><sub><b>fengxueguiren</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/zerocpp"><img src="https://avatars.githubusercontent.com/u/2630297?v=4" width="100px;" alt="fengxueguiren"/><br /><sub><b>zerocpp</b></sub></a><br />
      </td>
   </tr>
  </tbody>
</table>

# Acknowledgement
We sincerely thank the following individuals and organizations for their promotion and support!!!
<table>
  <tbody>
    <tr>
      <td align="center" valign="top">
        <a href="https://x.com/GitHub_Daily/status/1930610556731318781"><img src="https://pbs.twimg.com/profile_images/1660876795347111937/EIo6fIr4_400x400.jpg" width="100px;" alt="Github_Daily"/><br /><sub><b>Github_Daily</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://x.com/aigclink/status/1930897858963853746"><img src="https://pbs.twimg.com/profile_images/1729450995850027008/gllXr6bh_400x400.jpg" width="100px;" alt="AIGCLINK"/><br /><sub><b>AIGCLINK</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://www.ruanyifeng.com/blog/2025/06/weekly-issue-353.html"><img src="https://avatars.githubusercontent.com/u/905434" width="100px;" alt="阮一峰的网络日志"/><br /><sub><b>阮一峰的网络日志 <br> 科技爱好者周刊 <br> （第 353 期）</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://hellogithub.com/periodical/volume/111"><img src="https://github.com/user-attachments/assets/eff6b6dd-0323-40c4-9db6-444a51bbc80a" width="100px;" alt="《HelloGitHub》第 111 期"/><br /><sub><b>《HelloGitHub》<br> 月刊第 111 期</b></sub></a><br />
      </td>
    </tr>
  </tbody>
</table>


# Star history

[![Stargazers over time](https://starchart.cc/dw-dengwei/daily-arXiv-ai-enhanced.svg?variant=adaptive)](https://starchart.cc/dw-dengwei/daily-arXiv-ai-enhanced)

# Buy me a coffee
[here](./buy-me-a-coffee/README.md)
