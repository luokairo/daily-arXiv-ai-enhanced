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

本 fork 保留 `config/directions.yaml` 中的四个兴趣方向、稳定子主题和个人重要性权重。
默认流水线是：本地兴趣筛选 → 按偏好排序取最多 30 篇 → 模型分类与摘要 → 本地重要性评分 → Top-3 PDF 精读。
`MAX_DETAIL_ITEMS=0` 可取消候选数量限制；独立模型筛选与模型评分仍可选开启。
精读保留动机、方法、启发、实验、局限与个人收获等栏目，英文概述每节最多一句。

**迁移顺序：先部署本次代码，再在正常定时任务运行前统一更新以下配置；先运行诊断，再手动运行日报。**
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
| `FILTER_MODEL_NAME` | `deepseek-flash` | 可选模型筛选；未设置则继承 `MODEL_NAME` |
| `IMPORTANCE_MODEL_NAME` | `deepseek-flash` | 可选模型评分；未设置则继承 `MODEL_NAME` |
| `DEEP_READ_MODEL_NAME` | `deepseek-flash` | PDF 精读；未设置则继承 `MODEL_NAME` |
| `DEEPSEEK_THINKING` | `false` | 常规阶段显式关闭思考 |
| `DEEP_READ_THINKING` | `false` | 将原 `true` 改为 `false`；全文分析仍保留 |
| `ENABLE_DEEP_READ` | `true` | 本次默认启用精读；`false` 可暂停 |
| `DAILY_DEEP_READ_TOP_K` | `3` | 每日精读数量 |
| `MAX_DETAIL_ITEMS` | `30` | 本地优先级排序后的候选上限，`0` 表示无限制 |
| `USE_MODEL_FILTER` | `false` | 不确定候选直接由摘要模型判断相关性 |
| `USE_MODEL_IMPORTANCE` | `false` | 按个人权重进行本地评分，无额外模型调用 |
| `DETAIL_MAX_OUTPUT_TOKENS` | `2500` | 每次摘要输出上限 |
| `DEEP_READ_MAX_OUTPUT_TOKENS` | `6000` | 每次精读输出上限 |
| `DEEP_READ_MAX_CONTEXT_CHARS` | `40000` | PDF 上下文字符数，含章节标题；不等于 token 数 |
| `LANGUAGE` | `Chinese` | 摘要语言与输出文件名 |
| `NAME` | `luokairo` | 自动提交身份；未设置时使用仓库 owner |

保留现有 `EMAIL`、`CATEGORIES` 和并发设置。`FILTER_MAX_WORKERS`、`DETAIL_MAX_WORKERS`、
`IMPORTANCE_MAX_WORKERS`、`DEEP_READ_MAX_WORKERS` 默认均为 1；并发影响速度，不会直接减少 token。
兴趣配置存在时，抓取分类从兴趣方向配置合并得到，优先于 `CATEGORIES`。
`deepseek-v4*` 旧名称仍识别思考开关，推荐官方当前名称 `deepseek-flash`。
可选开启思考时，`DEEPSEEK_REASONING_EFFORT` 默认 `low`，仅允许 `low/high/max`。
原 GLM 配置仍兼容；切换至 DeepSeek 后无需设置 `GLM_REASONING_EFFORT`。

**如何节省 token**

- 摘要一次调用同时完成相关性判断、分类及各摘要字段，默认不额外调用筛选和评分模型。
- 保留全部预设子主题，每方向最多传入 5 个常用动态子主题；不重复发送计数和关键词。
- PDF 上下文去重并按章节分配额度，方法、实验获得较多额度，局限等章节不被前文挤掉。
- 摘要和精读的成功结果保存在 `data/ai_cache/`，随 `data` 分支恢复；相同论文、输入、模型、
  参数和实际提示上下文可复用。输入或配置变化会失效。失败与仅摘要降级的精读不缓存。
- Actions Step Summary 显示实际模型调用（含重试）、缓存命中、失败条目和输入／输出 token；
  API 未返回用量的调用标注“未知”。成功运行的用量另存 `data/run_metrics/`，不进入网页文件列表。

上下文上限和输出上限是单次预算，不是每日账单硬上限；重试也可能计费。不承诺固定节省比例。
应在相同“30 篇摘要＋3 篇精读”下比较；从精读关闭切换到开启会增加精读开销。

**失败与发布**

鉴权、模型和参数错误立即停止；限流、连接超时与服务器错误最多重试两次，不自动提高输出预算。
个别摘要解析失败会记录并排除，全部请求失败则终止；正常筛选后无相关论文允许空结果。
可选评分的个别失败使用本地评分，整个评分阶段失败则终止。精读全部生成失败时也终止，
PDF 提取失败的摘要分析明确标为 `abstract_fallback`，不能当作全文精读。
工作流和 `run.sh` 将日报与 taxonomy 写入暂存目录，通过全部阶段后才替换正式产物。
缓存文件不含 API Key。直接运行 `ai/enhance.py` 只保证增强阶段成功后再写入；完整发布保护请使用流水线。

**小规模验证**

Actions → **DeepSeek Pipeline Diagnostic** → Run workflow：选择 `data` 分支中存在的原始论文日期，
可指定论文 ID；未指定时选择本地优先级最高的候选。只处理一篇摘要和至多一篇精读，
不执行独立模型筛选／评分，不提交任何分支、不发布 Pages；结果和用量下载自 `deepseek-diagnostic` artifact。
若摘要模型判定该论文不相关，精读会正常跳过。首次部署后应检查摘要字段、精读来源标记及实际用量。
该入口固定使用 `deepseek-flash`，仍需先更新官方 API 的两个 Secrets。

本地无付费 API 调用的回归检查：

```bash
uv sync --locked
.venv/bin/python -m unittest discover -s tests -v
node tests/test_date_range.cjs
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
