# 本地召回离线验证：2026-09-30

> 历史验证记录：下表使用此前五方向、150/50/10 预算。当前规则已调整为七方向、180/60/15；保留历史数据原样，不将其视为新规则的验证结果。新规则通过本地回归和模拟模型测试验证，真实快照可用 `scripts/evaluate_local_recall.py` 重新离线评估（含 180 篇预算）。

数据：按 arXiv 公告日验证的 1187 篇去重原始论文，读取标题与完整摘要。没有模型请求，API token 用量为 0。

## 候选分配

| 候选上限 | 实际候选 | 世界模型 | 视频生成 | 音视频生成 | 统一理解生成 | 连续语言模型 | 仅探索 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 30 | 14 | 9 | 2 | 1 | 2 | 2 |
| 100 | 97 | 41 | 30 | 2 | 1 | 10 | 13 |
| 150 | 104 | 41 | 30 | 2 | 1 | 10 | 20 |

多方向论文按主要方向证据优先分池，仅占一个名额。方向名额不足时转移；方向候选和探索名额都不足时不强行凑满上限。探索论文未经模型确认，不能视作已相关。

## 已发现问题的代表论文

| 论文 ID / 标题 | 新 30 | 新 100 | 新 150 | 旧 30 (ba5b6fc) | 旧 100 | 旧 150 |
| --- | --- | --- | --- | --- | --- | --- |
| 2609.37004 · World2Motion: Turning Video World Models into 3D Human Motion Generators | 入选 | 入选 | 入选 | 入选 | 入选 | 入选 |
| 2609.37378 · Do-JEPA: From Masking to Intervention in Latent World Models | 未审阅 | 入选 | 入选 | 未审阅 | 入选 | 入选 |
| 2609.36413 · One from Infinity: Actualizing Futures from Pretrained World Models into Robot Actions | 入选 | 入选 | 入选 | 入选 | 入选 | 入选 |
| 2609.36438 · World4Scorer: Outcome-Grounded World Modeling for Autonomous Driving | 未审阅 | 入选 | 入选 | 入选 | 入选 | 入选 |
| 2609.37391 · Rethinking Soft Tokens for Parallel Decoding in Diffusion Language Models | 入选 | 入选 | 入选 | 未审阅 | 未审阅 | 未审阅 |
| 2609.37924 · Time-Anchored Diffusion Language Models: Latent-Space Caching for Fast Generation | 入选 | 入选 | 入选 | 未审阅 | 未审阅 | 未审阅 |
| 2609.37533 · E-MoE: Enhanced Mixture-of-Experts for Non-Factorized Diffusion Language Models | 未审阅 | 入选 | 入选 | 未审阅 | 未审阅 | 未审阅 |
| 2609.38066 · Alpha Diffusion Language Models: Factorization Alone Is Not the Problem | 未审阅 | 入选 | 入选 | 未审阅 | 未审阅 | 未审阅 |
| 2609.36452 · Reliable Parallel Decoding in Masked Diffusion Language Models | 未审阅 | 入选 | 入选 | 未审阅 | 未审阅 | 入选 |

旧版本比较仅运行 ba5b6fc 的本地匹配及排序函数，使用当前五方向配置，不执行旧主流程。该旧版本 ELF 子串匹配会误命中 self/shelf 等词；提高数量上限并不能修复匹配规则。

## 解释边界

这组论文是已知问题的针对性检查，没有人工标注全体 1,187 篇的相关性，因此结果不能称为真实召回率，也不能保证零遗漏。新规则仍可能遗漏无已知任务表达的新概念；稳定哈希探索只审阅其中一小部分。

详细摘要、简讯和精读的数量由模拟模型用例验证：最多 150 次轻筛、50 次详细摘要、辅助合计 10 次、3 篇精读；主方向精读优先保证 2 篇。真实方向判断、API 用量及费用需要后续由用户授权在线运行后测量。

完整逐篇匹配证据、方向分数与未审阅状态保存在离线输出目录的 candidates-30/100/150.json。
