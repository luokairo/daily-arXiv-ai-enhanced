"""Compare local candidate budgets against a saved raw snapshot, without any API calls."""
import argparse
import ast
import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'ai'), str(ROOT)]
import enhance
from local_recall import load_config, match_paper, select_candidates
from semantic_arxiv import load_directions, load_importance_config

REPRESENTATIVES = ['2609.37004', '2609.37378', '2609.36413', '2609.36438',
                   '2609.37391', '2609.37924', '2609.37533', '2609.38066', '2609.36452']
BUDGETS = [30, 100, 150, 180]


def markdown_table(headers):
    return ['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(['---'] + ['---:'] * (len(headers) - 1)) + ' |']


def old_selection(papers, directions, importance, cap, revision):
    # Execute only function definitions; never run historical main/model/network code.
    source = subprocess.check_output(['git', 'show', f'{revision}:ai/enhance.py'], cwd=ROOT, text=True)
    namespace = dict(enhance.__dict__)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = value
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    exec(compile(functions, '<historical-local-functions>', 'exec'), namespace)
    keywords = namespace['collect_filter_keywords'](directions, importance)
    candidates = [copy.deepcopy(p) for p in papers if namespace['local_filter_item'](p, directions, keywords)['state'] != 'drop']
    candidates.sort(key=lambda p: (-namespace['candidate_priority'](p, directions, importance), str(p['id'])))
    return {p['id'] for p in candidates[:cap]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    papers = list({p['id']: p for p in map(json.loads, args.data.read_text().splitlines())}.values())
    directions, importance, config = load_directions(), load_importance_config(), load_config()
    # Match full abstracts once. Allocations reuse the exact matcher output to reduce offline time.
    matches = {p['id']: match_paper(p, directions, importance, config) for p in papers}
    from unittest.mock import patch
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {}
    with patch('local_recall.match_paper', side_effect=lambda p, *unused: matches[p['id']]):
        for cap in BUDGETS:
            selected, audit, quotas = select_candidates(copy.deepcopy(papers), directions, importance, cap, args.data.stem, config)
            owners = Counter(p['_local_recall']['primary_direction_id'] or 'exploration_only' for p in selected)
            result = dict(budget=cap, selected=len(selected), directions=dict(owners), quotas=quotas,
                representative_status={key: audit[key]['selection_reason'] for key in REPRESENTATIVES if key in audit},
                model_calls=0, token_usage='0 (offline; no model requests)', papers=list(audit.values()))
            (args.output / f'candidates-{cap}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            runs[cap] = result
    labels = {d['id']: d['name'] for d in directions}; labels['exploration_only'] = '仅探索（无明确方向）'
    lines = [f'# 本地召回离线验证：{args.data.stem}', '',
        f'数据：按 arXiv 公告日验证的 {len(papers)} 篇去重原始论文，读取标题与完整摘要。没有模型请求，API token 用量为 0。', '',
        '## 候选分配', '', *markdown_table(['候选上限', '实际候选', *labels.values()])]
    for cap, row in runs.items():
        lines.append('| ' + ' | '.join(str(v) for v in [cap, row['selected'], *[row['directions'].get(key, 0) for key in labels]]) + ' |')
    lines += ['', '多方向论文按主要方向证据优先分池，仅占一个名额。方向名额不足时转移；方向候选和探索名额都不足时不强行凑满上限。探索论文未经模型确认，不能视作已相关。', '',
        '## 已发现问题的代表论文', '',
        *markdown_table(['论文 ID / 标题', *[f'新 {cap}' for cap in BUDGETS],
                         *[f'旧 {cap} (ba5b6fc)' for cap in BUDGETS]])]
    old = {cap: old_selection(papers, directions, importance, cap, 'ba5b6fc') for cap in BUDGETS}
    by_id = {p['id']: p for p in papers}
    for key in REPRESENTATIVES:
        if key not in by_id: continue
        status = ['入选' if runs[c]['representative_status'][key] != '未进入模型审阅' else '未审阅' for c in runs]
        status += ['入选' if key in old[c] else '未审阅' for c in runs]
        lines.append('| ' + ' | '.join([f"{key} · {by_id[key]['title']}", *status]) + ' |')
    lines += ['', f'旧版本比较仅运行 ba5b6fc 的本地匹配及排序函数，使用当前 {len(directions)} 个方向的配置，不执行旧主流程。该旧版本 ELF 子串匹配会误命中 self/shelf 等词；提高数量上限并不能修复匹配规则。', '',
        '## 解释边界', '',
        f'这组论文是已知问题的针对性检查，没有人工标注全体 {len(papers)} 篇的相关性，因此结果不能称为真实召回率，也不能保证零遗漏。新规则仍可能遗漏无已知任务表达的新概念；稳定哈希探索只审阅其中一小部分。', '',
        '详细摘要、简讯和精读的数量由模拟模型用例验证：默认最多 180 次轻筛、60 次详细摘要、辅助合计 15 次、3 篇精读；主方向精读优先保证 2 篇。这项离线评估不执行详细摘要或精读。真实方向判断、API 用量及费用需要后续在线运行后测量。', '',
        '完整逐篇匹配证据、方向分数与未审阅状态保存在离线输出目录的 candidates-30/100/150/180.json。']
    text = '\n'.join(lines) + '\n'
    (args.report or args.output / 'local-recall-validation.md').write_text(text)
    print('\n'.join(lines[:13]))


if __name__ == '__main__':
    main()
