"""Preference pools, bounded secondary summaries and primary deep-read reservations."""
from copy import deepcopy

from semantic_arxiv import direction_display, normalize_matched_directions


def direction_ids(paper):
    fields = paper.get('AI') or paper.get('_routing') or {}
    return set(fields.get('matched_direction_ids', [])) | {fields.get('primary_direction_id', '')}


def interest_tier(paper, directions):
    matched = direction_ids(paper)
    tiers = [d.get('tier', 'primary') for d in directions if d['id'] in matched]
    return 'primary' if 'primary' in tiers else ('secondary' if tiers else 'fallback')


def rank_key(paper):
    ai = paper.get('AI') or {}
    return (-float(ai.get('importance_rank_score', ai.get('importance_score', 0)) or 0),
            -float(ai.get('local_priority_score', 0) or 0), str(paper.get('id', '')))


def secondary_reservations(directions, limit):
    secondary = [d for d in directions if d.get('tier') == 'secondary']
    total = sum(int(d.get('detail_quota', 0)) for d in secondary)
    if not total:
        return {d['id']: 0 for d in secondary}
    raw = [(d['id'], limit * int(d.get('detail_quota', 0)) / total) for d in secondary]
    quotas = {key: int(value) for key, value in raw}
    # Largest remainders; config order breaks equal fractions deterministically.
    for key, _ in sorted(raw, key=lambda pair: -(pair[1] - int(pair[1])))[:limit - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def allocate_details(papers, directions, limit, priority, audit):
    secondary, detailed = [], []
    for paper in papers:
        route = paper['_routing']
        tier = interest_tier(paper, directions)
        paper['interest_tier'] = tier
        if route.get('routing_status') != 'ok' or not route.get('primary_direction_id') or tier == 'fallback':
            paper['interest_tier'] = 'fallback'
            detailed.append(paper)
            audit[paper['id']]['allocation_reason'] = 'routing_fallback'
        elif tier == 'primary':
            detailed.append(paper)
            audit[paper['id']]['allocation_reason'] = 'primary_full_coverage'
        else:
            secondary.append(paper)
    def key(p):
        route = p['_routing']
        local = priority(p)
        model = route['personal_relevance_score'] * .55 + route['research_value_score'] * .45
        return (-(.70 * model + .30 * max(0, min(100, local))), -local, str(p['id']))
    secondary.sort(key=key)
    selected = set()
    for direction, quota in secondary_reservations(directions, limit).items():
        selected.update(p['id'] for p in [p for p in secondary
                        if p['_routing']['primary_direction_id'] == direction][:quota])
    selected.update(p['id'] for p in [p for p in secondary if p['id'] not in selected][:limit - len(selected)])
    briefs = []
    for paper in secondary:
        if paper['id'] in selected:
            detailed.append(paper)
            audit[paper['id']]['allocation_reason'] = 'secondary_selected'
        else:
            audit[paper['id']].update(detail_status='quota_deferred', allocation_reason='secondary_quota')
            briefs.append(paper)
    return detailed, briefs


def make_brief(paper, directions):
    item = deepcopy(paper)
    route = item['_routing']
    primary = route['primary_direction_id']
    item.update(report_level='brief', interest_tier='secondary', relevance_status=route['decision'])
    item['primary_direction'] = direction_display(primary, directions)
    item['matched_directions'] = normalize_matched_directions(route['matched_direction_ids'], directions)
    item['AI'] = dict(primary_direction_id=primary, matched_direction_ids=route['matched_direction_ids'],
                      tldr=route['brief'], classification_reason=route['reason'],
                      subtopic_id='brief_queue', subtopic_name='辅助方向简讯',
                      deep_read_selected=False, deep_read_rank=None)
    return item


def select_deep_reads(papers, top_k, directions=None, primary_min=2):
    eligible = sorted([p for p in papers if p.get('report_level', 'detail') == 'detail'], key=rank_key)
    primary = [p for p in eligible if interest_tier(p, directions or []) == 'primary']
    selected = primary[:min(primary_min, top_k)]
    ids = {p['id'] for p in selected}
    selected += [p for p in eligible if p['id'] not in ids][:top_k - len(selected)]
    ranks = {p['id']: rank for rank, p in enumerate(selected, 1)}
    for p in papers:
        p['AI']['deep_read_selected'] = p['id'] in ranks
        p['AI']['deep_read_rank'] = ranks.get(p['id'])
    return papers
