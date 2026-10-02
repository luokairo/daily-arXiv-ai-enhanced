"""Free, auditable title/full-abstract recall and bounded candidate allocation."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config/local_recall.yaml'


def normalize(text):
    text = unicodedata.normalize('NFKC', str(text or '')).casefold()
    text = re.sub(r'[-‐‑‒–—−/\\_]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def phrase_spans(text, phrase):
    # Full boundaries avoid ELF inside self/shelf; aliases specify word forms.
    return list(re.finditer(r'(?<!\w)' + re.escape(normalize(phrase)) + r'(?!\w)', text))


def evidence(text, group):
    for sentence in re.split(r'[.!?;\n]+', unicodedata.normalize('NFKC', str(text or ''))):
        sentence = normalize(sentence)
        for alias in group['aliases']:
            for hit in phrase_spans(sentence, alias):
                contexts = group.get('context', [])
                if not contexts:
                    return dict(alias=alias, sentence=sentence[:600])
                for context in contexts:
                    for other in phrase_spans(sentence, context):
                        gap = sentence[min(hit.end(), other.end()):max(hit.start(), other.start())]
                        if len(gap.split()) <= 40:
                            return dict(alias=alias, context=context, sentence=sentence[:600])
    return None


def load_config(path=None):
    config = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding='utf-8'))
    if not isinstance(config, dict) or not config.get('directions'):
        raise ValueError('local_recall.yaml must define directions')
    quotas = config.get('candidate_quotas', {})
    if any(not isinstance(v, int) or v < 0 for v in quotas.values()) or not sum(quotas.values()):
        raise ValueError('Candidate quota weights must be nonnegative integers with positive total')
    return config


def match_paper(paper, directions, importance, config):
    scores, matches = {}, {}
    video_world = False
    for direction in directions:
        hits = []
        for group in config['directions'].get(direction['id'], []):
            hit = evidence(paper.get('title', ''), group)
            source = 'title'
            if not hit:
                hit = evidence(paper.get('summary', ''), group)
                source = 'abstract'
            if hit:
                conditional = bool(group.get('context'))
                score = (5 if source == 'title' else 3) if conditional else (8 if source == 'title' else 5)
                if group.get('bonus_only'):
                    score = 0
                hits.append(dict(concept=group['id'], source=source, score=score, **hit))
                if group.get('video_world'):
                    video_world = True
        hits.sort(key=lambda h: (-h['score'], h['concept']))
        if hits:
            weight = importance.get('direction_weights', {}).get(direction['id'], 1)
            scores[direction['id']] = round(sum(h['score'] for h in hits[:3]) * weight, 4)
            matches[direction['id']] = hits
    if video_world and 'world_model' in scores:
        scores['world_model'] += 4
    text = normalize(paper.get('title', '') + ' ' + paper.get('summary', ''))
    weak = [term for term in config.get('weak_signals', []) if phrase_spans(text, term)]
    negatives = [term for term in config.get('negative_domains', []) if phrase_spans(text, term)]
    strong = any(not h.get('context') for hits in matches.values() for h in hits)
    # Domain penalties affect weak evidence only, never an explicit target task.
    if negatives and not strong:
        scores = {key: round(value * .5, 4) for key, value in scores.items()}
    primary_ids = {d['id'] for d in directions if d.get('tier', 'primary') == 'primary'}
    choices = [key for key in scores if key in primary_ids] or list(scores)
    owner = min(choices, key=lambda key: (-scores[key], key)) if choices else ''
    weak_score = max(scores.values(), default=0) or min(len(weak), 3)
    if negatives and not strong and not scores:
        weak_score *= .5
    return dict(direction_scores=scores, matched_direction_ids=list(scores), primary_direction_id=owner,
                score=max(scores.values(), default=0), evidence=matches, weak_signals=weak,
                weak_score=weak_score, negative_domains=negatives, strong_evidence=strong,
                video_world_evidence=video_world)


def scaled_quotas(weights, limit):
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError('Candidate limit must be a positive integer')
    total = sum(weights.values())
    raw = {key: limit * value / total for key, value in weights.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    order = sorted(raw, key=lambda key: -(raw[key] - quotas[key]))
    for key in order[:limit - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def select_candidates(papers, directions, importance, limit=150, date='', config=None):
    config = config or load_config()
    unique = {str(p['id']): p for p in papers}
    audit, selected = {}, []
    signature = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    for key, p in sorted(unique.items()):
        result = match_paper(p, directions, importance, config)
        p['_local_recall'] = result
        audit[key] = dict(id=key, title=p.get('title', ''), local_recall=result,
                          local_rules_hash=signature, routing_status='not_reviewed',
                          detail_status='not_requested', selection_reason='未进入模型审阅')
    def rank(p):
        return (-p['_local_recall']['score'], str(p['id']))
    selected_ids = set()
    def take(pool, count, reason):
        for p in pool:
            key = str(p['id'])
            if count <= 0 or len(selected) >= limit:
                break
            if key in selected_ids:
                continue
            selected.append(p); selected_ids.add(key); count -= 1
            audit[key].update(routing_status='pending', selection_reason=reason)
    quotas = scaled_quotas(config['candidate_quotas'], limit)
    valid = sorted([p for p in unique.values() if p['_local_recall']['primary_direction_id']], key=rank)
    for direction, quota in quotas.items():
        if direction != 'exploration':
            take([p for p in valid if p['_local_recall']['primary_direction_id'] == direction], quota, 'direction_quota:' + direction)
    # Unused direction slots are transferred before exploration.
    take(valid, limit - quotas.get('exploration', 0) - len(selected), 'direction_transfer')
    exploration = quotas.get('exploration', 0)
    weak = sorted([p for p in unique.values() if p['_local_recall']['weak_score'] > 0],
                  key=lambda p: (-p['_local_recall']['weak_score'], str(p['id'])))
    take(weak, (exploration + 1) // 2, 'exploration_weak')
    # Hash sampling can discover papers with no known phrase: it is explicitly exploratory.
    pool = sorted(unique.values(), key=lambda p: (
        hashlib.sha256((date + ':' + str(p['id'])).encode()).hexdigest(), str(p['id'])))
    take(pool, exploration // 2, 'exploration_hash')
    take(valid, limit - len(selected), 'unused_exploration_transfer')
    return sorted(selected, key=lambda p: str(p['id'])), audit, quotas
