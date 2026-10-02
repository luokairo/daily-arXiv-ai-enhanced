import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'ai'), str(ROOT)]
import enhance
import routing
import selection
import semantic_arxiv
from structure import RoutingStructure
from test_pipeline import IsolatedTest, structured


class AllocationTests(IsolatedTest):
    def setUp(self):
        super().setUp()
        self.directions = semantic_arxiv.load_directions()
        self.importance = semantic_arxiv.load_importance_config()

    def paper(self, key, primary='continuous_language_multimodal', score=80, matched=None):
        return dict(id=key, title=key, summary='Abstract evidence', authors=['Author'], categories=['cs.CL'],
            abs='https://arxiv.org/abs/' + key,
            _routing=dict(decision='relevant', primary_direction_id=primary,
                matched_direction_ids=matched or [primary], routing_status='ok',
                personal_relevance_score=score, research_value_score=score,
                brief='一句话论文贡献', reason='Concrete evidence in abstract'))

    def allocate(self, papers, limit=10):
        audit = {p['id']: dict(detail_status='pending') for p in papers}
        detail, briefs = selection.allocate_details(papers, self.directions, limit,
            lambda p: enhance.candidate_priority(p, self.directions, self.importance), audit)
        return detail, briefs, audit

    def test_configuration_covers_new_interests_and_audio_categories(self):
        self.assertEqual([d['tier'] for d in self.directions], ['primary'] * 3 + ['secondary'] * 2)
        self.assertEqual(len(self.directions), 5)
        self.assertTrue({'cs.CV', 'cs.CL', 'cs.AI', 'cs.LG', 'cs.MM', 'cs.RO', 'cs.SD', 'eess.AS'} <= set(semantic_arxiv.recall_categories(self.directions)))
        self.assertGreater(self.importance['direction_weights']['world_model'], self.importance['direction_weights']['continuous_language_multimodal'])

    def test_five_each_with_primary_overlap_processed_once(self):
        papers = [self.paper('c' + str(i), score=99) for i in range(8)]
        papers += [self.paper('u' + str(i), 'multimodal_understanding_generation', 20) for i in range(8)]
        papers += [self.paper('main', 'continuous_language_multimodal', matched=['continuous_language_multimodal', 'world_model'])]
        detail, briefs, audit = self.allocate(papers)
        self.assertEqual(sum(p['id'].startswith('c') for p in detail), 5)
        self.assertEqual(sum(p['id'].startswith('u') for p in detail), 5)
        self.assertEqual(len(detail), 11)
        self.assertEqual(len(briefs), 6)
        self.assertEqual(audit['main']['allocation_reason'], 'primary_full_coverage')
        self.assertTrue(all(audit[p['id']]['detail_status'] == 'quota_deferred' for p in briefs))

    def test_reallocation_zero_budget_and_undetermined_fallback(self):
        papers = [self.paper('c' + str(i)) for i in range(12)]
        papers += [self.paper('u', 'multimodal_understanding_generation')]
        detail, briefs, _ = self.allocate(papers)
        self.assertEqual(len(detail), 10)
        self.assertIn('u', {p['id'] for p in detail})
        self.assertEqual(len(briefs), 3)
        failed = self.paper('failed'); failed['_routing']['routing_status'] = 'failed'
        unknown = self.paper('unknown', primary=''); unknown['_routing']['matched_direction_ids'] = []
        detail, briefs, audit = self.allocate(papers + [failed, unknown], 0)
        self.assertEqual({p['id'] for p in detail}, {'failed', 'unknown'})
        self.assertEqual(len(briefs), 13)
        self.assertEqual(audit['unknown']['allocation_reason'], 'routing_fallback')

    def test_secondary_dual_match_counted_once_and_stable_order(self):
        papers = [self.paper(key, matched=['continuous_language_multimodal', 'multimodal_understanding_generation']) for key in ['3', '1', '2']]
        detail, briefs, _ = self.allocate(papers, 2)
        self.assertEqual({p['id'] for p in detail}, {'1', '2'})
        self.assertEqual([p['id'] for p in briefs], ['3'])

    def test_deep_reads_reserve_primary_and_exclude_briefs(self):
        papers = [self.paper('c', score=99), self.paper('u', 'multimodal_understanding_generation', 98),
                  self.paper('w', 'world_model', 20), self.paper('v', 'video_generation', 10)]
        for p in papers:
            p['AI'] = dict(primary_direction_id=p['_routing']['primary_direction_id'],
                importance_score=p['_routing']['personal_relevance_score'])
        brief = selection.make_brief(self.paper('brief', score=100), self.directions)
        brief['AI']['importance_score'] = 100
        selection.select_deep_reads(papers + [brief], 3, self.directions)
        self.assertEqual({p['id'] for p in papers if p['AI']['deep_read_selected']}, {'w', 'v', 'c'})
        self.assertFalse(brief['AI']['deep_read_selected'])
        selection.select_deep_reads(papers[:3], 3, self.directions)
        self.assertTrue(all(p['AI']['deep_read_selected'] for p in papers[:3]))

    def test_overlapping_priority_terms_only_one_concept_bonus(self):
        paper = self.paper('p', 'world_model')
        paper['summary'] = 'Video world model with world model planning.'
        config = dict(priority_subtopics=[dict(name='video', weight=1.8, keywords=['video world model']),
                                         dict(name='world', weight=1.4, keywords=['world model'])])
        score = enhance.heuristic_importance_score(paper, config)
        self.assertEqual(score, 45 + 12 + .8 * 30)

    def test_primary_and_brief_pipeline_failures_do_not_refill_quota(self):
        papers = [self.paper('c' + str(i)) for i in range(13)]
        source = self.root / '2026-10-01.jsonl'
        source.write_text(''.join(json.dumps(p) + '\n' for p in papers))
        taxonomy = self.root / 'taxonomy.json'; taxonomy.write_text('{}')
        args = SimpleNamespace(data=str(source), taxonomy=str(taxonomy), directions=None, max_workers=1)
        def route(inputs):
            return RoutingStructure(**self.paper(inputs['paper_id'])['_routing'])
        def detail(items, *args):
            self.assertEqual(len(items), 10)
            for p in items:
                p['_detail_status'] = 'irrelevant'
            items[0]['_detail_status'] = 'failed'
            return []
        with patch.object(enhance, 'parse_args', return_value=args), patch.object(routing, 'CachedChain', return_value=Mock(invoke=Mock(side_effect=route))), patch.object(enhance, 'process_all_items', side_effect=detail):
            enhance.main()
        output = list(map(json.loads, (self.root / '2026-10-01_AI_enhanced_Chinese.jsonl').read_text().splitlines()))
        self.assertEqual(len(output), 3)
        self.assertTrue(all(p['report_level'] == 'brief' and not p['AI']['deep_read_selected'] for p in output))
        audit = json.loads((self.root / 'run_metrics/2026-10-01-selection.json').read_text())
        self.assertEqual(audit['per_direction']['continuous_language_multimodal']['detail_requests'], 10)
        self.assertEqual(audit['per_direction']['continuous_language_multimodal']['briefs'], 3)
        self.assertTrue(list((self.root / 'run_metrics').glob('taxonomy-before-*.json')))

    def test_taxonomy_migration_preserves_backup_input_and_retires_old_topics(self):
        old = semantic_arxiv.empty_taxonomy([dict(id='multimodal_generation', name='old', canonical_subtopics=[])])
        old['directions']['multimodal_understanding_generation'] = dict(name='old', subtopics=[dict(id='vlm_mllm_general', name='VLM', is_canonical=True)])
        source = self.root / 'taxonomy.json'; source.write_text(json.dumps(old))
        loaded = semantic_arxiv.load_taxonomy(str(source), self.directions)
        self.assertEqual(json.loads(source.read_text()), old)
        self.assertNotIn('multimodal_generation', loaded['directions'])
        self.assertNotIn('vlm_mllm_general', {s['id'] for s in loaded['directions']['multimodal_understanding_generation']['subtopics']})

    def test_taxonomy_scope_change_removes_only_affected_dynamic_topics(self):
        taxonomy = semantic_arxiv.empty_taxonomy(self.directions)
        for key in ('world_model', 'multimodal_understanding_generation'):
            taxonomy['directions'][key]['subtopics'].append(dict(id='custom', name='dynamic'))
        source = self.root / 'taxonomy.json'; source.write_text(json.dumps(taxonomy))
        revised = copy.deepcopy(self.directions)
        next(d for d in revised if d['id'] == 'multimodal_understanding_generation')['description'] += ' tightened'
        loaded = semantic_arxiv.load_taxonomy(str(source), revised)
        self.assertIn('custom', [s['id'] for s in loaded['directions']['world_model']['subtopics']])
        self.assertNotIn('custom', [s['id'] for s in loaded['directions']['multimodal_understanding_generation']['subtopics']])

    def test_markdown_briefs_do_not_have_empty_analysis_sections(self):
        spec = importlib.util.spec_from_file_location('convert', ROOT / 'to_md/convert.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        brief = selection.make_brief(self.paper('brief'), self.directions)
        rendered = module.render_paper('', brief, 1, {d['id']: d for d in self.directions})
        self.assertIn('简讯，未做详细分析', rendered)
        self.assertNotIn('Motivation', rendered)
        self.assertNotIn('Method', rendered)
        self.assertEqual(module.report_group(brief), 'secondary_brief')
        self.assertEqual(module.report_group({'id': 'historical'}), 'primary_detail')


if __name__ == '__main__':
    unittest.main()
