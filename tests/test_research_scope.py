"""Regression coverage for representation interests, pool boundaries and migration."""
import contextlib
import copy
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
NODE_BINARY = os.environ.get('NODE_BINARY') or shutil.which('node')
sys.path[:0] = [str(ROOT / 'ai'), str(ROOT)]
import enhance
import local_recall
import routing
import runtime
import selection
import semantic_arxiv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from scripts import evaluate_local_recall
from structure import RoutingStructure
from test_pipeline import IsolatedTest


class ResearchScopeTests(IsolatedTest):
    def setUp(self):
        super().setUp()
        self.directions = semantic_arxiv.load_directions()
        self.importance = semantic_arxiv.load_importance_config()
        self.config = local_recall.load_config()

    def match(self, title='', summary=''):
        return local_recall.match_paper(dict(title=title, summary=summary),
            self.directions, self.importance, self.config)

    def test_representation_research_without_generation_experiments(self):
        for title, summary in [
            ('Representation Autoencoders for Image Reconstruction', ''),
            ('VideoRAE', 'VideoRAE learns semantic video latents.'),
            ('RAEs for visual representations', ''),
            ('DINOv3', 'DINOv3 improves self-supervised visual representations.'),
            ('DINO for image representation learning', ''),
            ('V-JEPA 2 for video pretraining', ''),
            ('I-JEPA visual representations', ''),
            ('MAE for masked image pretraining', ''),
            ('VideoMAE for self-supervised video learning', ''),
            ('Semantic latent spaces for images', ''),
            ('Visual tokenizer with representation alignment', ''),
        ]:
            with self.subTest(title=title):
                result = self.match(title, summary)
                self.assertEqual(result['primary_direction_id'], 'world_model')
                self.assertNotIn('continuous_language_multimodal', result['matched_direction_ids'])

    def test_foundations_applications_and_audio_have_distinct_owners(self):
        cases = {
            'Action-conditioned video prediction': 'world_model',
            'Multimodal world models': 'world_model',
            'Robot planning with an embodied world model': 'world_model_applications',
            'Learned neural simulator for physical dynamics': 'world_model_applications',
            '3D world models for environment simulation': 'world_model_applications',
            'Video-to-speech from silent faces': 'video_audio_generation',
            'Foley generation for video soundtracks': 'video_audio_generation',
            'Joint audio-video generation': 'audio_video_generation',
            'Speech-driven talking head generation': 'audio_video_generation',
            'Unified understanding and generation': 'multimodal_understanding_generation',
            'Diffusion language models': 'continuous_language_multimodal',
        }
        for title, owner in cases.items():
            with self.subTest(title=title):
                result = self.match(title)
                self.assertEqual(result['primary_direction_id'], owner)
                expected_tier = next(d['tier'] for d in self.directions if d['id'] == owner)
                paper = dict(_routing=dict(primary_direction_id=owner, matched_direction_ids=[owner]))
                self.assertEqual(selection.interest_tier(paper, self.directions), expected_tier)

    def test_joint_audio_alias_does_not_consume_standalone_video_quota(self):
        for title in ('Joint audio-video generation', 'Joint audio/video synthesis',
                      'Joint audio and video generation'):
            with self.subTest(title=title):
                result = self.match(title)
                self.assertEqual(result['primary_direction_id'], 'audio_video_generation')
                self.assertNotIn('video_generation', result['matched_direction_ids'])
        mixed = self.match('Joint audio-video generation and independent video generation')
        self.assertIn('video_generation', mixed['matched_direction_ids'])

    def test_acronym_boundaries_and_unrelated_tasks(self):
        for title in ['RAE financial forecasting', 'MAE metric for image regression',
                      'Dinosaur image classification', 'JerAE image reconstruction',
                      'Text-to-speech synthesis', 'Text-to-music generation',
                      'Audio recognition and speech transcription',
                      'Ordinary VQA using visual reasoning', 'Robotic policy optimization']:
            with self.subTest(title=title):
                self.assertFalse(self.match(title)['matched_direction_ids'])

    def test_prompts_require_contributions_and_keep_baseline_usage_reviewable(self):
        paper = dict(id='baseline', title='DINO for image classification',
            categories=['cs.CV'], summary='We use an existing DINO encoder as a frozen baseline for recognition.')
        # Local recall is deliberately broader; semantic review must receive this evidence.
        self.assertIn('world_model', self.match(paper['title'], paper['summary'])['matched_direction_ids'])
        result = RoutingStructure(decision='irrelevant', reason='Only uses an existing encoder as a baseline.',
            personal_relevance_score=0, research_value_score=20)
        chain = Mock(invoke=Mock(return_value=result))
        with patch.object(routing, 'CachedChain', return_value=chain) as factory:
            self.assertEqual(routing.route_all_items([paper], 'model', 1,
                self.directions, self.importance, {}), [])
        prompt = factory.call_args.args[0]
        text = prompt.format(title=paper['title'], categories='cs.CV', abstract=paper['summary'])
        system = (ROOT / 'ai/system.txt').read_text()
        for scope in (text, system):
            for term in ('RAE/VideoRAE', 'No generation or world-model experiment',
                         'tool or baseline', 'world_model_applications', 'video_audio_generation',
                         'primary audio_video_generation', 'shared model', 'Background mentions'):
                with self.subTest(term=term):
                    self.assertIn(term.casefold(), scope.casefold())
        self.assertEqual(chain.invoke.call_args.args[0]['abstract'], paper['summary'])
        for direction in self.directions:
            self.assertIn(direction['id'], text)

    def test_secondary_limit_is_configurable_and_clamped_by_total(self):
        papers = [dict(id=str(i), _routing=dict(routing_status='ok',
            primary_direction_id='video_audio_generation', matched_direction_ids=['video_audio_generation'],
            personal_relevance_score=80, research_value_score=80)) for i in range(30)]
        for limit, total, expected in [(15, 60, 15), (20, 60, 20), (20, 7, 7), (0, 60, 0)]:
            audit = {p['id']: {} for p in papers}
            detail, briefs = selection.allocate_details(copy.deepcopy(papers), self.directions,
                limit, lambda p: 80, audit, total_limit=total)
            self.assertEqual(len(detail), expected)
            self.assertEqual(len(detail) + len(briefs), len(papers))
        self.assertEqual(selection.secondary_reservations(self.directions, 15),
            dict(world_model_applications=5, video_audio_generation=5, continuous_language_multimodal=5))
        self.assertEqual(selection.secondary_reservations(self.directions, 2),
            dict(world_model_applications=1, video_audio_generation=1, continuous_language_multimodal=0))

    def test_moved_topics_are_seeded_and_original_taxonomy_is_preserved(self):
        old_directions = copy.deepcopy(self.directions[:4] + [self.directions[-1]])
        old_world = old_directions[0]
        applications = next(d for d in self.directions if d['id'] == 'world_model_applications')
        audio_only = next(d for d in self.directions if d['id'] == 'video_audio_generation')
        old_world['name'] = '世界模型'
        old_world['description'] = 'Old broader world-model scope'
        old_world['canonical_subtopics'] = [old_world['canonical_subtopics'][0], *applications['canonical_subtopics']]
        old_audio = next(d for d in old_directions if d['id'] == 'audio_video_generation')
        old_audio['canonical_subtopics'] += audio_only['canonical_subtopics']
        old = semantic_arxiv.empty_taxonomy(old_directions)
        old['directions']['world_model']['subtopics'].append(dict(id='dynamic_old', name='old dynamic'))
        source = self.root / 'taxonomy.json'
        source.write_text(json.dumps(old))
        migrated = semantic_arxiv.load_taxonomy(str(source), self.directions)
        self.assertEqual(json.loads(source.read_text()), old)
        self.assertEqual(set(migrated['directions']), {d['id'] for d in self.directions})
        topics = lambda key: {s['id'] for s in migrated['directions'][key]['subtopics']}
        self.assertNotIn('dynamic_old', topics('world_model'))
        self.assertNotIn('embodied_world_model', topics('world_model'))
        self.assertIn('embodied_world_model', topics('world_model_applications'))
        self.assertNotIn('video_soundtrack', topics('audio_video_generation'))
        self.assertIn('video_soundtrack', topics('video_audio_generation'))
        self.assertIn('representation_autoencoder', topics('world_model'))

    def test_changed_scope_invalidates_cache_without_clearing_old_entries(self):
        model = RunnableLambda(Mock(return_value=AIMessage(content='cached response')))
        responder = model.func
        with patch.object(runtime, 'ChatOpenAI', return_value=model):
            old = runtime.CachedChain(ChatPromptTemplate.from_template('Old scope: {content}'), 'model', 'detail')
            revised = runtime.CachedChain(ChatPromptTemplate.from_template(
                semantic_arxiv.compact_directions_for_prompt(self.directions) + '\n{content}'), 'model', 'detail')
        old.invoke(dict(paper_id='same', content='Image RAE'))
        revised.invoke(dict(paper_id='same', content='Image RAE'))
        revised.invoke(dict(paper_id='same', content='Image RAE'))
        self.assertEqual(responder.call_count, 2)
        self.assertEqual(len(list((self.root / 'cache/detail').glob('*.json'))), 2)

    def test_offline_evaluation_has_dynamic_columns_and_180_budget(self):
        source = self.root / 'sample.jsonl'
        source.write_text(json.dumps(dict(id='rae', title='Representation Autoencoders', summary='')) + '\n')
        output = self.root / 'evaluation'
        with patch.object(sys, 'argv', ['evaluate_local_recall', '--data', str(source), '--output', str(output)]), \
             patch.object(evaluate_local_recall, 'old_selection', return_value=set()), \
             contextlib.redirect_stdout(io.StringIO()):
            evaluate_local_recall.main()
        report = (output / 'local-recall-validation.md').read_text()
        for direction in self.directions:
            self.assertIn(direction['name'], report)
        tables = [line for line in report.splitlines() if line.startswith('|')]
        for line in tables[:6]:
            self.assertEqual(line.count('|'), 11)
        audit = json.loads((output / 'candidates-180.json').read_text())
        self.assertEqual(audit['model_calls'], 0)
        self.assertEqual(audit['quotas'], self.config['candidate_quotas'])

    def test_web_preserves_historical_tiers_and_reads_explicit_new_tiers(self):
        node = NODE_BINARY
        if not node:
            self.skipTest('Node is required to verify the existing browser parser')
        source = (ROOT / 'js/app.js').read_text()
        start = source.index('function parseJsonlData(')
        end = source.index('// 获取所有类别并按偏好排序', start)
        rows = [
            dict(id='old-unified', categories=['cs.CV'], AI=dict(primary_direction_id='multimodal_understanding_generation')),
            dict(id='new-unified', categories=['cs.CV'], interest_tier='primary', AI=dict(primary_direction_id='multimodal_understanding_generation')),
            dict(id='new-audio', categories=['cs.CV'], interest_tier='secondary', AI=dict(primary_direction_id='video_audio_generation')),
            dict(id='historical', categories=['cs.CV']),
        ]
        program = source[start:end] + "\nconst fs = require('fs'); const result = parseJsonlData(fs.readFileSync(0, 'utf8'), 'date'); process.stdout.write(JSON.stringify(Object.values(result).flat().map(p => [p.id, p.interestTier, p.reportLevel])));"
        response = subprocess.run([node, '-e', program], input='\n'.join(map(json.dumps, rows)),
            text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(response.stdout), [
            ['old-unified', 'secondary', 'detail'], ['new-unified', 'primary', 'detail'],
            ['new-audio', 'secondary', 'detail'], ['historical', 'primary', 'detail']])
