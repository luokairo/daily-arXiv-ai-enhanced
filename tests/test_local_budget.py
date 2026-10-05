import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'ai'), str(ROOT)]
import enhance
import local_recall as recall
import progress
import routing
import runtime
import selection
import semantic_arxiv
from structure import RoutingStructure
from test_pipeline import IsolatedTest, structured


class RecallTests(unittest.TestCase):
    def setUp(self):
        self.directions = semantic_arxiv.load_directions()
        self.importance = semantic_arxiv.load_importance_config()
        self.config = recall.load_config()

    def match(self, title='', summary=''):
        return recall.match_paper(dict(title=title, summary=summary), self.directions, self.importance, self.config)

    def test_seven_directions_inflections_and_full_abstract(self):
        positives = {
            'world_model': ['Video World Models', 'Video World Modelling', 'Action-conditioned video prediction', 'Representation Autoencoders', 'VideoRAE visual latents'],
            'video_generation': ['Text-to-Video', 'I2V', 'Generating Videos', 'Video-to-video editing'],
            'audio_video_generation': ['Joint audio/video synthesis', 'Audio-driven video'],
            'video_audio_generation': ['Video-to-Speech', 'Foley Generation'],
            'world_model_applications': ['World Models', 'Learned simulator for physics'],
            'multimodal_understanding_generation': ['Unified understanding and generation', 'Any-to-any multimodal'],
            'continuous_language_multimodal': ['Masked diffusion language models', 'Embedding-space language models', 'ELF: embedded language flow'],
        }
        for direction, terms in positives.items():
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(direction, self.match(summary=term)['matched_direction_ids'])
        self.assertTrue(self.match('Video World Models')['video_world_evidence'])
        self.assertGreater(self.match('Video generation')['score'], self.match(summary='Video generation')['score'])

    def test_context_boundaries_and_generic_negatives(self):
        for text in ['Self supervised off-the-shelf diffusion', 'CoLa improves clinical prediction',
                     'Diffusion and flow matching for image generation', 'A tokenizer benchmark for ordinary VLM reasoning',
                     'Text-to-speech music synthesis', 'Lip synchronization recognition',
                     'Continuous tokens for image reconstruction']:
            with self.subTest(text=text):
                self.assertFalse(self.match(text)['matched_direction_ids'])
        for text in ['ELF language model', 'CoLa text generation', 'Soft tokens for diffusion language models',
                     'Lip synchronization for talking head generation', 'Video diffusion distillation']:
            self.assertTrue(self.match(summary=text)['matched_direction_ids'])
        group = dict(aliases=['tokenizer'], context=['video generation'])
        self.assertIsNone(recall.evidence('Tokenizer. Video generation is unrelated.', group))
        self.assertIsNone(recall.evidence('Tokenizer ' + 'word ' * 41 + 'video generation.', group))
        self.assertIsNotNone(recall.evidence('Tokenizer for faster video generation.', group))

    def test_aliases_dedup_and_domain_penalties(self):
        one = self.match('Video world model')
        repeated = self.match('Video world model; video world models; video world modeling; video world modelling')
        self.assertEqual(one['score'], repeated['score'])
        self.assertEqual(self.match('Clinical video generation')['score'], self.match('Video generation')['score'])
        self.assertLess(self.match('Medical learned simulator for physics')['score'], self.match('Learned simulator for physics')['score'])

    def test_quotas_dedup_stability_transfer_exploration(self):
        papers = [dict(id=f'{i:03}', title='Video generation', summary='') for i in range(190)]
        papers += [dict(id='both', title='Video world models for video generation', summary='')]
        selected, audit, quotas = recall.select_candidates(papers + papers[:2], self.directions, self.importance, 180, '2026-09-30')
        other, other_audit, _ = recall.select_candidates(list(reversed(papers)), self.directions, self.importance, 180, '2026-09-30')
        self.assertEqual(len(selected), 180)
        self.assertEqual([p['id'] for p in selected], [p['id'] for p in other])
        self.assertEqual(len(audit), 191)
        self.assertEqual(quotas['world_model'], 60)
        self.assertEqual(audit['both']['local_recall']['primary_direction_id'], 'world_model')
        reasons = [a['selection_reason'] for a in audit.values()]
        self.assertEqual(sum(r == 'exploration_hash' for r in reasons), 10)
        self.assertEqual(sum(r == 'exploration_weak' for r in reasons), 10)
        self.assertTrue(any(a['routing_status'] == 'not_reviewed' for a in audit.values()))
        self.assertEqual(sum(recall.scaled_quotas(self.config['candidate_quotas'], 100).values()), 100)
        self.assertEqual(recall.scaled_quotas(self.config['candidate_quotas'], 100),
            dict(world_model=33, video_generation=17, multimodal_understanding_generation=14,
                 audio_video_generation=11, world_model_applications=6, video_audio_generation=5,
                 continuous_language_multimodal=3, exploration=11))
        for limit in [0, -1]:
            with self.assertRaises(ValueError):
                recall.select_candidates(papers, self.directions, self.importance, limit)


class BudgetTests(IsolatedTest):
    def setUp(self):
        super().setUp()
        progress.STOP.clear()
        self.directions = semantic_arxiv.load_directions()
        self.importance = semantic_arxiv.load_importance_config()

    def paper(self, number, primary='world_model', status='ok'):
        return dict(id=f'{number:04}', title={
                'world_model': 'VideoRAE learns video representations',
                'video_generation': 'Video diffusion generation',
                'multimodal_understanding_generation': 'Unified understanding and generation',
                'audio_video_generation': 'Joint audio video generation',
                'world_model_applications': 'Learned simulator for physics',
                'video_audio_generation': 'Video-to-Speech',
                'continuous_language_multimodal': 'Diffusion language models',
            }[primary], summary='We introduce a new method for this task.',
            categories=['cs.CV'], authors=['Author'], abs='https://arxiv.org/abs/example',
            _local_recall=dict(score=number % 40 + 1),
            _routing=dict(decision='relevant', routing_status=status,
                primary_direction_id=primary if status == 'ok' else '',
                matched_direction_ids=[primary] if status == 'ok' else [],
                personal_relevance_score=80, research_value_score=80, brief='贡献简讯', reason='Evidence'))

    def allocate(self, papers, limit=15):
        audit = {p['id']: dict(detail_status='pending') for p in papers}
        detail, briefs = selection.allocate_details(papers, self.directions, limit,
            lambda p: 80, audit, total_limit=60)
        return detail, briefs, audit

    def test_60_15_reservations_primary_brief_failed_queue(self):
        papers = [self.paper(i) for i in range(80)]
        papers += [self.paper(i, 'continuous_language_multimodal') for i in range(80, 100)]
        papers += [self.paper(i, 'video_audio_generation') for i in range(100, 120)]
        papers += [self.paper(i, 'world_model_applications') for i in range(150, 170)]
        papers += [self.paper(i, status='failed') for i in range(120, 150)]
        detail, briefs, audit = self.allocate(papers)
        self.assertEqual(len(detail), 60)
        self.assertEqual(sum(p['interest_tier'] == 'secondary' for p in detail), 15)
        self.assertEqual(sum(p['_routing']['primary_direction_id'] == 'continuous_language_multimodal' for p in detail), 5)
        self.assertTrue(any(p['interest_tier'] == 'primary' for p in briefs))
        self.assertTrue(any(a.get('detail_status') == 'awaiting_review' for a in audit.values()))
        self.assertTrue(all(p['_routing']['routing_status'] == 'ok' for p in briefs))
        brief = selection.make_brief(next(p for p in briefs if p['interest_tier'] == 'primary'), self.directions)
        self.assertEqual(brief['interest_tier'], 'primary')
        # No secondary demand: all sixty slots can transfer to the primary pool.
        self.assertEqual(len(self.allocate([self.paper(i) for i in range(80)])[0]), 60)
        self.assertEqual(len(self.allocate([self.paper(i, 'continuous_language_multimodal') for i in range(80)], 99)[0]), 60)

    def test_mock_end_to_end_180_60_15_3_no_paid_requests(self):
        direction_ids = [d['id'] for d in self.directions]
        papers = [self.paper(i, direction_ids[i % 7]) for i in range(280)]
        source = self.root / '2026-09-30.jsonl'
        source.write_text(''.join(json.dumps(p) + '\n' for p in papers))
        args = SimpleNamespace(data=str(source), taxonomy=str(self.root / 'taxonomy.json'), directions=None, max_workers=1)
        def route(inputs):
            number = int(inputs['paper_id'])
            if number % 17 == 0:
                raise ValueError('Routing parsing failure')
            primary = direction_ids[number % 7]
            return RoutingStructure(decision='relevant', primary_direction_id=primary,
                matched_direction_ids=[primary], personal_relevance_score=80, research_value_score=80,
                brief='研究贡献', reason='Abstract evidence')
        def detail(items, *args, **kwargs):
            self.assertEqual(len(items), 60)
            self.assertEqual(sum(p['interest_tier'] == 'secondary' for p in items), 15)
            for p in items:
                p['_detail_status'] = 'relevant'
                p['report_level'] = 'detail'
                p['AI'] = structured().model_dump()
                p['AI']['primary_direction_id'] = p['_routing'].get('primary_direction_id') or 'world_model'
                p['AI']['matched_direction_ids'] = [p['AI']['primary_direction_id']]
                kwargs['on_complete'](p)
            return items
        chain = Mock(invoke=Mock(side_effect=route))
        with patch.object(enhance, 'parse_args', return_value=args), patch.object(routing, 'CachedChain', return_value=chain), patch.object(enhance, 'process_all_items', side_effect=detail):
            enhance.main()
        self.assertEqual(chain.invoke.call_count, 180)
        output = [json.loads(line) for line in (self.root / '2026-09-30_AI_enhanced_Chinese.jsonl').read_text().splitlines()]
        self.assertEqual(sum(p['report_level'] == 'detail' for p in output), 60)
        deep = [p for p in output if p['AI']['deep_read_selected']]
        self.assertEqual(len(deep), 3)
        self.assertGreaterEqual(sum(selection.interest_tier(p, self.directions) == 'primary' for p in deep), 2)
        self.assertTrue(all(p['report_level'] == 'detail' for p in deep))
        audit = json.loads((self.root / 'run_metrics/2026-09-30-selection.json').read_text())
        self.assertEqual(audit['unreviewed'], 100)
        self.assertEqual(audit['limits'], dict(candidates=180, details=60, secondary_details=15))
        self.assertEqual(sum(p.get('detail_requested', False) for p in audit['papers']), 60)
        self.assertTrue(any(p['report_level'] == 'brief' and p['interest_tier'] == 'primary' for p in output))
        import deep_read
        read_args = SimpleNamespace(data=str(self.root / '2026-09-30_AI_enhanced_Chinese.jsonl'),
            output=str(self.root / 'deep_reads/report.md'), output_json=str(self.root / 'deep_reads/report.json'), directions=None)
        def read(chain, paper, rank, tmp):
            chain.invoke(dict(paper_id=paper['id']))
            return dict(id=paper['id'], rank=rank, status='full_text', report='Mock report')
        deep_chain = Mock()
        with patch.object(deep_read, 'parse_args', return_value=read_args), patch.object(deep_read, 'CachedChain', return_value=deep_chain), patch.object(deep_read, 'run_deep_read', side_effect=read):
            deep_read.main()
        self.assertEqual(deep_chain.invoke.call_count, 3)

    def test_invalid_caps_fail_before_any_chain(self):
        source = self.root / 'sample.jsonl'; source.write_text(json.dumps(self.paper(1)) + '\n')
        args = SimpleNamespace(data=str(source), taxonomy=str(self.root / 'taxonomy.json'), directions=None, max_workers=1)
        for name in ['MAX_AI_CANDIDATES', 'MAX_DETAIL_ITEMS']:
            with patch.dict(os.environ, {name: '0'}), patch.object(enhance, 'parse_args', return_value=args), patch.object(routing, 'CachedChain') as chain:
                with self.assertRaises(runtime.FatalAIError):
                    enhance.main()
                chain.assert_not_called()

    def test_progress_waiting_and_atomic_usage(self):
        with patch.dict(os.environ, AI_METRICS_PATH=str(self.root / 'usage.json')):
            runtime.metric('filter', calls=1)
            self.assertEqual(json.loads((self.root / 'usage.json').read_text())['filter']['calls'], 1)
        with contextlib.redirect_stderr(io.StringIO()) as stream, ThreadPoolExecutor(max_workers=1) as executor:
            import time
            futures = {executor.submit(time.sleep, .035): 0}
            status = progress.Progress('filter', 1)
            for future in progress.completed(futures, status, heartbeat=.01):
                future.result(); status.complete()
        text = stream.getvalue()
        self.assertIn('start: 0/1', text)
        self.assertIn('waiting:', text)
        self.assertIn('complete: 1/1', text)

    def test_cancel_stops_queued_work_and_preserves_completed_checkpoint(self):
        done, pending = Future(), Future()
        done.set_result('successful')
        chain = Mock(stopped=progress.Event())
        iterator = progress.completed({done: 'done', pending: 'pending'}, progress.Progress('filter', 2), chain)
        self.assertIs(next(iterator), done)
        progress.checkpoint(self.root / 'audit.json', dict(completed=[done.result()]))
        iterator.close()
        self.assertTrue(pending.cancelled())
        self.assertTrue(chain.stopped.is_set())
        self.assertEqual(json.loads((self.root / 'audit.json').read_text())['completed'], ['successful'])

    def test_recovery_checksums_and_success_cache_reuse(self):
        spec = importlib.util.spec_from_file_location('recovery', ROOT / 'scripts/recovery.py')
        recovery = importlib.util.module_from_spec(spec); spec.loader.exec_module(recovery)
        data = self.root / 'data'; stage = self.root / 'stage'; output = self.root / 'recovery'
        data.mkdir(); stage.mkdir()
        (data / '2026-09-30.jsonl').write_text('{}\n')
        responder = Mock(return_value=runtime.AIMessage(content='successful result'))
        from langchain_core.prompts import ChatPromptTemplate
        with patch.dict(os.environ, AI_CACHE_DIR=str(data / 'ai_cache')), patch.object(runtime, 'ChatOpenAI', return_value=responder):
            chain = runtime.CachedChain(ChatPromptTemplate.from_messages([('human', '{content}')]), 'deepseek-flash', 'detail')
            chain.invoke(dict(content='same', paper_id='p'))
        recovery.save_bundle(data, stage, output, '2026-09-30')
        archive = self.root / 'bundle.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            for path in output.rglob('*'):
                if path.is_file(): z.write(path, path.relative_to(output))
        restored = self.root / 'restored'
        recovery.restore_zip(archive, restored)
        with patch.dict(os.environ, AI_CACHE_DIR=str(restored / 'ai_cache'), MAX_AI_CANDIDATES='100'), patch.object(runtime, 'ChatOpenAI', return_value=responder):
            chain = runtime.CachedChain(ChatPromptTemplate.from_messages([('human', '{content}')]), 'deepseek-flash', 'detail')
            self.assertEqual(chain.invoke(dict(content='same', paper_id='p')).content, 'successful result')
        self.assertEqual(responder.call_count, 1)
        manifest = json.loads((output / 'recovery.json').read_text())
        manifest['files']['data/2026-09-30.jsonl'] = 'invalid checksum'
        corrupt = self.root / 'corrupt.zip'
        with zipfile.ZipFile(corrupt, 'w') as z:
            for path in output.rglob('*'):
                if path.is_file():
                    z.writestr(str(path.relative_to(output)), json.dumps(manifest) if path.name == 'recovery.json' else path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'checksum'):
            recovery.restore_zip(corrupt, self.root / 'bad-restore')
        # Changing the actual prompt/input invalidates the recovered cached result.
        chain.invoke(dict(content='changed', paper_id='p'))
        self.assertEqual(responder.call_count, 2)
