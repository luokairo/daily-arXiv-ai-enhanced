"""Exercise the production schedule policy, checkpoint, restore and publish path."""
import json
import os
import runpy
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_pipeline import IsolatedTest, ROOT, structured, runtime, enhance
from test_dated_crawl import catchup
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from structure import RoutingStructure
from scripts import crawl_dated as crawl
from scripts import plan_crawl_retry as planner
from scripts.process_report import process, write_state
from scripts.recovery import save_bundle, restore_zip
from scripts.publish_staged import publish


DAY = '2026-11-06'


class AutomaticRecoveryTests(IsolatedTest):
    def test_schedule_crosses_peak_restores_snapshot_and_cache_then_publishes(self):
        data, stage = self.root / 'first/data', self.root / 'first/stage'
        html = catchup(day=DAY, summary='A novel world model benchmark.')
        with patch.object(crawl, 'recall_categories', return_value=['cs.CV']):
            audit = crawl.crawl(DAY, data, stage / 'run_metrics', Mock(get=Mock(return_value=html)))
        original = (data / f'{DAY}.jsonl').read_bytes()
        raw = AIMessage(content='ok', usage_metadata=dict(input_tokens=10, output_tokens=5, total_tokens=15))
        route = RoutingStructure(decision='relevant', matched_direction_ids=['world_model'],
            primary_direction_id='world_model', brief='论文贡献简讯', reason='World model contribution',
            personal_relevance_score=90, research_value_score=80)
        filter_calls = Mock(return_value=dict(raw=raw, parsed=route, parsing_error=None))
        detail_calls = Mock(return_value=dict(raw=raw, parsed=structured(), parsing_error=None))

        def fake_model(**settings):
            model = RunnableLambda(filter_calls if settings['max_tokens'] == 512 else detail_calls)
            model.with_structured_output = lambda *a, **k: model
            return model

        def execute(command, cwd, env):
            with patch.dict(os.environ, env):
                if command[1] == 'enhance.py':
                    args = SimpleNamespace(data=command[3], taxonomy=str(Path(command[3]).parent / 'taxonomy.json'),
                                           directions=None, max_workers=1)
                    with patch.object(enhance, 'parse_args', return_value=args):
                        try:
                            enhance.main()
                        except runtime.OffPeakWindowClosed:
                            return subprocess.CompletedProcess(command, 1)
                else:
                    previous = Path.cwd()
                    try:
                        os.chdir(cwd)
                        with patch('sys.argv', command[1:]):
                            runpy.run_path(str(cwd / command[1]), run_name='__main__')
                    finally:
                        os.chdir(previous)
            return subprocess.CompletedProcess(command, 0)

        environment = dict(ENFORCE_DEEPSEEK_OFF_PEAK='true', MODEL_NAME='deepseek-flash',
            SELECTION_MODE='semantic', ENABLE_DEEP_READ='false', FILTER_MAX_WORKERS='1',
            DETAIL_MAX_WORKERS='1', USE_MODEL_IMPORTANCE='false', AI_CACHE_DIR=str(data / 'ai_cache'))
        with patch.dict(os.environ, environment), patch.object(runtime, 'ChatOpenAI', side_effect=fake_model):
            # A schedule begins in the allowed window and crosses the boundary
            # between routing and detailed analysis. No partial report is promoted.
            with patch.object(runtime, 'datetime') as clock:
                clock.now.side_effect = [datetime(2026, 11, 9, 0, 44, tzinfo=timezone.utc),
                                         datetime(2026, 11, 9, 0, 45, tzinfo=timezone.utc)]
                self.assertFalse(process(DAY, data, stage, run=execute))
            filter_calls.assert_called_once()
            detail_calls.assert_not_called()
            self.assertFalse((stage / f'{DAY}.md').exists())
            deferred = json.loads((stage / f'run_metrics/{DAY}-pipeline.json').read_text())
            self.assertEqual(deferred['status'], 'deferred')
            self.assertEqual(deferred['phase'], 'enhance')
            bundle = self.root / 'bundle'
            save_bundle(data, stage, bundle, DAY)
            archive = self.root / 'bundle.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                for path in bundle.rglob('*'):
                    if path.is_file():
                        stream.write(path, str(path.relative_to(bundle)))
            attempt = dict(id=1, name='selection-audit-1', expired=False,
                           created_at='2026-11-09T00:46:00Z', workflow_run=dict(head_branch='main'))
            rows = [audit, deferred]
            with patch.object(planner, 'read_audits', return_value=rows):
                peak = datetime(2026, 11, 9, 1, tzinfo=timezone.utc).timestamp()
                noon = datetime(2026, 11, 9, 4, tzinfo=timezone.utc).timestamp()
                self.assertIsNone(planner.plan({'schedule': '17 * * * *'}, '', [attempt], peak))
                self.assertEqual(planner.plan({'schedule': '17 * * * *'}, '', [attempt], noon), DAY)
            # A new runner restores only verified input/cache, then runs the
            # scheduled policy again. A network call here is a test failure.
            restored, resumed = self.root / 'second/data', self.root / 'second/stage'
            restore_zip(archive, restored)
            network = Mock(get=Mock(side_effect=AssertionError('Snapshot must avoid network')))
            with patch.object(crawl, 'recall_categories', return_value=['cs.CV']):
                crawl.crawl(DAY, restored, resumed / 'run_metrics', network)
            network.get.assert_not_called()
            self.assertEqual((restored / f'{DAY}.jsonl').read_bytes(), original)
            with patch.dict(os.environ, AI_CACHE_DIR=str(restored / 'ai_cache')), patch.object(runtime, 'datetime') as clock:
                clock.now.return_value = datetime(2026, 11, 9, 4, tzinfo=timezone.utc)
                self.assertTrue(process(DAY, restored, resumed, run=execute))
            filter_calls.assert_called_once()  # Routing reused the first runner's cache.
            detail_calls.assert_called_once()
            publish(resumed, restored)
            self.assertTrue((restored / f'{DAY}.md').exists())
            self.assertEqual(len((restored / f'{DAY}_AI_enhanced_Chinese.jsonl').read_text().splitlines()), 1)
            with patch.object(planner, 'read_audits', return_value=rows):
                self.assertIsNone(planner.plan({'schedule': '17 * * * *'}, '', [attempt], noon,
                                              published_dates={DAY}))

    def test_terminal_processing_failure_stays_red_and_is_not_deferred(self):
        runner = Mock(return_value=subprocess.CompletedProcess(['enhance.py'], 1))
        with self.assertRaises(subprocess.CalledProcessError):
            process(DAY, self.root, self.root / 'stage', run=runner)
        state = json.loads((self.root / f'stage/run_metrics/{DAY}-pipeline.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertFalse(state['retryable'])

    def test_deep_read_deferral_prevents_publication_even_with_markdown(self):
        stage = self.root / 'stage'
        def execute(command, cwd, env):
            if command[1] == 'enhance.py':
                (stage / f'{DAY}_AI_enhanced_Chinese.jsonl').write_text('{}\n')
            elif command[1] == 'convert.py':
                (stage / f'{DAY}.md').write_text('staged report')
            else:
                write_state(Path(env['PIPELINE_STATE_PATH']), DAY, status='deferred', phase='deep_read',
                            error_type='OffPeakWindowClosed', retryable=True, cooldown_scope='date', retry_not_before=9999999999)
                return subprocess.CompletedProcess(command, 1)
            return subprocess.CompletedProcess(command, 0)
        with patch.dict(os.environ, ENABLE_DEEP_READ='true'):
            self.assertFalse(process(DAY, self.root, stage, run=execute))
