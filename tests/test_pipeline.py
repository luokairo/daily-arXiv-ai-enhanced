import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ai"), str(ROOT)]
import runtime
import enhance
import deep_read
import semantic_arxiv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from structure import Structure, RoutingStructure
import routing

DIRECTIONS = [dict(id="world_model", name="世界模型", keywords=["world model"], arxiv_categories=["cs.AI"], canonical_subtopics=[])]
PAPER = dict(id="2609.00001", title="A world model", summary="A novel world model benchmark.", categories=["cs.AI"], authors=[])


def structured(relevant=True):
    return Structure(**{**enhance.default_ai_fields(), "is_relevant": relevant,
                        "primary_direction_id": "world_model", "matched_direction_ids": ["world_model"],
                        "tldr": "摘要", "method": "方法", "motivation": "动机", "result": "实验", "conclusion": "结论"})


class IsolatedTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.environment = patch.dict(os.environ, dict(OPENAI_BASE_URL=" https://api.deepseek.com/ ", OPENAI_API_KEY=" fake ",
            AI_CACHE_DIR=str(self.root / "cache")), clear=True)
        self.environment.start()
        runtime._METRICS.clear()

    def tearDown(self):
        self.environment.stop()
        self.tmp.cleanup()
        runtime._METRICS.clear()


class RuntimeTests(IsolatedTest):
    def test_configuration_defaults_aliases_and_validation(self):
        for name in (" deepseek-flash ", "deepseek-v4-flash", "deepseek-v4-pro"):
            config = runtime.model_settings(name, "detail")
            self.assertEqual(config["extra_body"], {"thinking": {"type": "disabled"}})
            self.assertEqual(config["base_url"], "https://api.deepseek.com")
            self.assertEqual(config["api_key"], "fake")
            self.assertEqual(config["max_tokens"], 2500)
            self.assertEqual(config["max_retries"], 0)
        self.assertEqual(runtime.model_settings("deepseek-flash", "deep_read")["max_tokens"], 6000)
        self.assertNotIn("extra_body", runtime.model_settings("gpt-test", "detail"))
        with patch.dict(os.environ, OPENAI_API_KEY=" "):
            with self.assertRaises(runtime.FatalAIError):
                runtime.model_settings("deepseek-flash", "detail")
        with patch.dict(os.environ, DETAIL_MAX_OUTPUT_TOKENS="0"):
            with self.assertRaises(runtime.FatalAIError):
                runtime.model_settings("deepseek-flash", "detail")

    def chain(self, responder, schema=None, template="{content}", model="deepseek-flash"):
        model_runnable = RunnableLambda(responder)
        model_runnable.with_structured_output = lambda *a, **k: model_runnable
        with patch.object(runtime, "ChatOpenAI", return_value=model_runnable):
            return runtime.CachedChain(ChatPromptTemplate.from_template(template), model, "detail", schema)

    def test_cache_usage_and_invalidation(self):
        response = AIMessage(content="ok", usage_metadata=dict(input_tokens=10, output_tokens=5, total_tokens=15))
        responder = Mock(return_value=response)
        chain = self.chain(responder)
        inputs = dict(paper_id="1", content="same abstract")
        chain.invoke(inputs)
        chain.invoke(inputs)
        self.assertEqual(responder.call_count, 1)
        row = runtime._METRICS["detail"]
        self.assertEqual((row["cache_hits"], row["input_tokens"], row["output_tokens"]), (1, 10, 5))
        chain.invoke({**inputs, "paper_id": "2"})
        chain.invoke({**inputs, "content": "new abstract"})
        self.chain(responder, template="updated {content}").invoke(inputs)
        self.chain(responder, model="deepseek-v4-pro").invoke(inputs)
        with patch.dict(os.environ, DETAIL_MAX_OUTPUT_TOKENS="2600"):
            self.chain(responder).invoke(inputs)
        self.assertEqual(responder.call_count, 6)
        for path in (self.root / "cache").rglob("*.json"):
            self.assertNotIn("fake", path.read_text())

    def test_structured_cache_and_truncation_are_not_retried(self):
        responder = Mock(return_value={"raw": AIMessage(content="", response_metadata={"finish_reason": "stop"}),
                                      "parsed": structured(), "parsing_error": None})
        chain = self.chain(responder, Structure)
        self.assertIsInstance(chain.invoke(dict(paper_id="1", content="abstract")), Structure)
        self.assertIsInstance(chain.invoke(dict(paper_id="1", content="abstract")), Structure)
        self.assertEqual(responder.call_count, 1)
        failed = Mock(return_value=AIMessage(content="truncated", response_metadata={"finish_reason": "length"}))
        chain = self.chain(failed)
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "token limit"):
                chain.invoke(dict(content="new"))
        self.assertEqual(failed.call_count, 2)

    def test_invalid_structured_output_not_cached(self):
        responder = Mock(return_value={"raw": AIMessage(content="bad"), "parsed": None, "parsing_error": ValueError()})
        chain = self.chain(responder, Structure)
        for _ in range(2):
            with self.assertRaises(ValueError):
                chain.invoke(dict(content="same"))
        self.assertEqual(responder.call_count, 2)

    def test_bounded_transient_retries_and_fatal_stop(self):
        class HttpError(Exception):
            def __init__(self, status):
                self.status_code = status
        for error in (HttpError(429), HttpError(503), TimeoutError()):
            responder = Mock(side_effect=error)
            chain = self.chain(responder)
            with patch.object(chain.stopped, "wait") as sleep:
                with self.assertRaises(type(error)):
                    chain.invoke(dict(content="transient"))
            self.assertEqual(responder.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
        for status in (400, 401, 403, 404, 422):
            responder = Mock(side_effect=HttpError(status))
            chain = self.chain(responder)
            for _ in range(2):
                with self.assertRaises(runtime.FatalAIError):
                    chain.invoke(dict(content="fatal"))
            self.assertEqual(responder.call_count, 1)

    def test_unknown_usage_and_fallback_not_cached(self):
        responder = Mock(return_value=AIMessage(content="abstract fallback"))
        chain = self.chain(responder)
        for _ in range(2):
            chain.invoke(dict(content="a", _cache_allowed=False))
        self.assertEqual(responder.call_count, 2)
        self.assertEqual(runtime._METRICS["detail"]["unknown_usage"], 2)

    def test_off_peak_block_has_no_paid_call_and_reports_reason(self):
        summary = self.root / 'summary.md'
        responder = Mock(return_value=AIMessage(content='should not be requested'))
        with patch.dict(os.environ, ENFORCE_DEEPSEEK_OFF_PEAK='true', GITHUB_STEP_SUMMARY=str(summary)), \
                patch.object(runtime, 'datetime') as clock:
            chain = self.chain(responder)
            clock.now.return_value = datetime(2026, 10, 5, 1, tzinfo=timezone.utc)
            with self.assertRaisesRegex(runtime.OffPeakWindowClosed, 'off-peak protection'):
                chain.invoke(dict(content='uncached'))
        responder.assert_not_called()
        self.assertTrue(chain.stopped.is_set())
        row = runtime._METRICS['detail']
        self.assertEqual((row['calls'], row['unknown_usage'], row['failures']), (0, 0, 0))
        self.assertIn('incomplete reports will not be published', summary.read_text())

    def test_off_peak_toggle_preserves_cache_and_busy_cache_hits_are_free(self):
        responder = Mock(return_value=AIMessage(content='successful result'))
        inputs = dict(paper_id='1', content='same abstract')
        original = self.chain(responder)
        original.invoke(inputs)
        with patch.dict(os.environ, ENFORCE_DEEPSEEK_OFF_PEAK='true'):
            guarded = self.chain(responder)
        self.assertEqual(original.signature, guarded.signature)
        with patch.object(runtime, 'datetime') as clock:
            clock.now.return_value = datetime(2026, 10, 5, 1, tzinfo=timezone.utc)
            self.assertEqual(guarded.invoke(inputs).content, 'successful result')
            clock.now.assert_not_called()
        self.assertEqual(responder.call_count, 1)
        self.assertEqual(runtime._METRICS['detail']['cache_hits'], 1)

    def test_transient_retry_checks_window_again_without_counting_blocked_call(self):
        responder = Mock(side_effect=[TimeoutError(), AIMessage(content='must not retry')])
        with patch.dict(os.environ, ENFORCE_DEEPSEEK_OFF_PEAK='true'):
            chain = self.chain(responder)
        with patch.object(runtime, 'datetime') as clock, patch.object(chain.stopped, 'wait'):
            clock.now.side_effect = [datetime(2026, 10, 5, 0, 44, 59, tzinfo=timezone.utc),
                                     datetime(2026, 10, 5, 0, 45, tzinfo=timezone.utc)]
            with self.assertRaises(runtime.OffPeakWindowClosed):
                chain.invoke(dict(content='retry crossing boundary'))
        self.assertEqual(responder.call_count, 1)
        self.assertEqual(runtime._METRICS['detail']['calls'], 1)
        self.assertEqual(runtime._METRICS['detail']['unknown_usage'], 1)

    def test_started_request_can_complete_and_is_cached(self):
        responder = Mock(return_value=AIMessage(content='completed before the next check'))
        with patch.dict(os.environ, ENFORCE_DEEPSEEK_OFF_PEAK='true'):
            chain = self.chain(responder)
        with patch.object(runtime, 'datetime') as clock:
            clock.now.side_effect = [datetime(2026, 10, 5, 0, 44, 59, tzinfo=timezone.utc),
                                     datetime(2026, 10, 5, 1, tzinfo=timezone.utc)]
            self.assertEqual(chain.invoke(dict(content='admitted')).content, 'completed before the next check')
            self.assertEqual(chain.invoke(dict(content='admitted')).content, 'completed before the next check')
            with self.assertRaises(runtime.OffPeakWindowClosed):
                chain.invoke(dict(content='next uncached request'))
        self.assertEqual(responder.call_count, 1)
        self.assertEqual(runtime._METRICS['detail']['cache_hits'], 1)
        self.assertEqual(len(list((self.root / 'cache/detail').glob('*.json'))), 1)

    def test_manual_default_and_other_providers_do_not_enforce_deepseek_prices(self):
        cases = [({}, 'deepseek-flash'),
                 ({'ENFORCE_DEEPSEEK_OFF_PEAK': 'true'}, 'gpt-test'),
                 ({'ENFORCE_DEEPSEEK_OFF_PEAK': 'true', 'OPENAI_BASE_URL': 'https://example.com/v1'}, 'deepseek-flash')]
        for index, (environment, model) in enumerate(cases):
            with self.subTest(model=model, environment=environment), patch.dict(os.environ, environment):
                responder = Mock(return_value=AIMessage(content='allowed'))
                chain = self.chain(responder, model=model)
                with patch.object(runtime, 'check_deepseek_off_peak') as check:
                    chain.invoke(dict(content=str(index)))
                check.assert_not_called()
                responder.assert_called_once()


class OffPeakTimeTests(unittest.TestCase):
    def test_china_weekday_boundaries_and_utc_conversion(self):
        china = timezone(timedelta(hours=8))
        cases = [(0, 0, 0, True), (1, 23, 0, True), (8, 44, 59, True),
                 (8, 45, 0, False), (9, 0, 0, False), (11, 59, 59, False),
                 (12, 0, 0, True), (13, 44, 59, True), (13, 45, 0, False),
                 (14, 0, 0, False), (17, 59, 59, False), (18, 0, 0, True),
                 (23, 59, 59, True)]
        for hour, minute, second, expected in cases:
            with self.subTest(time=(hour, minute, second)):
                local = datetime(2026, 10, 5, hour, minute, second, tzinfo=china)
                self.assertEqual(runtime.deepseek_off_peak_allowed(local), expected)
                self.assertEqual(runtime.deepseek_off_peak_allowed(local.astimezone(timezone.utc)), expected)

    def test_weekends_and_conservative_holiday_policy(self):
        for day in (3, 4):
            for hour in range(24):
                self.assertTrue(runtime.deepseek_off_peak_allowed(datetime(2026, 10, day, hour, tzinfo=timezone.utc)))
        # Chinese holidays do not bypass the ordinary UTC weekday guard.
        self.assertFalse(runtime.deepseek_off_peak_allowed(datetime(2026, 10, 1, 1, tzinfo=timezone.utc)))
        with self.assertRaisesRegex(ValueError, 'timezone-aware'):
            runtime.deepseek_off_peak_allowed(datetime(2026, 10, 5, 1))


class InterestTests(IsolatedTest):
    def test_priority_cap_and_deep_read_selection(self):
        directions = semantic_arxiv.load_directions()
        importance = semantic_arxiv.load_importance_config()
        low = {**PAPER, "id": "1", "title": "A generic study", "summary": "", "categories": ["cs.CV"]}
        high = {**PAPER, "id": "2", "title": "Video world model with action-conditioned video", "summary": "video world model"}
        result = enhance.filter_all_items([low, high], "unused", 1, directions, importance, max_items=1)
        self.assertEqual([paper["id"] for paper in result], ["2"])
        papers = [{"id": str(i), "AI": {"importance_score": score}} for i, score in enumerate((20, 80, 95, 90))]
        enhance.mark_deep_read_selection(papers, 3)
        self.assertFalse(papers[0]["AI"]["deep_read_selected"])
        self.assertEqual(papers[2]["AI"]["deep_read_rank"], 1)
        # Equal-priority candidate order is deterministic, independent of crawl order.
        same = [{**PAPER, "id": key} for key in ("3", "1", "2")]
        result = enhance.filter_all_items(same, "unused", 1, DIRECTIONS, {}, max_items=2)
        self.assertEqual([paper["id"] for paper in result], ["1", "2"])

    def test_taxonomy_keeps_all_canonical_and_five_dynamic(self):
        subs = [dict(id=f"c{i}", name=f"canonical{i}", is_canonical=True, paper_count=i) for i in range(12)]
        subs += [dict(id=f"d{i}", name=f"dynamic{i}", paper_count=i) for i in range(9)]
        text = semantic_arxiv.compact_taxonomy_for_prompt({"directions": {"test": {"subtopics": subs}}})
        self.assertEqual(text.count("[CANONICAL]"), 12)
        self.assertEqual(text.count("dynamic"), 5)
        self.assertIn("dynamic8", text)
        self.assertNotIn("dynamic0", text)
        self.assertNotIn("papers:", text)

    def test_detail_mixed_failure_all_failure_and_irrelevant(self):
        items = [copy.deepcopy(PAPER), {**PAPER, "id": "2"}]
        with patch.object(enhance, "CachedChain", return_value=Mock()), patch.object(enhance, "process_single_item", side_effect=[ValueError("bad JSON"), items[1]]):
            result = enhance.process_all_items(items, "model", "Chinese", 1, DIRECTIONS, {})
            self.assertEqual([item["id"] for item in result], ["2"])
        with patch.object(enhance, "CachedChain", return_value=Mock()), patch.object(enhance, "process_single_item", side_effect=ValueError("bad JSON")):
            with self.assertRaisesRegex(RuntimeError, "All detail"):
                enhance.process_all_items(items, "model", "Chinese", 1, DIRECTIONS, {})
        with patch.object(enhance, "CachedChain", return_value=Mock()), patch.object(enhance, "process_single_item", return_value={}):
            self.assertEqual(enhance.process_all_items(items, "model", "Chinese", 1, DIRECTIONS, {}), [])
        with patch.object(enhance, "CachedChain", return_value=Mock()), patch.object(enhance, "process_single_item", side_effect=runtime.FatalAIError("auth")):
            with self.assertRaises(runtime.FatalAIError):
                enhance.process_all_items(items, "model", "Chinese", 1, DIRECTIONS, {})

    def test_optional_model_stages_fail_without_publishing_placeholders(self):
        uncertain = {**PAPER, "title": "Generic research", "summary": "", "categories": ["cs.CV"]}
        with patch.dict(os.environ, USE_MODEL_FILTER="true"), patch.object(enhance, "CachedChain", return_value=Mock(invoke=Mock(side_effect=ValueError("parse")))):
            with self.assertRaisesRegex(RuntimeError, "All model-filter"):
                enhance.filter_all_items([uncertain], "model", 1, DIRECTIONS, {}, max_items=30)
        paper = {**PAPER, "AI": structured().model_dump()}
        with patch.dict(os.environ, USE_MODEL_IMPORTANCE="true"), patch.object(enhance, "CachedChain", return_value=Mock(invoke=Mock(side_effect=ValueError("parse")))):
            with self.assertRaisesRegex(RuntimeError, "All importance"):
                enhance.score_importance_for_items([paper], "model", 1, {})
        invalid = structured().model_copy(update={"primary_direction_id": "stale", "matched_direction_ids": []})
        with self.assertRaisesRegex(ValueError, "no configured"):
            enhance.validate_classification(invalid, DIRECTIONS)
        enhance.validate_classification(structured(False), DIRECTIONS)


    @patch.dict(os.environ, SELECTION_MODE="legacy")
    def test_main_preserves_existing_files_on_failure_and_stages_success(self):
        source = self.root / "2026-10-01.jsonl"
        source.write_text(json.dumps(PAPER) + "\n")
        target = self.root / "2026-10-01_AI_enhanced_Chinese.jsonl"
        target.write_text("old successful report")
        taxonomy = self.root / "taxonomy.json"
        taxonomy.write_text("{}")
        args = SimpleNamespace(data=str(source), taxonomy=str(taxonomy), directions=None, max_workers=1)
        with patch.object(enhance, "parse_args", return_value=args), patch.object(enhance, "process_all_items", side_effect=RuntimeError("all failed")):
            with self.assertRaises(RuntimeError):
                enhance.main()
        self.assertEqual(target.read_text(), "old successful report")
        self.assertEqual(taxonomy.read_text(), "{}")
        stage = self.root / "stage"
        item = {**PAPER, "AI": structured().model_dump()}
        with patch.dict(os.environ, AI_OUTPUT_DIR=str(stage)), patch.object(enhance, "parse_args", return_value=args), patch.object(enhance, "process_all_items", return_value=[item]):
            enhance.main()
        self.assertTrue((stage / target.name).exists())
        self.assertTrue((stage / "taxonomy.json").exists())
        self.assertEqual(target.read_text(), "old successful report")
        self.assertEqual(taxonomy.read_text(), "{}")


class SemanticRoutingTests(IsolatedTest):
    def result(self, decision='relevant'):
        return RoutingStructure(decision=decision,
            matched_direction_ids=['world_model'] if decision == 'relevant' else [],
            primary_direction_id='world_model' if decision == 'relevant' else '', brief='论文贡献简讯',
            reason='Relevant abstract evidence', personal_relevance_score=90, research_value_score=80)

    def test_off_peak_stop_preserves_reports_and_saves_success_cache_for_recovery(self):
        from scripts.recovery import save_bundle
        source = self.root / '2026-10-01.jsonl'
        source.write_text(json.dumps(PAPER) + '\n')
        target = self.root / '2026-10-01_AI_enhanced_Chinese.jsonl'
        target.write_text('old successful report')
        markdown = self.root / '2026-10-01.md'
        markdown.write_text('old successful Markdown')
        taxonomy = self.root / 'taxonomy.json'
        taxonomy.write_text('{}')
        stage = self.root / 'stage'
        args = SimpleNamespace(data=str(source), taxonomy=str(taxonomy), directions=None, max_workers=1)
        raw = AIMessage(content='route', usage_metadata=dict(input_tokens=10, output_tokens=5, total_tokens=15))
        filter_responder = Mock(return_value=dict(raw=raw, parsed=self.result(), parsing_error=None))
        detail_responder = Mock(return_value=dict(raw=raw, parsed=structured(), parsing_error=None))
        def fake_model(**settings):
            model = RunnableLambda(filter_responder if settings['max_tokens'] == 512 else detail_responder)
            model.with_structured_output = lambda *a, **k: model
            return model
        environment = dict(SELECTION_MODE='semantic', ENFORCE_DEEPSEEK_OFF_PEAK='true',
                           AI_CACHE_DIR=str(self.root / 'ai_cache'), AI_OUTPUT_DIR=str(stage))
        with patch.dict(os.environ, environment), patch.object(enhance, 'parse_args', return_value=args), \
                patch.object(runtime, 'ChatOpenAI', side_effect=fake_model), patch.object(runtime, 'datetime') as clock:
            clock.now.side_effect = [datetime(2026, 10, 5, 0, 44, tzinfo=timezone.utc),
                                     datetime(2026, 10, 5, 0, 45, tzinfo=timezone.utc)]
            with self.assertRaises(runtime.OffPeakWindowClosed):
                enhance.main()
        filter_responder.assert_called_once()
        detail_responder.assert_not_called()
        self.assertEqual(target.read_text(), 'old successful report')
        self.assertEqual(markdown.read_text(), 'old successful Markdown')
        self.assertEqual(taxonomy.read_text(), '{}')
        self.assertFalse((stage / target.name).exists())
        self.assertTrue((stage / 'run_metrics/2026-10-01-selection.json').exists())
        bundle = self.root / 'recovery'
        save_bundle(self.root, stage, bundle, '2026-10-01')
        self.assertEqual(len(list((bundle / 'data/ai_cache/filter').glob('*.json'))), 1)
        self.assertTrue((bundle / 'data/run_metrics/2026-10-01-selection.json').exists())
        self.assertTrue((bundle / 'recovery.json').exists())

    def test_all_abstracts_reviewed_including_local_drops_and_uncertain_forwarded(self):
        papers = [{**PAPER, 'id': str(i), 'title': 'Unfamiliar terminology', 'categories': ['cs.RO']}
                  for i in range(35)]
        # These are discarded by the legacy title-only gate.
        self.assertEqual(enhance.local_filter_item(papers[0], DIRECTIONS,
            enhance.collect_filter_keywords(DIRECTIONS, {}))['state'], 'drop')
        def respond(inputs):
            return self.result({'0': 'irrelevant', '1': 'uncertain'}.get(inputs['paper_id'], 'relevant'))
        chain = Mock(invoke=Mock(side_effect=respond))
        audit = {}
        with patch.dict(os.environ, MAX_DETAIL_ITEMS='1', USE_MODEL_FILTER='false'), patch.object(routing, 'CachedChain', return_value=chain):
            selected = routing.route_all_items(papers, 'model', 2, DIRECTIONS, {}, audit)
        self.assertEqual(chain.invoke.call_count, 35)
        self.assertEqual(len(selected), 34)
        self.assertIn('1', [p['id'] for p in selected])
        self.assertEqual(audit['0']['detail_status'], 'not_requested')

    def test_failures_forwarded_but_all_failures_and_auth_abort(self):
        papers = [{**PAPER, 'id': str(i)} for i in range(2)]
        audit = {}
        chain = Mock(invoke=Mock(side_effect=[ValueError('invalid'), self.result('irrelevant')]))
        with patch.object(routing, 'CachedChain', return_value=chain):
            selected = routing.route_all_items(papers, 'model', 1, DIRECTIONS, {}, audit)
        self.assertEqual([p['id'] for p in selected], ['0'])
        self.assertEqual(audit['0']['routing_status'], 'failed')
        for error in (ValueError('invalid'), runtime.FatalAIError('auth')):
            with patch.object(routing, 'CachedChain', return_value=Mock(invoke=Mock(side_effect=error))):
                with self.assertRaises((RuntimeError, runtime.FatalAIError)):
                    routing.route_all_items(papers, 'model', 1, DIRECTIONS, {}, {})

    def test_short_keywords_and_unclipped_ranking(self):
        for text in ['self supervision', 'off-the-shelf', 'itself']:
            self.assertFalse(enhance.keyword_matches(text, 'elf'))
        self.assertTrue(enhance.keyword_matches('ELF: embedded language flow', 'elf'))
        self.assertTrue(enhance.keyword_matches('CoLa language model', 'cola'))
        self.assertFalse(enhance.keyword_matches('chocolate model', 'cola'))
        config = {'priority_subtopics': [{'name': 'worlds', 'weight': 4, 'keywords': ['world model']}],
                  'boost_keywords': ['extra']}
        low = {**PAPER, 'AI': {}}
        high = {**PAPER, 'summary': PAPER['summary'] + ' extra', 'AI': {}}
        self.assertGreater(enhance.heuristic_importance_score(high, config),
                           enhance.heuristic_importance_score(low, config))
        ranked = enhance.score_importance_for_items([low, high], 'unused', 1, config)
        enhance.mark_deep_read_selection([{**ranked[0], 'id': '2'}, {**ranked[1], 'id': '1'}], 1)
        self.assertFalse(ranked[0]['AI']['deep_read_selected'])
        self.assertTrue(ranked[1]['AI']['deep_read_selected'])

    def test_main_deduplicates_applies_detail_cap_and_writes_audit(self):
        source = self.root / '2026-10-01.jsonl'
        source.write_text('\n'.join(json.dumps({**PAPER, 'id': key}) for key in ['1', '1', '2', '3']))
        args = SimpleNamespace(data=str(source), taxonomy=str(self.root / 'taxonomy.json'), directions=None, max_workers=1)
        chain = Mock(invoke=Mock(side_effect=lambda inputs: self.result('uncertain')))
        def detail(items, *args, **kwargs):
            for p in items:
                p['_detail_status'] = 'relevant'
                p['AI'] = structured().model_dump()
            return items
        with patch.dict(os.environ, MAX_DETAIL_ITEMS='1', USE_MODEL_FILTER='false'), patch.object(enhance, 'parse_args', return_value=args), patch.object(routing, 'CachedChain', return_value=chain), patch.object(enhance, 'process_all_items', side_effect=detail):
            enhance.main()
        self.assertEqual(chain.invoke.call_count, 3)
        audit = json.loads((self.root / 'run_metrics/2026-10-01-selection.json').read_text())
        self.assertEqual(audit['unique_papers'], 3)
        self.assertEqual(sum(p['detail_status'] == 'relevant' for p in audit['papers']), 1)
        self.assertEqual(sum(p['detail_status'] == 'quota_deferred' for p in audit['papers']), 2)
        output = list(map(json.loads, (self.root / '2026-10-01_AI_enhanced_Chinese.jsonl').read_text().splitlines()))
        self.assertEqual(len(output), 3)
        self.assertIn('70%', output[0]['AI']['importance_reason'])

    def test_invalid_routes_and_all_irrelevant(self):
        for result in [self.result().model_copy(update={'matched_direction_ids': ['invalid']}),
                       self.result().model_copy(update={'matched_direction_ids': []})]:
            with self.assertRaises(ValueError):
                routing.validate_route(result, DIRECTIONS)
        with patch.object(routing, 'CachedChain', return_value=Mock(invoke=Mock(return_value=self.result('irrelevant')))):
            self.assertEqual(routing.route_all_items([PAPER.copy()], 'model', 1, DIRECTIONS, {}, {}), [])

    def test_previously_missed_papers_reach_model_and_failed_run_keeps_audit(self):
        papers = [dict(id='2610.01016', title='Scaling and Distilling Text Embeddings for Better Diffusibility',
                       categories=['cs.CL'], summary='Continuous diffusion language models use text embeddings.'),
                  dict(id='2610.02054', title='UniWAM: Unified World-Action Model',
                       categories=['cs.RO'], summary='Unified world generation and action prediction.')]
        chain = Mock(invoke=Mock(return_value=self.result('uncertain')))
        with patch.object(routing, 'CachedChain', return_value=chain):
            selected = routing.route_all_items(papers, 'model', 1, DIRECTIONS, {}, {})
        self.assertEqual({p['id'] for p in selected}, {'2610.01016', '2610.02054'})
        self.assertEqual({c.args[0]['abstract'] for c in chain.invoke.call_args_list}, {p['summary'] for p in papers})
        source = self.root / 'sample.jsonl'
        source.write_text(json.dumps(PAPER) + '\n')
        target = self.root / 'sample_AI_enhanced_Chinese.jsonl'
        target.write_text('previous successful report')
        args = SimpleNamespace(data=str(source), taxonomy=str(self.root / 'taxonomy.json'), directions=None, max_workers=1)
        with patch.object(enhance, 'parse_args', return_value=args), patch.object(routing, 'CachedChain',
                return_value=Mock(invoke=Mock(side_effect=ValueError('invalid')))):
            with self.assertRaisesRegex(RuntimeError, 'All routing'):
                enhance.main()
        self.assertEqual(target.read_text(), 'previous successful report')
        audit = json.loads((self.root / 'run_metrics/sample-selection.json').read_text())
        self.assertEqual(audit['papers'][0]['routing_status'], 'failed')


class DeepReadTests(IsolatedTest):
    def test_section_budgets_and_no_duplicate_abstract(self):
        text = "Introduction\n" + "intro " * 5000 + "\nMethod\n" + "METHOD_EVIDENCE " * 2000 + "\nExperiments\n" + "EXPERIMENT_EVIDENCE " * 2000 + "\nLimitations\n" + "LIMITATION_EVIDENCE " * 2000
        context, note = deep_read.build_reading_context(PAPER, text)
        self.assertLessEqual(len(context), 40000)
        for marker in ("METHOD_EVIDENCE", "EXPERIMENT_EVIDENCE", "LIMITATION_EVIDENCE"):
            self.assertIn(marker, context)
        self.assertEqual(context.count(PAPER["summary"]), 1)
        self.assertNotIn("abstract-only fallback", note)
        context, note = deep_read.build_reading_context(PAPER, "short")
        self.assertIn("not a full-text", note)
        self.assertEqual(context, PAPER["summary"])

    def test_all_deep_read_failures_preserve_existing_report(self):
        source = self.root / "papers.jsonl"
        source.write_text(json.dumps({**PAPER, "AI": {"importance_score": 90}}) + "\n")
        output = self.root / "report.md"
        output.write_text("old report")
        args = SimpleNamespace(data=str(source), output=str(output), output_json="", directions=None)
        with patch.object(deep_read, "parse_args", return_value=args), patch.object(deep_read, "CachedChain", return_value=Mock()), patch.object(deep_read, "run_deep_read", return_value={"status": "generation_failed"}):
            with self.assertRaisesRegex(RuntimeError, "All deep-read"):
                deep_read.main()
        self.assertEqual(output.read_text(), "old report")

    def test_pdf_failure_is_labelled_abstract_fallback(self):
        chain = Mock(return_value=None)
        chain.invoke.return_value = AIMessage(content="# Analysis")
        with patch.object(deep_read, "download_pdf", side_effect=ValueError("bad PDF")):
            result = deep_read.run_deep_read(chain, PAPER, 1, self.root)
        self.assertEqual(result["status"], "abstract_fallback")
        self.assertFalse(chain.invoke.call_args.args[0]["_cache_allowed"])
        self.assertIn("not a full-text deep read", chain.invoke.call_args.args[0]["source_note"])

    def test_publish_validates_whole_batch_before_replacing_files(self):
        spec = importlib.util.spec_from_file_location("publish", ROOT / "scripts/publish_staged.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stage, target = self.root / "stage", self.root / "data"
        stage.mkdir(); target.mkdir()
        (target / "taxonomy.json").write_text("old")
        (stage / "taxonomy.json").write_text("{}")
        (stage / "2026-10-01_AI_enhanced_Chinese.jsonl").write_text("broken JSON")
        with self.assertRaises(ValueError):
            module.publish(stage, target)
        self.assertEqual((target / "taxonomy.json").read_text(), "old")
        (stage / "2026-10-01_AI_enhanced_Chinese.jsonl").write_text("")
        module.publish(stage, target)
        self.assertEqual((target / "taxonomy.json").read_text(), "{}")


class SDKIntegrationTests(IsolatedTest):
    def test_real_sdk_payload_pipeline_and_repeat_cache(self):
        import httpx
        from langchain_openai import ChatOpenAI as RealChatOpenAI
        requests = []

        def transport(request):
            body = json.loads(request.content)
            requests.append(body)
            message = {"role": "assistant", "content": "# 精读\n方法、实验和局限。"}
            finish = "stop"
            if body.get("tools"):
                name = body["tools"][0]["function"]["name"]
                result = (RoutingStructure(decision='relevant', matched_direction_ids=['world_model'],
                    primary_direction_id='world_model', brief='世界模型研究简讯',
                    reason='World model research', personal_relevance_score=95, research_value_score=80)
                    if name == 'RoutingStructure' else structured())
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": name, "arguments": result.model_dump_json()}}]}
                finish = "tool_calls"
            return httpx.Response(200, json={"id": "fake-response", "object": "chat.completion", "created": 1,
                "model": "deepseek-flash", "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})

        client = httpx.Client(transport=httpx.MockTransport(transport))
        self.addCleanup(client.close)
        factory = lambda **kwargs: RealChatOpenAI(**kwargs, http_client=client)
        source = self.root / "2026-10-01.jsonl"
        source.write_text(json.dumps(PAPER) + "\n")
        taxonomy = self.root / "taxonomy.json"
        taxonomy.write_text("{}")
        stage = self.root / "stage"
        enhanced = stage / "2026-10-01_AI_enhanced_Chinese.jsonl"
        summary_args = SimpleNamespace(data=str(source), taxonomy=str(taxonomy), directions=None, max_workers=1)
        read_args = SimpleNamespace(data=str(enhanced), output=str(stage / "deep_reads/2026-10-01.md"),
                                   output_json=str(stage / "deep_reads/2026-10-01.json"), directions=None)
        full_text = "Method\n" + "Method evidence " * 200 + "\nExperiments\n" + "Experimental evidence " * 200 + "\nLimitations\nLimited scale."
        with patch.dict(os.environ, AI_OUTPUT_DIR=str(stage)), patch.object(runtime, "ChatOpenAI", side_effect=factory), patch.object(deep_read, "download_pdf", return_value=self.root / "fake.pdf"), patch.object(deep_read, "extract_pdf_text", return_value=full_text):
            for _ in range(2):
                with patch.object(enhance, "parse_args", return_value=summary_args):
                    enhance.main()
                with patch.object(deep_read, "parse_args", return_value=read_args):
                    deep_read.main()
        self.assertEqual(len(requests), 3, "Second complete run must reuse routing, summary and deep-read results")
        self.assertEqual([r.get("max_tokens", r.get("max_completion_tokens")) for r in requests], [512, 2500, 6000])
        self.assertTrue(all(r["thinking"] == {"type": "disabled"} for r in requests))
        self.assertEqual(json.loads(Path(read_args.output_json).read_text())[0]["status"], "ok")
        self.assertEqual(runtime._METRICS["detail"]["cache_hits"], 1)
        self.assertEqual(runtime._METRICS["filter"]["cache_hits"], 1)
        self.assertEqual(runtime._METRICS["deep_read"]["cache_hits"], 1)
        self.assertEqual(runtime._METRICS["deep_read"]["input_tokens"], 100)
        self.assertEqual(taxonomy.read_text(), "{}", "Staged pipeline must not change published taxonomy")


if __name__ == "__main__":
    unittest.main()
