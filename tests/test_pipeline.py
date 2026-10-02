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
sys.path[:0] = [str(ROOT / "ai"), str(ROOT)]
import runtime
import enhance
import deep_read
import semantic_arxiv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from structure import Structure

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
            with patch.object(runtime.time, "sleep") as sleep:
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


class InterestTests(IsolatedTest):
    def test_priority_cap_and_deep_read_selection(self):
        directions = semantic_arxiv.load_directions()
        importance = semantic_arxiv.load_importance_config()
        low = {**PAPER, "id": "1", "title": "A generic study", "summary": "", "categories": ["cs.CV"]}
        high = {**PAPER, "id": "2", "title": "Continuous vision-language model with latent flow", "summary": "continuous vlm latent flow"}
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
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": name, "arguments": structured().model_dump_json()}}]}
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
        self.assertEqual(len(requests), 2, "Second complete run must reuse both successful results")
        self.assertEqual([r.get("max_tokens", r.get("max_completion_tokens")) for r in requests], [2500, 6000])
        self.assertTrue(all(r["thinking"] == {"type": "disabled"} for r in requests))
        self.assertEqual(json.loads(Path(read_args.output_json).read_text())[0]["status"], "ok")
        self.assertEqual(runtime._METRICS["detail"]["cache_hits"], 1)
        self.assertEqual(runtime._METRICS["deep_read"]["cache_hits"], 1)
        self.assertEqual(runtime._METRICS["deep_read"]["input_tokens"], 100)
        self.assertEqual(taxonomy.read_text(), "{}", "Staged pipeline must not change published taxonomy")


if __name__ == "__main__":
    unittest.main()
