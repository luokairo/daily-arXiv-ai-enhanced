"""Provider configuration, bounded requests, successful-result cache and usage reports."""
import atexit
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from threading import Event, Lock
from urllib.parse import urlparse

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError

ROOT = Path(__file__).resolve().parents[1]
_LOCK = Lock()
_METRICS_WRITE_LOCK = Lock()
_METRICS = {}


class FatalAIError(RuntimeError):
    """A configuration/authentication error that must stop publication."""


def env_value(name, default=""):
    return os.environ.get(name, "").strip() or default


def enabled(name, default=False):
    return env_value(name, "true" if default else "false").lower() in {"1", "true", "yes"}


def positive_int(name, default):
    try:
        value = int(env_value(name, str(default)))
    except ValueError as error:
        raise FatalAIError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise FatalAIError(f"{name} must be a positive integer")
    return value


def model_settings(model_name, stage):
    model_name = model_name.strip()
    base_url = env_value("OPENAI_BASE_URL").rstrip("/")
    api_key = env_value("OPENAI_API_KEY")
    missing = [name for name, value in (("MODEL_NAME", model_name),
               ("OPENAI_BASE_URL", base_url), ("OPENAI_API_KEY", api_key)) if not value]
    if missing:
        raise FatalAIError("Missing required configuration: " + ", ".join(missing))
    parsed = urlparse(base_url)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise FatalAIError("OPENAI_BASE_URL must be an HTTP(S) API URL")
    budget_name, budget = {
        "detail": ("DETAIL_MAX_OUTPUT_TOKENS", 2500),
        "deep_read": ("DEEP_READ_MAX_OUTPUT_TOKENS", 6000),
        "filter": ("FILTER_MAX_OUTPUT_TOKENS", 512),
        "importance": ("IMPORTANCE_MAX_OUTPUT_TOKENS", 512),
    }[stage]
    kwargs = dict(model=model_name, base_url=base_url, api_key=api_key,
                  timeout=240 if stage == "deep_read" else 120,
                  max_retries=0, max_tokens=positive_int(budget_name, budget))
    if model_name.lower().startswith("deepseek-"):
        thinking = enabled("DEEP_READ_THINKING" if stage == "deep_read" else "DEEPSEEK_THINKING")
        kwargs["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
        if thinking:
            kwargs["reasoning_effort"] = env_value("DEEPSEEK_REASONING_EFFORT", "low")
            if kwargs["reasoning_effort"] not in {"low", "high", "max"}:
                raise FatalAIError("DEEPSEEK_REASONING_EFFORT must be low, high or max")
        else:
            kwargs["temperature"] = 0
    elif model_name.lower().startswith("glm-5.3"):
        kwargs["reasoning_effort"] = env_value("GLM_REASONING_EFFORT", "low")
        if kwargs["reasoning_effort"] not in {"low", "high", "max"}:
            raise FatalAIError("GLM_REASONING_EFFORT must be low, high or max")
    else:
        kwargs["temperature"] = 0
        host = parsed.hostname
        if host == "volces.com" or host.endswith(".volces.com"):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return kwargs


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def metrics_snapshot():
    with _LOCK:
        return {key: dict(value) for key, value in _METRICS.items()}


def persist_metrics():
    output = env_value('AI_METRICS_PATH')
    if output:
        with _METRICS_WRITE_LOCK:
            atomic_write(output, json.dumps(metrics_snapshot(), ensure_ascii=False, indent=2) + '\n')


def metric(stage, **counts):
    with _LOCK:
        row = _METRICS.setdefault(stage, dict(calls=0, cache_hits=0, failures=0,
            input_tokens=0, output_tokens=0, unknown_usage=0))
        for key, value in counts.items():
            row[key] += value
    persist_metrics()


def record_usage(stage, raw):
    usage = getattr(raw, "usage_metadata", None) or {}
    legacy = (getattr(raw, "response_metadata", None) or {}).get("token_usage", {})
    input_tokens = usage.get("input_tokens", legacy.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", legacy.get("completion_tokens"))
    if input_tokens is None or output_tokens is None:
        metric(stage, unknown_usage=1)
    else:
        metric(stage, input_tokens=input_tokens, output_tokens=output_tokens)


def fatal_request(error):
    return isinstance(error, FatalAIError) or getattr(error, "status_code", None) in {400, 401, 402, 403, 404, 405, 422}


def transient_request(error):
    status = getattr(error, "status_code", None)
    return isinstance(error, (APIConnectionError, APITimeoutError, TimeoutError)) or status in {408, 409, 429} or (status is not None and status >= 500)


class CachedChain:
    def __init__(self, prompt, model_name, stage, schema=None, validator=None):
        self.prompt, self.stage, self.schema, self.validator = prompt, stage, schema, validator
        self.stopped = Event()
        settings = model_settings(model_name, stage)
        self.signature = {key: value for key, value in settings.items() if key != "api_key"}
        llm = ChatOpenAI(**settings)
        if schema:
            thinking = settings.get("extra_body", {}).get("thinking", {}).get("type") == "enabled"
            if thinking:
                prompt = ChatPromptTemplate.from_messages([
                    *prompt.messages,
                    ("system", "Return a JSON object matching this schema: {output_schema}"),
                ]).partial(output_schema=json.dumps(schema.model_json_schema(), ensure_ascii=False))
                self.prompt = prompt
            llm = llm.with_structured_output(schema, method="json_mode" if thinking else "function_calling", include_raw=True)
        self.chain = prompt | llm
        self.cache_root = Path(env_value("AI_CACHE_DIR", str(ROOT / "data" / "ai_cache")))
        metric(stage)

    def invoke(self, inputs):
        from progress import STOP
        if STOP.is_set():
            raise FatalAIError('Workflow cancelled before request')
        if self.stopped.is_set():
            raise FatalAIError("Stage stopped after a fatal API error")
        payload = dict(version=1, stage=self.stage, model=self.signature, paper_id=inputs.get("paper_id"),
                       schema=self.schema.model_json_schema() if self.schema else None,
                       prompt=self.prompt.format_prompt(**inputs).to_string())
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.cache_root / self.stage / f"{key}.json"
        try:
            if not inputs.get("_cache_allowed", True):
                raise ValueError("Fallback results are not cached")
            saved = json.loads(path.read_text(encoding="utf-8"))
            result = self.schema.model_validate(saved["result"]) if self.schema else AIMessage(content=saved["result"])
            if self.validator:
                self.validator(result)
            metric(self.stage, cache_hits=1)
            return result
        except (OSError, ValueError, KeyError, TypeError):
            pass
        for attempt in range(3):
            if self.stopped.is_set() or STOP.is_set():
                raise FatalAIError("Stage stopped after a fatal API error")
            metric(self.stage, calls=1)
            try:
                response = self.chain.invoke(inputs)
            except Exception as error:
                metric(self.stage, unknown_usage=1)
                if fatal_request(error):
                    self.stopped.set()
                    metric(self.stage, failures=1)
                    raise FatalAIError(f"{self.stage}: API configuration/authentication failed (HTTP {getattr(error, 'status_code', 'unknown')})") from error
                if transient_request(error) and attempt < 2:
                    self.stopped.wait(2 ** attempt)
                    continue
                metric(self.stage, failures=1)
                raise
            # Parsing and truncation failures are never retried automatically.
            raw = response.get("raw") if self.schema else response
            record_usage(self.stage, raw)
            try:
                if (getattr(raw, "response_metadata", None) or {}).get("finish_reason") == "length":
                    raise ValueError(f"{self.stage}: output token limit reached")
                if self.schema:
                    if response.get("parsing_error") or response.get("parsed") is None:
                        raise ValueError(f"{self.stage}: invalid structured output")
                    result = response["parsed"]
                    serialized = result.model_dump()
                else:
                    if not isinstance(raw.content, str) or not raw.content.strip():
                        raise ValueError(f"{self.stage}: empty text response")
                    result, serialized = raw, raw.content
                if self.validator:
                    self.validator(result)
            except Exception:
                metric(self.stage, failures=1)
                raise
            if inputs.get("_cache_allowed", True):
                atomic_write(path, json.dumps({"result": serialized}, ensure_ascii=False))
            return result


def flush_metrics():
    snapshot = metrics_snapshot()
    if not snapshot:
        return
    rows = ["### AI usage", "", "| Stage | Calls (including retries) | Cache hits | Failed items | Input tokens | Output tokens | Unknown usage calls |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: |"]
    for stage, row in sorted(snapshot.items()):
        def tokens(name):
            return f"{row[name]} + 未知" if row["unknown_usage"] else str(row[name])
        rows.append(f"| {stage} | {row['calls']} | {row['cache_hits']} | {row['failures']} | {tokens('input_tokens')} | {tokens('output_tokens')} | {row['unknown_usage']} |")
    report = "\n".join(rows) + "\n"
    print(report, file=sys.stderr)
    summary = env_value("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(report)
    output = env_value("AI_METRICS_PATH")
    if output:
        atomic_write(output, json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")


atexit.register(flush_metrics)
