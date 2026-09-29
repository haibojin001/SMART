"""Cost and latency accounting for a run, added from outside the core.

The paper reports, per episode: model calls, input tokens, output tokens, tool calls and
end-to-end runtime. A core module counts only `Claude.calls`, so the rest has to come from
somewhere. It deliberately does not come from editing the thirty core files.

    Why from outside. Each core module is byte-for-byte the system that produced the
    results - that is the property the build proves, and the release is only worth anything
    if it holds. Threading a token counter through `Claude._invoke`, `chat`, `tool_loop` and
    `ToolExecutor.execute` in thirty files would break it thirty times over, for a
    measurement that is not part of the method. So the accounting wraps the class at runtime
    instead: the files on disk stay exactly what was proven, and the numbers still get
    collected.

    What it can see. The Messages API returns `usage.{input_tokens, output_tokens}` on every
    invocation, so tokens are read rather than estimated. Latency is wall-clock around the
    HTTP call, which is the honest number for the paper's "under serial execution" runtime.

    How roles are attributed. A core module's system prompts are module-scope constants
    (`DEEP_RESEARCH_SYSTEM`, `JUDGE_SYSTEM`, ...) plus the router's per-agent prompts, so the
    system string a call was made with identifies the role that made it. `role_map()` builds
    that index once; anything unrecognised is attributed to `other` rather than guessed at,
    and `other` staying near zero is itself a check that the index is complete.

Usage:
    import instrument
    mod = directions.load("en2zh")
    usage = instrument.attach(mod)
    ...                                 # run the pipeline as normal
    usage.dump("out/usage_en2zh.json")
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

# List prices in USD per million tokens, matched by substring against the model id actually
# used. Cost is the one reported number that is not measured, so it is kept explicitly
# derived and overridable rather than baked into the token counts:
#     SMART_PRICE_IN=3.0 SMART_PRICE_OUT=15.0 python3 run_smart.py ...
# A model id that matches nothing gets no cost, and `priced` in the summary says so.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def price_for(model_id: str) -> tuple[float, float] | None:
    """($/Mtok in, $/Mtok out) for a model id, or None if it is not in the table."""
    env_in, env_out = os.environ.get("SMART_PRICE_IN"), os.environ.get("SMART_PRICE_OUT")
    if env_in and env_out:
        return float(env_in), float(env_out)
    # Longest key first: `claude-sonnet-4-6` must win over `claude-sonnet-4`.
    for key in sorted(PRICES, key=len, reverse=True):
        if key in model_id:
            return PRICES[key]
    return None


class Usage:
    """Everything one run spent, by role and in total."""

    def __init__(self, direction: str = "", model_id: str = ""):
        self.direction = direction
        self.model_id = model_id
        self.started = time.time()
        self.finished: float | None = None
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.api_seconds = 0.0
        self.throttles = 0
        self.by_role: dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0})
        self.tool_calls: dict[str, int] = defaultdict(int)
        self.tool_seconds: dict[str, float] = defaultdict(float)
        self._role = "other"

    # ---- recording ----

    @contextmanager
    def role(self, name: str):
        """Attribute everything invoked inside this block to `name`."""
        prev, self._role = self._role, name
        try:
            yield
        finally:
            self._role = prev

    def record_call(self, seconds: float, usage: dict, role: str | None = None) -> None:
        r = self.by_role[role or self._role]
        tin = int(usage.get("input_tokens", 0) or 0)
        tout = int(usage.get("output_tokens", 0) or 0)
        self.calls += 1
        self.input_tokens += tin
        self.output_tokens += tout
        self.api_seconds += seconds
        r["calls"] += 1
        r["input_tokens"] += tin
        r["output_tokens"] += tout
        r["seconds"] += seconds

    def record_tool(self, name: str, seconds: float) -> None:
        self.tool_calls[name] += 1
        self.tool_seconds[name] += seconds

    def finish(self) -> "Usage":
        self.finished = time.time()
        return self

    # ---- reporting ----

    @property
    def wall_seconds(self) -> float:
        return (self.finished or time.time()) - self.started

    def cost(self) -> float | None:
        p = price_for(self.model_id)
        if p is None:
            return None
        return self.input_tokens / 1e6 * p[0] + self.output_tokens / 1e6 * p[1]

    def summary(self) -> dict:
        c = self.cost()
        return {
            "direction": self.direction,
            "model_id": self.model_id,
            "model_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": sum(self.tool_calls.values()),
            "throttle_waits": self.throttles,
            "wall_seconds": round(self.wall_seconds, 2),
            "wall_minutes": round(self.wall_seconds / 60, 2),
            "api_seconds": round(self.api_seconds, 2),
            # The gap between api_seconds and wall_seconds is parsing, retry sleeps and the
            # pipeline's own 0.2s inter-group pause. Reported separately so a latency claim
            # can say which it means.
            "non_api_seconds": round(self.wall_seconds - self.api_seconds, 2),
            "usd": None if c is None else round(c, 4),
            "priced": c is not None,
            "by_role": {k: {**v, "seconds": round(v["seconds"], 2)}
                        for k, v in sorted(self.by_role.items(),
                                           key=lambda kv: -kv[1]["calls"])},
            "by_tool": {k: {"calls": self.tool_calls[k],
                            "seconds": round(self.tool_seconds[k], 2)}
                        for k in sorted(self.tool_calls, key=lambda t: -self.tool_calls[t])},
        }

    def dump(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    def __str__(self) -> str:
        s = self.summary()
        cost = "unpriced" if s["usd"] is None else f"${s['usd']:.2f}"
        return (f"{s['model_calls']} calls, {s['input_tokens']:,} in / "
                f"{s['output_tokens']:,} out tokens, {s['tool_calls']} tool calls, "
                f"{s['wall_minutes']:.1f} min, {cost}")


# --------------------------------------------------------------------------------
# attaching to a core module
# --------------------------------------------------------------------------------

# Module-scope system prompts, mapped to the role that uses them. Names, not values, because
# the values differ per direction - that is the whole point of thirty files.
_ROLE_CONSTANTS = {
    "DEEP_RESEARCH_SYSTEM": "research",
    "IDIOM_BANK_SYSTEM": "idiom_bank",
    "SCENE_SEGMENT_SYSTEM": "scene_segment",
    "DOMAIN_TERM_IDENTIFIER_SYSTEM": "domain_terms",
    "JUDGE_SYSTEM": "judge",
    "REFINER_SYSTEM": "refiner",
    "DOC_REFINER_SYSTEM": "doc_refine",
}


class _ThrottleCounter(logging.Handler):
    """Counts the core's `ThrottlingException, waiting Ns` warnings into a `Usage`."""

    def __init__(self, usage: "Usage"):
        super().__init__(level=logging.WARNING)
        self.usage = usage

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if "ThrottlingException" in record.getMessage():
                self.usage.throttles += 1
        except Exception:      # a logging handler must never break the run it is watching
            pass


def role_map(mod) -> dict[str, str]:
    """{system prompt text -> role name} for one core module.

    Agent prompts are templates with `{duration}`/`{max_chars}` in them and get `.format()`ed
    before use, so the stored text is not what arrives at `_invoke`. Only the stable prefix is
    indexed for those, and lookup falls back to prefix matching.
    """
    out: dict[str, str] = {}
    for const, role in _ROLE_CONSTANTS.items():
        text = getattr(mod, const, None)
        if isinstance(text, str):
            out[text] = role
    for name, cfg in getattr(mod, "DEFAULT_AGENTS", {}).items():
        prompt = cfg.get("prompt", "")
        if prompt:
            out[prompt] = f"agent:{name}"
    return out


def _prefix_index(roles: dict[str, str], n: int = 120) -> dict[str, str]:
    """Role lookup keyed on the first `n` characters, for prompts that get formatted."""
    out: dict[str, str] = {}
    for text, role in roles.items():
        key = text[:n]
        # A prefix shared by two roles is useless as a discriminator; drop it rather than
        # let it silently attribute one role's calls to another.
        out[key] = role if out.get(key, role) == role else ""
    return {k: v for k, v in out.items() if v}


def attach(mod, direction: str = "", extra_roles: dict[str, str] | None = None) -> Usage:
    """Wrap `mod`'s Claude client and tool executor so a run accounts for itself.

    Idempotent: attaching twice returns the same `Usage` and does not double-count, which
    matters because `self_evolve` loads a direction once and runs many epochs through it.
    """
    existing = getattr(mod, "_SMART_USAGE", None)
    if existing is not None:
        return existing

    usage = Usage(direction or getattr(mod, "__name__", ""), getattr(mod, "MODEL_ID", ""))
    roles = role_map(mod)
    roles.update(extra_roles or {})
    prefixes = _prefix_index(roles)

    claude_cls = mod.Claude
    raw_invoke = claude_cls._invoke
    raw_chat = claude_cls.chat
    raw_tool_loop = claude_cls.tool_loop

    def _role_for(system: str) -> str:
        if system in roles:
            return roles[system]
        return prefixes.get(system[:120], "other")

    def _invoke(self, body: dict) -> dict:
        # The system string is on the body, so a call can be attributed even when it did not
        # come through chat/tool_loop.
        role = _role_for(body.get("system", "")) if "system" in body else usage._role
        t0 = time.perf_counter()
        result = raw_invoke(self, body)
        # Includes the core's own throttle backoff, because `_invoke` retries internally and
        # does not report having done so. `throttle_waits` is what makes that visible: a run
        # with a high api_seconds and zero waits was genuinely slow, one with waits was queued.
        usage.record_call(time.perf_counter() - t0, result.get("usage", {}), role)
        return result

    def chat(self, system: str, user_msg: str, **kw):
        with usage.role(_role_for(system)):
            return raw_chat(self, system, user_msg, **kw)

    def tool_loop(self, system: str, user_msg: str, tools, executor, **kw):
        with usage.role(_role_for(system)):
            return raw_tool_loop(self, system, user_msg, tools, executor, **kw)

    claude_cls._invoke = _invoke
    claude_cls.chat = chat
    claude_cls.tool_loop = tool_loop

    exec_cls = mod.ToolExecutor
    raw_execute = exec_cls.execute

    def execute(self, tool_name: str, tool_input: dict) -> dict:
        t0 = time.perf_counter()
        try:
            return raw_execute(self, tool_name, tool_input)
        finally:
            usage.record_tool(tool_name, time.perf_counter() - t0)

    exec_cls.execute = execute

    # Throttling is only visible in the core's own log line - `_invoke` retries internally and
    # `calls` advances once per *success*, so counting calls cannot see a wait. A handler on
    # the module's logger can. This is the least invasive place to read it from.
    handler = _ThrottleCounter(usage)
    mod.logger.addHandler(handler)

    mod._SMART_USAGE = usage
    mod._SMART_UNPATCH = lambda: (
        setattr(claude_cls, "_invoke", raw_invoke),
        setattr(claude_cls, "chat", raw_chat),
        setattr(claude_cls, "tool_loop", raw_tool_loop),
        setattr(exec_cls, "execute", raw_execute),
        mod.logger.removeHandler(handler),
        delattr(mod, "_SMART_USAGE"),
    )
    return usage


def detach(mod) -> None:
    """Undo `attach`. Only needed by tests; a normal run exits instead."""
    undo = getattr(mod, "_SMART_UNPATCH", None)
    if undo:
        undo()
        del mod._SMART_UNPATCH


def merge(usages: list[Usage]) -> dict:
    """Totals across many runs, plus the per-episode means the cost table reports."""
    live = [u for u in usages if u.calls]
    if not live:
        return {"runs": 0}
    keys = ("model_calls", "input_tokens", "output_tokens", "tool_calls")
    sums = [u.summary() for u in live]
    total = {k: sum(s[k] for s in sums) for k in keys}
    total["wall_seconds"] = sum(s["wall_seconds"] for s in sums)
    costed = [s["usd"] for s in sums if s["usd"] is not None]
    n = len(live)
    return {
        "runs": n,
        "total": {**total, "usd": round(sum(costed), 4) if costed else None,
                  "priced_runs": len(costed)},
        "per_run_mean": {
            **{k: round(total[k] / n, 1) for k in keys},
            "wall_minutes": round(total["wall_seconds"] / 60 / n, 2),
            "usd": round(sum(costed) / len(costed), 4) if costed else None,
        },
    }
