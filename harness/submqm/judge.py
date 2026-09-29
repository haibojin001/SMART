"""The SubMQM evaluator: one judge call per window, nineteen penalties out.

This is the measurement instrument, and it is kept strictly separate from the judge that lives
*inside* SMART. The two are easy to confuse and must not share code:

    SMART's internal judge is a system component. It scores candidate translations 1-10 on its
    own criteria so the refiner knows what to fix, and it runs during translation. It is part
    of what is being measured.

    The SubMQM judge is the evaluator. It scores nineteen error types in {0, 5, 10} against the
    published rubric, it runs after translation, and the same rubric and the same model are
    applied to every system - SMART, the baselines and the references alike. Sharing prompts or
    thresholds between the two would make the system its own examiner.

Three things the rubric needs that the rubric module cannot know:

    The display budget. Two of the nineteen types are "exceeds the predefined
    character-per-line constraint" and "exceeds the predefined maximum number of lines", so the
    judge has to be told what the constraint is - and it differs per direction, 20 characters a
    line into Chinese against 42 into German. `budget_for()` reads it out of the direction's own
    core module, so the number the judge enforces is the number the system was built to.

    The languages. Language Detection Error is meaningless without knowing what the expected
    source and target languages are.

    Which judge model. The paper reports an evaluator-robustness check that swaps the default
    Claude Sonnet 4.6 for GPT-5.5 with the translation system held fixed, so the backend is
    pluggable and its identity is recorded in every result file. A number produced under a
    different judge is not comparable, and the only thing that makes that visible after the
    fact is having written it down.

Judging is cached on disk by a hash of (backend, model, system, user). A re-run of an episode
that has already been judged costs nothing, which is what makes it practical to re-score after
fixing an aggregation bug rather than being tempted not to.

    python3 judge.py --self-test                       # no credentials needed
    python3 judge.py source.srt hyp.srt --direction en2zh -o out/scores.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__:
    from . import align, rubric
else:                                    # run directly: python3 judge.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import align
    import rubric

logger = logging.getLogger("submqm.judge")

# The judge model the paper's main tables use. Overridable, and whatever is used is recorded.
DEFAULT_JUDGE_MODEL = os.environ.get("SUBMQM_JUDGE_MODEL", "claude-sonnet-4-6")
DEFAULT_PROVIDER = os.environ.get("ANTHROPIC_PROVIDER", "anthropic").lower()

# Lines per displayed subtitle box. The paper names the constraint but never its value, and
# unlike characters-per-line there is no module constant for it either - the cores enforce it
# structurally, in `wrap_subtitle`, which wraps an over-long translation onto *two* lines and
# never onto three. So two is what the system was built to and two is what the judge is told.
MAX_LINES_PER_BOX = 2


# --------------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an expert subtitle quality evaluator applying a subtitle-adapted \
Multidimensional Quality Metrics (MQM) protocol.

You will be shown a window of consecutive subtitles from one episode: the {src} source and a \
{tgt} translation of it. Assign, for each of the {n_types} error types below, a single discrete \
penalty for the window as a whole:

  0  = no error of this type in this window, OR the type does not apply here
  5  = a minor issue that does not impede comprehension
  10 = a severe error that changes meaning, misleads the viewer, or violates a hard subtitle \
constraint

Every score is a PENALTY, so 0 is the best possible score and 10 the worst. Judge the window as \
a unit: one penalty per error type, not one per subtitle.

Score the window against these {n_types} error types, grouped by dimension:

{rubric}

Subtitle display constraints for this direction, which the technical error types refer to:
  - maximum {max_line} {unit} per line
  - maximum {max_lines} lines per subtitle box
  - reading-speed budget {max_cps} {unit} per second of display time

Expected source language: {src}. Expected target language: {tgt}. Language Detection Error \
applies when text is in neither the expected language nor a language the scene plausibly calls \
for.

{mode_note}

Return ONLY a JSON object with exactly these {n_types} keys and an integer 0, 5 or 10 for each. \
No commentary, no markdown fence, no extra keys:

{skeleton}"""

BLOCK_NOTE = """The translation preserves the source segmentation, so the window is shown as \
numbered source/target pairs. Judge each pair against its own source, and judge the window's \
coherence across pairs."""

PASSAGE_NOTE = """This target was segmented independently of the source, so it is shown as a \
passage covering the same span of time rather than block by block. Judge Accuracy, Terminology, \
Fluency, Locale and Audience against the passage as a whole: content that appears anywhere in \
the target passage is NOT an omission, and content moved between adjacent subtitles is NOT a \
mistranslation. Judge the technical error types - line length and lines per box - against the \
target cues as displayed, which are listed separately."""


def judge_system_prompt(src: str, tgt: str, max_line: int, max_cps: int, mode: str,
                        max_lines: int = MAX_LINES_PER_BOX, unit: str = "characters") -> str:
    """The evaluator's system prompt: one rubric, parameterised by direction and mode.

    Built from `rubric` rather than written out, so an error type cannot exist in the scorer and
    be missing from the prompt. That failure is silent and flattering: the judge never scores
    the type, `normalise_window` defaults it to 0, and the column reads as perfect.
    """
    return JUDGE_SYSTEM.format(
        src=src, tgt=tgt, n_types=len(rubric.ALL_TYPES), rubric=rubric.rubric_text(),
        max_line=max_line, max_lines=max_lines, max_cps=max_cps, unit=unit,
        mode_note=BLOCK_NOTE if mode == "block" else PASSAGE_NOTE,
        skeleton=rubric.json_skeleton())


def user_message(window: "align.WindowSpec", total: int) -> str:
    return (f"Episode window {window.idx} of {total}, source subtitles "
            f"{window.first}-{window.last}.\n\n{window.render()}\n\n"
            f"Return the JSON object now.")


def budget_for(direction: str) -> dict:
    """The direction's display budget, read from its own core module.

    Falls back to the Latin defaults with a warning rather than failing, so an evaluation of a
    file that has no matching core module - a baseline system's output, a reference - still
    runs. The fallback is reported in the result file, because a CJK target judged against a
    42-character line budget would under-report CharLim across the board.
    """
    unit = "characters"
    try:
        harness = Path(__file__).resolve().parent.parent
        if str(harness) not in sys.path:
            sys.path.insert(0, str(harness))
        import directions
        d = directions.resolve(direction)
        c = directions.constraints(d)
        tgt_key = d.tgt
        # Into CJK the cores count CJK characters, not code points; saying so keeps the judge
        # from measuring a 20-character Chinese line as if it were 20 Latin characters.
        if tgt_key in ("zh_CN", "ko_KR"):
            unit = "characters (counted as CJK/Hangul characters)"
        return {
            "max_line": c.get("MAX_LINE", 42), "max_cps": c.get("MAX_CPS", 17),
            "max_lines": MAX_LINES_PER_BOX, "unit": unit,
            "source_language": directions.LOCALES[d.src].qualified,
            "target_language": directions.LOCALES[d.tgt].qualified,
            "from_module": d.filename,
        }
    except Exception as exc:
        logger.warning("no core module for %r (%s); using the Latin default budget",
                       direction, exc)
        return {"max_line": 42, "max_cps": 17, "max_lines": MAX_LINES_PER_BOX, "unit": unit,
                "source_language": "the source language",
                "target_language": "the target language", "from_module": None}


# --------------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------------

class Backend:
    """A judge model. `complete` returns the model's text for one (system, user) pair."""

    name = "abstract"
    model = ""

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.model}


class AnthropicBackend(Backend):
    """Claude through the Messages API - the default evaluator.

    `provider` selects the endpoint: the direct API, Bedrock or Vertex. No credential is passed
    as an argument; each client reads the environment its own SDK documents, so pointing an
    evaluation at a different account from the translation run is done by changing the
    environment rather than by threading a key through this constructor.

    The SDK is imported here rather than at module scope on purpose: `--self-test` runs the whole
    scoring chain against the stub backend, and it has to work on a machine where no model client
    is installed at all.
    """

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, provider: str = DEFAULT_PROVIDER,
                 max_tokens: int = 2000):
        import anthropic
        self._sdk = anthropic
        self.model = model
        self.provider = provider
        self.max_tokens = max_tokens
        if provider == "bedrock":
            self._client = anthropic.AnthropicBedrock(max_retries=0)
        elif provider == "vertex":
            self._client = anthropic.AnthropicVertex(max_retries=0)
        elif provider == "anthropic":
            self._client = anthropic.Anthropic(max_retries=0)
        else:
            raise SystemExit(f"ANTHROPIC_PROVIDER={provider!r}: expected "
                             f"anthropic, bedrock or vertex")

    # Same set the cores retry on, and for the same reason: a 400 or a 401 will not improve.
    RETRY_STATUS = (429, 500, 502, 503, 504, 529)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        last: Exception | None = None
        for attempt in range(6):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                # exclude_none for the same reason the cores use it: it yields the shape that came
                # off the wire, rather than every optional field of the response model spelled out
                # as an explicit null.
                payload = resp.model_dump(exclude_none=True)
                return "".join(b.get("text", "") for b in payload.get("content", []))
            except self._sdk.APIStatusError as exc:
                last = exc
                if exc.status_code not in self.RETRY_STATUS:
                    raise
                wait = 2 ** attempt
                logger.warning("judge got %s, waiting %ds", exc.status_code, wait)
                time.sleep(wait)
            except self._sdk.APIConnectionError as exc:
                last = exc
                wait = 2 ** attempt
                logger.warning("judge connection error, waiting %ds", wait)
                time.sleep(wait)
        raise RuntimeError(f"judge gave up after 6 attempts: {last}")

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.model, "provider": self.provider}


class OpenAIBackend(Backend):
    """An OpenAI-protocol judge - the evaluator-robustness swap (GPT-5.5 in the paper).

    Kept behind the same one-method interface as Bedrock so the swap really does change only
    the judge model. `OPENAI_BASE_URL` is honoured, so this also serves any OpenAI-compatible
    gateway without a second backend class.
    """

    name = "openai"

    def __init__(self, model: str = "gpt-5.5", max_tokens: int = 2000):
        from openai import OpenAI
        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                              base_url=os.environ.get("OPENAI_BASE_URL") or None)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return resp.choices[0].message.content or ""


class StubBackend(Backend):
    """A deterministic backend for testing the parser and the arithmetic with no credentials.

    Returns all-zero windows unless `scores` is given. Exists so `--self-test` can exercise
    the whole path - prompt construction, parsing, validation, aggregation - on a machine that
    cannot reach a model, which is where most of this code's bugs would otherwise hide.
    """

    name = "stub"

    def __init__(self, scores: dict | None = None, model: str = "stub"):
        self.model = model
        self.scores = scores or {}
        self.calls = 0

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls += 1
        return json.dumps({t: self.scores.get(t, 0) for t in rubric.ALL_TYPES})


def make_backend(name: str, model: str | None = None) -> Backend:
    if name in ("anthropic", "bedrock", "vertex"):
        # "bedrock" and "vertex" are accepted as backend names, not just as providers, because
        # job files written before the backends were unified name them here. They select the same
        # class with the matching endpoint, so an old job file still describes the same run.
        if name == "anthropic":
            return AnthropicBackend(model or DEFAULT_JUDGE_MODEL)
        return AnthropicBackend(model or DEFAULT_JUDGE_MODEL, provider=name)
    if name == "openai":
        return OpenAIBackend(model or "gpt-5.5")
    if name == "stub":
        return StubBackend(model=model or "stub")
    raise SystemExit(f"unknown judge backend {name!r} (anthropic, bedrock, vertex, openai, stub)")


# --------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------

class Cache:
    """Content-addressed disk cache of judge responses.

    Keyed on the exact (backend, model, system, user) text, so changing the rubric, the budget
    or the window geometry invalidates it automatically - there is no version number to forget
    to bump. One file per response rather than one big index, so two evaluations running in
    parallel over different directions cannot corrupt each other's cache.
    """

    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        self.hits = 0
        self.misses = 0
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def key(self, backend: Backend, system: str, user: str) -> str:
        h = hashlib.sha256()
        for part in (backend.name, backend.model, system, user):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:32]

    def get(self, key: str) -> str | None:
        if not self.root:
            return None
        p = self.root / f"{key}.txt"
        if p.exists():
            self.hits += 1
            return p.read_text(encoding="utf-8")
        self.misses += 1
        return None

    def put(self, key: str, value: str) -> None:
        if self.root:
            (self.root / f"{key}.txt").write_text(value, encoding="utf-8")


# --------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict:
    """The JSON object out of a model response, tolerating fences and surrounding prose.

    Scans for a brace-balanced object rather than using a greedy `\\{.*\\}`, because a response
    that appends a sentence containing a brace would otherwise fail to parse and cost a retry.
    """
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start < 0:
        raise ValueError("no JSON object in the response")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start:i + 1])
    raise ValueError("unbalanced JSON object in the response")


def parse_window_response(text: str) -> rubric.Window:
    """One judge response as a validated window of nineteen penalties."""
    raw = extract_json(text)
    if not isinstance(raw, dict):
        raise ValueError(f"judge returned {type(raw).__name__}, not an object")
    # Some models answer `{"scores": {...}}` however the skeleton is phrased. Unwrap a single
    # nested object rather than failing: the retry is a whole model call, and this is free.
    if len(raw) == 1 and isinstance(next(iter(raw.values())), dict):
        raw = next(iter(raw.values()))
    return rubric.normalise_window(raw)


# --------------------------------------------------------------------------------
# judging
# --------------------------------------------------------------------------------

@dataclass
class WindowScore:
    window: int
    first_cue: int
    last_cue: int
    scores: rubric.Window
    attempts: int = 1
    cached: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        d = {"window": self.window, "first_cue": self.first_cue, "last_cue": self.last_cue,
             "attempts": self.attempts, "cached": self.cached, "scores": self.scores}
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class JudgeRun:
    """One (system, direction, episode) judged: the windows, and how the judging itself went."""
    windows: list[WindowScore] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    backend: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)
    seconds: float = 0.0

    def scored_windows(self) -> list[rubric.Window]:
        return [w.scores for w in self.windows if not w.error]


def judge_windows(windows: list["align.WindowSpec"], backend: Backend, budget: dict,
                  mode: str, *, cache: Cache | None = None, retries: int = 2,
                  temperature: float = 0.0, raw_dir: Path | None = None,
                  progress_every: int = 10) -> JudgeRun:
    """Judge every window. A window the judge cannot be made to score is dropped, not zeroed.

    Dropping is the only honest option. A window recorded as all-zero is indistinguishable from
    a perfect window, so a judge that failed on the hardest 5% of an episode would *improve* the
    score. `failed` records every one, and `evaluate.py` refuses to publish a run whose failure
    rate is above a threshold.
    """
    system = judge_system_prompt(
        budget["source_language"], budget["target_language"], budget["max_line"],
        budget["max_cps"], mode, budget.get("max_lines", MAX_LINES_PER_BOX),
        budget.get("unit", "characters"))
    run = JudgeRun(backend=backend.describe(), budget=dict(budget))
    started = time.time()
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)

    for w in windows:
        user = user_message(w, len(windows))
        key = cache.key(backend, system, user) if cache else None
        text = cache.get(key) if cache else None
        from_cache = text is not None
        last_error = ""
        scored: WindowScore | None = None

        for attempt in range(1, retries + 2):
            if text is None:
                try:
                    text = backend.complete(system, user, temperature)
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.warning("window %d attempt %d: %s", w.idx, attempt, last_error)
                    text = None
                    continue
            try:
                scores = parse_window_response(text)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("window %d attempt %d: unparseable response: %s",
                               w.idx, attempt, last_error)
                if raw_dir:
                    (raw_dir / f"w{w.idx:04d}_attempt{attempt}_bad.txt").write_text(
                        text, encoding="utf-8")
                # A cached response that will not parse is a poisoned cache entry, not a model
                # failure; drop it so the retry actually re-asks.
                text, from_cache = None, False
                continue
            if cache and key and not from_cache:
                cache.put(key, text)
            if raw_dir:
                (raw_dir / f"w{w.idx:04d}.txt").write_text(text, encoding="utf-8")
            scored = WindowScore(w.idx, w.first, w.last, scores, attempt, from_cache)
            break

        if scored is None:
            run.failed.append({"window": w.idx, "first_cue": w.first, "last_cue": w.last,
                               "error": last_error or "no response"})
            continue
        run.windows.append(scored)
        if progress_every and len(run.windows) % progress_every == 0:
            logger.info("  judged %d/%d windows", len(run.windows), len(windows))

    run.seconds = round(time.time() - started, 1)
    if cache:
        run.cache = {"hits": cache.hits, "misses": cache.misses,
                     "dir": str(cache.root) if cache.root else None}
    return run


def judge_episode(source: str | Path, target: str | Path, *, direction: str,
                  backend: Backend, mode: str = "block", window: int = align.DEFAULT_WINDOW,
                  overlap: float = align.DEFAULT_OVERLAP, normalise: bool | None = None,
                  cache_dir: str | Path | None = None, raw_dir: str | Path | None = None,
                  retries: int = 2, temperature: float = 0.0) -> dict:
    """Align, judge and score one (system, direction, episode). Returns the full record."""
    windows, report = align.prepare(source, target, mode=mode, window=window, overlap=overlap,
                                    normalise=normalise)
    budget = budget_for(direction)
    run = judge_windows(windows, backend, budget, mode,
                        cache=Cache(cache_dir) if cache_dir else None, retries=retries,
                        temperature=temperature,
                        raw_dir=Path(raw_dir) if raw_dir else None)
    scored = run.scored_windows()
    return {
        "direction": direction,
        "source": str(source),
        "target": str(target),
        "alignment": report,
        "budget": run.budget,
        "judge": run.backend,
        "temperature": temperature,
        "windows_total": len(windows),
        "windows_scored": len(scored),
        "windows_failed": len(run.failed),
        "failed": run.failed,
        "cache": run.cache,
        "seconds": run.seconds,
        "score": rubric.score_episode(scored),
        "per_window": [w.to_dict() for w in run.windows],
    }


# --------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------

def self_test() -> int:
    """Exercise prompt, parser and arithmetic with no credentials and no model."""
    failures: list[str] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        if not cond:
            failures.append(f"{label}{': ' + detail if detail else ''}")

    # 1. The prompt carries all nineteen types and the direction's real budget.
    sysmsg = judge_system_prompt("English", "Mandarin Chinese", 20, 15, "block")
    for t in rubric.ALL_TYPES:
        check(f"prompt is missing {t}", t in sysmsg)
    check("prompt lost the line budget", "maximum 20 characters per line" in sysmsg)
    check("prompt lost the lines-per-box budget", "maximum 2 lines" in sysmsg)
    check("prompt lost the reading-speed budget", "15 characters per second" in sysmsg)
    check("block mode got the passage note", "independently of the source" not in sysmsg)
    passage = judge_system_prompt("English", "Mandarin Chinese", 20, 15, "passage")
    check("passage mode lost its note", "NOT an omission" in passage)

    # 2. The parser survives the ways a model actually answers.
    good = json.dumps({t: 0 for t in rubric.ALL_TYPES})
    variants = {
        "bare": good,
        "fenced": f"```json\n{good}\n```",
        "prose before": f"Here is my assessment:\n{good}",
        "prose after": f"{good}\nOverall this window is clean.",
        "nested": json.dumps({"scores": json.loads(good)}),
        "brace in trailing prose": f"{good}\nNote: the set {{a, b}} was fine.",
        "partial keys": json.dumps({"mistranslation": 5}),
        "float scores": json.dumps({**json.loads(good), "grammar": 5.0}),
    }
    for label, text in variants.items():
        try:
            w = parse_window_response(text)
            check(f"parser dropped keys on {label}", len(w) == 19, str(len(w)))
        except Exception as exc:
            failures.append(f"parser failed on {label}: {type(exc).__name__}: {exc}")
    for label, text in {"out of domain": json.dumps({"grammar": 7}),
                        "unknown key": json.dumps({"vibes": 0}),
                        "not an object": "[0, 0, 0]",
                        "no json": "I cannot evaluate this."}.items():
        try:
            parse_window_response(text)
            failures.append(f"parser accepted {label}, which it must reject")
        except Exception:
            pass

    # 3. The whole path, on synthetic files, with a stub judge.
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    cues = "\n\n".join(
        f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},900\nline {i} of the source"
        for i in range(1, 13))
    (tmp / "src.srt").write_text(cues + "\n", encoding="utf-8")
    (tmp / "hyp.srt").write_text(cues.replace("source", "target") + "\n", encoding="utf-8")
    stub = StubBackend({"mistranslation": 10, "grammar": 5})
    rec = judge_episode(tmp / "src.srt", tmp / "hyp.srt", direction="en2zh", backend=stub,
                        mode="block", window=5, cache_dir=tmp / "cache")
    check("wrong window count", rec["windows_total"] == 3, str(rec["windows_total"]))
    check("windows not all scored", rec["windows_failed"] == 0)
    check("stub not called once per window", stub.calls == 3, str(stub.calls))
    # accuracy = mean(10, 0, 0) = 3.333; overall = .30*3.333 + .08*(5/4)
    acc = rec["score"]["dimensions"]["accuracy"]
    ling = rec["score"]["dimensions"]["linguistic_conventions"]
    check("accuracy arithmetic", abs(acc - 10 / 3) < 1e-9, f"{acc}")
    check("linguistic arithmetic", abs(ling - 1.25) < 1e-9, f"{ling}")
    want = 0.30 * (10 / 3) + 0.08 * 1.25
    check("overall arithmetic", abs(rec["score"]["overall"] - want) < 1e-9,
          f"{rec['score']['overall']} != {want}")
    check("budget not read from the core module",
          rec["budget"]["max_line"] == 20 and rec["budget"]["max_cps"] == 15,
          json.dumps(rec["budget"]))
    check("target language not resolved",
          rec["budget"]["target_language"] == "Mandarin Chinese")

    # 4. The cache really caches: a second run must not call the model again.
    stub2 = StubBackend({"mistranslation": 10, "grammar": 5})
    rec2 = judge_episode(tmp / "src.srt", tmp / "hyp.srt", direction="en2zh", backend=stub2,
                         mode="block", window=5, cache_dir=tmp / "cache")
    check("cache did not hit", stub2.calls == 0, f"{stub2.calls} model calls on a cached run")
    check("cached run scored differently",
          rec2["score"]["overall"] == rec["score"]["overall"])
    check("cache stats not recorded", rec2["cache"].get("hits") == 3,
          json.dumps(rec2["cache"]))
    # A cache keyed on the prompt must miss when the budget changes.
    stub3 = StubBackend()
    judge_episode(tmp / "src.srt", tmp / "hyp.srt", direction="en2de", backend=stub3,
                  mode="block", window=5, cache_dir=tmp / "cache")
    check("cache ignored the direction's budget", stub3.calls == 3, f"{stub3.calls}")

    # 5. A judge that always fails must drop windows, never zero them.
    class Broken(Backend):
        name, model = "broken", "broken"

        def complete(self, system, user, temperature=0.0):
            raise RuntimeError("no")

    bad = judge_episode(tmp / "src.srt", tmp / "hyp.srt", direction="en2zh", backend=Broken(),
                        mode="block", window=5)
    check("failed windows were scored", bad["windows_scored"] == 0)
    check("failures not recorded", bad["windows_failed"] == 3)
    check("a fully failed run scored 0.0 instead of nothing",
          bad["score"]["windows"] == 0 and bad["score"]["overall"] == 0.0)

    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"PASSED: prompt carries {len(rubric.ALL_TYPES)} error types; parser handles "
          f"{len(variants)} response shapes and rejects 4 bad ones; end-to-end arithmetic, "
          f"per-direction budget, cache and failure handling all check out")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Judge one episode against the SubMQM rubric.")
    ap.add_argument("source", nargs="?")
    ap.add_argument("target", nargs="?")
    ap.add_argument("--direction", default="en2zh")
    ap.add_argument("--mode", choices=("block", "passage"), default="block")
    ap.add_argument("--judge", choices=("anthropic", "bedrock", "vertex", "openai", "stub"),
                    default="anthropic")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--window", type=int, default=align.DEFAULT_WINDOW)
    ap.add_argument("--overlap", type=float, default=align.DEFAULT_OVERLAP)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.self_test:
        return self_test()
    if not args.source or not args.target:
        ap.error("source and target are required (or use --self-test)")

    rec = judge_episode(args.source, args.target, direction=args.direction,
                        backend=make_backend(args.judge, args.judge_model), mode=args.mode,
                        window=args.window, overlap=args.overlap, cache_dir=args.cache_dir,
                        raw_dir=args.raw_dir, temperature=args.temperature)
    text = json.dumps(rec, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"{rec['windows_scored']}/{rec['windows_total']} windows scored, "
              f"Overall {rec['score']['overall']:.2f} -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
