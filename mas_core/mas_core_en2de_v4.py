"""
MAS Core v4 — Multi-Agent Subtitle Translation (en→de-DE, SRT format)

Pipeline
  Phase -1  ContentProfile deep research (conservative: high-confidence facts only)
  Phase 0   Scene segmentation, then domain_notes enriched by web_search
  Phase 1   Mixture-of-agents translation with tool use, judged and refined
  Phase 2   Document-level consistency refinement over overlapping windows
  Memory    SeriesMemory: cross-episode terminology, characters and routing policy

Locale bindings, English → German
  Display budget   MAX_CPS = 17 chars/second, MAX_LINE = 42 chars/line
  Length model     counts characters and wraps at clause, then word, then hyphen
  Leak recovery    no charset test is possible; keep the last non-commentary line
  Leak detection   English prose in a German subtitle is agent commentary
  web_search       resolves English slang, culture-bound references and idioms
  idiom_lookup     proposes equivalent German fixed expressions
  fluency_check    catches English-calqued German and register drift
  Research profile each character carries its German name under the "de" key
  Series memory    series_memory_en2de.json

One file per direction, by design: the prompts, the display budget, the leak-recovery
strategy and the length model are all locale-dependent, and a single parameterised core
would hide those differences behind branches instead of making them reviewable.
"""
from __future__ import annotations


import json
import re
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path
from collections import defaultdict

import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Config
# ============================================================

MODEL_ID = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Which Anthropic-compatible endpoint to talk to: "anthropic" for the direct API, or "bedrock" or
# "vertex" for sites that can only reach Claude through a cloud provider. Only the transport
# differs - the request and response shapes below are the Messages API in all three cases, which
# is why one client class covers them and why no result in the paper depends on the choice.
PROVIDER = os.environ.get("ANTHROPIC_PROVIDER", "anthropic").lower()

MAX_TOKENS = 20000   # required by the Messages API, cannot be omitted; raised to avoid truncating long subtitles
SLIDING_WINDOW = 5
MAX_CPS = 17
MAX_LINE = 42

RESEARCH_SAMPLE_SIZE = 40

# --- Guardrails for the Phase 2 (doc_refine) sliding-window rewrite ---
# The windows overlap (stride = window // 2) and several passes are run, so the same
# subtitle is visited repeatedly. Without guardrails it never backs out of a bad rewrite,
# oscillates, rewrites past the duration limit anyway, and leaves no trace.
DOC_REFINE_MAX_REWRITES = 2         # max rewrites allowed for one subtitle across all of Phase 2
DOC_REFINE_MIN_COVERAGE = 0.6       # min fraction of a window's lines that must parse, else discard the window
DOC_REFINE_MAX_LOCKED_TERMS = 120   # cap on locked (must-match) terminology entries injected into the prompt
DOC_REFINE_MAX_REFERENCE_TERMS = 40 # cap on reference-only terminology entries injected into the prompt
# Phase 2's job is consistency and idiomaticity, not expansion. Phase 1 output has already
# been scored by the judge and passed constraint_check, so a significant length increase is
# itself a warning sign — short lines (<1.5s) take the meaning-first exemption, so an absolute CPS
# bound will not stop one being rewritten from 9 characters to 17.
DOC_REFINE_MAX_GROWTH_RATIO = 1.3   # upper bound on (length after rewrite) / (length before rewrite)
DOC_REFINE_GROWTH_GRACE_CHARS = 4   # growth of at most this many characters is exempt from the ratio check

# ============================================================
# Creative Translation Enhancement (configurable)
# ============================================================
ENABLE_CREATIVE_MODE = True  # enable creative translation (sound-alike puns, cultural transposition, dialect)
CREATIVITY_WEIGHT = 0.15     # weight of the creativity dimension in scoring (0-1)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Segment:
    index: int
    start: str
    end: str
    text: str
    speaker: str = ""
    scene_id: int = 0
    scene_theme: str = ""

    @property
    def duration(self) -> float:
        return self._sec(self.end) - self._sec(self.start)

    @staticmethod
    def _sec(t: str) -> float:
        parts = t.replace(",", ".").split(":")
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s


@dataclass
class Scene:
    id: int
    start_idx: int
    end_idx: int
    theme: str
    description: str
    tone: str
    characters: list = field(default_factory=list)
    domain_notes: str = ""


@dataclass
class ContentProfile:
    """Results from Phase -1 deep research (conservative)."""
    title: str = ""
    media_type: str = ""
    genre: list = field(default_factory=list)
    setting: str = ""
    synopsis: str = ""
    characters: list = field(default_factory=list)   # {name, de, role}; high-confidence only
    domain_knowledge: list = field(default_factory=list)
    terminology: dict = field(default_factory=dict)  # pre-seeded {en: de}
    idiom_bank: dict = field(default_factory=dict)   # German idioms, built per episode
    raw: dict = field(default_factory=dict)


@dataclass
class Context:
    segment: Segment
    preceding_src: list
    succeeding_src: list
    preceding_tgt: list
    scene: Optional[Scene] = None
    profile: Optional[ContentProfile] = None
    features: dict = field(default_factory=dict)


@dataclass
class Candidate:
    text: str
    agent: str
    tools_called: list = field(default_factory=list)
    score: float = 0.0
    detail: dict = field(default_factory=dict)


@dataclass
class Result:
    segment: Segment
    source: str
    translation: str
    agent: str
    score: float = 0.0
    tools: list = field(default_factory=list)
    refined: bool = False           # final translation != the candidate the judge picked (changed at some stage)
    refined_segment: bool = False   # changed by the Phase 1 per-segment refine()
    refined_doc: bool = False       # changed by the Phase 2 doc_refine()
    translation_phase1: str = ""    # translation as it entered Phase 2, so a rewrite is traceable from this file alone
    reasoning_trace: list = field(default_factory=list)  # NEW: record of all tool calls with results


# ============================================================
# Length constraints
# ============================================================

def wrap_subtitle(text: str) -> str:
    """Break a too-long single line into two at a clause (or word) boundary.

    Without this step MAX_LINE is effectively enforced as a cap on the whole
    translation: on one 645-segment episode none of the published lines contained a
    line break, and all 11 "Line too long" violations were single lines just over
    the limit while 9 of them had plenty of CPS budget left. Real subtitles wrap,
    so squeezing the text instead only loses content — segments 272-274 lost two
    whole source sentences to that squeeze.

    Two lines only. Anything that needs three belongs to segmentation, not here.
    """
    if "\n" in text or len(text) <= MAX_LINE:
        return text

    def balance(i):
        return abs(len(text[:i].rstrip()) - len(text[i:].lstrip()))

    def fits(i):
        return 0 < len(text[:i].rstrip()) <= MAX_LINE \
            and 0 < len(text[i:].lstrip()) <= MAX_LINE

    # Clause boundary first, then a word boundary, and only as a last resort after
    # a hyphen inside a compound — that reads worse but still beats forcing the
    # model to cut content, which is what the alternative amounts to. Never
    # mid-word otherwise.
    punct = [i for i in range(1, len(text)) if text[i - 1] in ",;:.!?\u2014"]
    words = [m.end() for m in re.finditer(r"\s+", text)]
    hyphens = [i for i in range(1, len(text)) if text[i - 1] == "-"]
    for cuts in (punct, words, hyphens):
        ok = [i for i in cuts if fits(i)]
        if ok:
            cut = min(ok, key=balance)           # keep the two lines even
            return text[:cut].rstrip() + "\n" + text[cut:].lstrip()
    return text                                  # one long unbroken word/clause


def length_report(text: str, duration: float) -> dict:
    """Length check for one subtitle line. _constraint_check and doc_refine share this one
    implementation, so the Phase 1 tool and the Phase 2 guardrail cannot apply different
    standards."""
    total_chars = len(text)
    max_chars = int(duration * MAX_CPS)
    cps = total_chars / max(duration, 0.1)
    # Judge line length on the wrapped form: if it fits in two lines it passes.
    # Otherwise the model compresses to hit MAX_LINE for nothing — publishing
    # wraps the line anyway — and compression is what loses content.
    lines = wrap_subtitle(text).split("\n")
    longest_line = max((len(line) for line in lines), default=0)

    violations = []
    is_short = duration < 1.5
    if total_chars > max_chars and not is_short:
        violations.append(f"Too long: {total_chars} chars, max {max_chars}.")
    if longest_line > MAX_LINE:
        violations.append(
            f"Line too long: {longest_line} chars, max {MAX_LINE} "
            f"(even after splitting at clause boundaries — add a comma or shorten)."
        )

    return {
        "valid": not violations,
        "chars": total_chars,
        "max_chars": max_chars,
        "cps": round(cps, 1),
        "longest_line": longest_line,
        "violations": violations,
        "note": "Short segment — meaning > length" if is_short else "",
    }


# ============================================================
# SRT Parser
# ============================================================

def parse_srt(path: str) -> list[Segment]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    segments = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1].strip()
        )
        if not ts_match:
            continue
        start, end = ts_match.group(1), ts_match.group(2)
        text = " ".join(l.strip() for l in lines[2:] if l.strip())
        if not text:
            continue
        segments.append(Segment(index=idx, start=start, end=end, text=text))

    return segments


# ============================================================
# Phase -1: Conservative Deep Content Research
# ============================================================

DEEP_RESEARCH_SYSTEM = """You are a careful media research consultant helping with English→German (de-DE) subtitle translation.

Given a sample of English subtitle lines, identify the content and build a knowledge base. Be CONSERVATIVE:
- Only include German dubbed character names you are highly confident about (official dub versions).
- If you are not sure about a character's dubbed German name, OMIT the "de" field.
- For terminology, only include terms with established standard German translations.
- Do NOT invent or guess German names if uncertain.

Output JSON:
{
  "title": "show or film title (or best guess)",
  "media_type": "tv_series|film|documentary|other",
  "genre": ["crime", "drama", etc.],
  "setting": "when and where (e.g., 'Present-day Los Angeles, crime drama')",
  "synopsis": "one sentence",
  "characters": [
    {"name": "English name", "de": "German dubbed name (ONLY if highly confident)", "role": "brief role", "habits": "notable traits for translation — omit if none"}
  ],
  "domain_knowledge": [
    "A key fact a translator needs to know about this content's domain"
  ],
  "terminology": {
    "English term": "established German translation"
  }
}

CRITICAL RULES for terminology:
- terminology is ONLY for technical/domain terms with established standard translations (e.g. "district attorney" → "Staatsanwalt").
- NEVER put character nicknames, terms of address, or culturally-specific names in terminology — even if you know what they mean. Those belong in characters[].habits or domain_knowledge.
- If a word is a name or nickname used to address someone, do NOT translate it in terminology. Keep proper names as-is."""


def deep_research(claude, segments: list[Segment], source_path: str) -> ContentProfile:
    logger.info("Phase -1: Deep content research (conservative)...")

    sample_size = min(RESEARCH_SAMPLE_SIZE, len(segments))
    step = max(1, len(segments) // sample_size)
    sampled = segments[::step][:sample_size]

    filename = Path(source_path).stem
    lines_text = "\n".join(f"[{s.start}] {s.text}" for s in sampled)

    msg = f"""Filename: {filename}

Sample subtitle lines (spread across the work):
{lines_text}

Identify the media work conservatively. Only include German dubbed names and terms you are HIGHLY CONFIDENT about.
Output JSON."""

    resp = claude.chat(DEEP_RESEARCH_SYSTEM, msg)

    profile = ContentProfile()
    try:
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if m:
            data = json.loads(m.group())
            profile.title = data.get("title", filename)
            profile.media_type = data.get("media_type", "unknown")
            profile.genre = data.get("genre", [])
            profile.setting = data.get("setting", "")
            profile.synopsis = data.get("synopsis", "")
            profile.characters = data.get("characters", [])
            profile.domain_knowledge = data.get("domain_knowledge", [])
            profile.terminology = data.get("terminology", {})
            profile.raw = data
            logger.info(f"  Identified: '{profile.title}' ({profile.media_type}) "
                        f"— {len(profile.characters)} chars, "
                        f"{len(profile.terminology)} terms, "
                        f"{len(profile.domain_knowledge)} domain facts")
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"Deep research parse error: {e}. Using filename as title.")
        profile.title = filename

    # Build dynamic idiom bank for this content
    profile.idiom_bank = build_idiom_bank(claude, profile, sampled)

    return profile


IDIOM_BANK_SYSTEM = """You are a German (de-DE) language expert specializing in idiomatic expressions, Redewendungen, and Sprichwörter.

Given information about a media work (genre, setting, tone), predict the German expressions useful for subtitle translation.

For each category, provide:
- 5-8 relevant Redewendungen/Sprichwörter (idiomatic expressions) for formal/expressive use
- 5-8 umgangssprachliche Ausdrücke (colloquial expressions) for casual speech
- Usage notes

Output JSON:
{
  "categories": {
    "category_name": {
      "idioms": ["Redewendung1", "Sprichwort2", ...],
      "colloquial": ["umgangssprachlicher Ausdruck1", "umgangssprachlicher Ausdruck2", ...],
      "usage_note": "when to use these expressions"
    }
  },
  "genre_notes": "observations about this genre's German linguistic needs"
}

Focus on expressions actually useful for this specific content."""


def build_idiom_bank(claude, profile: ContentProfile, sample_segments: list) -> dict:
    """Build content-specific idiom bank during Phase -1.

    Returns a dict of categories with idioms/colloquial expressions.
    Falls back to empty dict if build fails (dynamic lookup will be used).
    """
    logger.info("  Building content-specific idiom bank...")

    # Analyze tone patterns from sample
    tone_analysis = analyze_tone_patterns(sample_segments)

    genre_str = ", ".join(profile.genre) if profile.genre else "unknown"

    msg = f"""Content: {profile.title}
Media type: {profile.media_type}
Genre: {genre_str}
Setting: {profile.setting}

Sample subtitle analysis:
- Tone patterns detected: {tone_analysis}

Domain knowledge:
{chr(10).join(f"  - {d}" for d in profile.domain_knowledge[:5])}

Predict useful German idiomatic/colloquial expression categories for translating this content.
Focus on 10-15 most relevant categories."""

    try:
        resp = claude.chat(IDIOM_BANK_SYSTEM, msg)
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if m:
            data = json.loads(m.group())
            categories = data.get("categories", {})

            if not categories:
                logger.warning(f"  Idiom bank returned empty categories. Will use dynamic lookup.")
                return {}

            logger.info(f"  ✓ Built idiom bank: {len(categories)} categories "
                       f"(~{sum(len(v.get('idioms', [])) + len(v.get('colloquial', [])) for v in categories.values())} expressions)")
            return categories
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"  ✗ Idiom bank build failed: {e}. Will use dynamic lookup as fallback.")
    except Exception as e:
        logger.error(f"  ✗ Unexpected error building idiom bank: {e}. Will use dynamic lookup.")

    return {}


def analyze_tone_patterns(segments: list) -> str:
    """Quick heuristic analysis of tone patterns in sample."""
    text_all = " ".join(s.text for s in segments).lower()

    patterns = []
    if re.search(r"\b(shit|fuck|damn|hell|ass|bastard|bitch|goddamn|bullshit)\b", text_all):
        patterns.append("profanity/vulgar")
    if text_all.count("!") > len(segments) * 0.2:
        patterns.append("exclamatory/intense")
    if text_all.count("?") > len(segments) * 0.3:
        patterns.append("interrogative/investigative")
    if re.search(r"\b(court|judge|lawyer|objection|witness)\b", text_all):
        patterns.append("legal/formal")
    if re.search(r"\b(detective|cop|police|murder|case)\b", text_all):
        patterns.append("crime/investigation")
    if re.search(r"\b(love|heart|kiss|feel|beautiful)\b", text_all):
        patterns.append("romantic/emotional")

    return ", ".join(patterns) if patterns else "neutral/conversational"


# ============================================================
# Scene Segmentation (with domain_notes, enriched by research)
# ============================================================

SCENE_SEGMENT_SYSTEM = """You are a film/TV scene analyst. Given subtitle lines, identify scene boundaries and label each scene.

Scene change indicators:
- Large time gaps (>30 seconds)
- Change of location or setting
- Change of speakers/characters
- Shift in topic or activity

For each scene provide:
- theme: short snake_case label
- tone: emotional register (tense, humorous, formal, aggressive, casual, etc.)
- description: one sentence about what's happening
- characters: who seems to be speaking
- domain_notes: specific domain knowledge a translator needs for this scene
  (e.g., "police ranks: Detective II is senior to I", "legal: civil deposition vs criminal testimony", etc.
   Leave empty string if no special domain needed.)

Output JSON:
{"scenes": [{"start_line": N, "end_line": N, "theme": "...", "tone": "...",
             "description": "...", "characters": ["..."], "domain_notes": "..."}]}"""


def segment_into_scenes(claude, segments: list[Segment],
                        profile: ContentProfile,
                        batch_size: int = 50) -> list[Scene]:
    all_scenes = []
    scene_id = 0

    context_header = ""
    if profile.title:
        context_header = f"Content: {profile.title} ({profile.media_type})\n"
    if profile.setting:
        context_header += f"Setting: {profile.setting}\n"
    if profile.domain_knowledge:
        context_header += "Domain knowledge:\n" + "\n".join(
            f"  - {d}" for d in profile.domain_knowledge[:8]
        ) + "\n"

    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start:batch_start + batch_size]
        lines_text = "\n".join(f"{s.index}. [{s.start}] {s.text}" for s in batch)

        msg = f"""{context_header}
Subtitle lines:
{lines_text}

Identify scene boundaries with domain notes. Output JSON."""

        resp = claude.chat(SCENE_SEGMENT_SYSTEM, msg)

        try:
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            if m:
                data = json.loads(m.group())
                for s_data in data.get("scenes", []):
                    scene = Scene(
                        id=scene_id,
                        start_idx=s_data.get("start_line", batch[0].index),
                        end_idx=s_data.get("end_line", batch[-1].index),
                        theme=s_data.get("theme", "unknown"),
                        description=s_data.get("description", ""),
                        tone=s_data.get("tone", "neutral"),
                        characters=s_data.get("characters", []),
                        domain_notes=s_data.get("domain_notes", ""),
                    )
                    all_scenes.append(scene)
                    scene_id += 1
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Scene segmentation parse error: {e}")
            all_scenes.append(Scene(
                id=scene_id, start_idx=batch[0].index, end_idx=batch[-1].index,
                theme="unknown", description="", tone="neutral"
            ))
            scene_id += 1

    for seg in segments:
        for scene in all_scenes:
            if scene.start_idx <= seg.index <= scene.end_idx:
                seg.scene_id = scene.id
                seg.scene_theme = scene.theme
                break

    logger.info(f"Identified {len(all_scenes)} scenes")

    # Enrich domain_notes with web_search for key terminology
    all_scenes = enrich_scene_domain_notes(claude, all_scenes, segments, profile)

    return all_scenes


DOMAIN_TERM_IDENTIFIER_SYSTEM = """You are a domain terminology analyst for English→German subtitle translation.

Given scene information and English subtitle text, identify domain-specific terms that need clarification for accurate translation into German.

Look for:
- Technical jargon (medical, legal, sports, police, forensic, etc.)
- Wordplay, puns, or cultural references
- Rank/title systems (military, police, corporate)
- Idioms or slang whose German equivalent needs research
- Ambiguous phrases that could be misunderstood

Output JSON:
{
  "terms": [
    {
      "term": "the exact phrase from subtitles",
      "query": "specific question to search (e.g., 'What does top of the inning mean in baseball?')",
      "priority": "high|medium|low"
    }
  ]
}

Only include terms that genuinely need external knowledge. Max 5 terms per scene."""


def enrich_scene_domain_notes(claude, scenes: list[Scene],
                               segments: list[Segment],
                               profile: ContentProfile) -> list[Scene]:
    """
    Enrich scene domain_notes by identifying key terms and performing web_search.

    Strategy:
    - For each scene with potential domain-specific content
    - Identify 3-5 key terms that need clarification
    - Perform web_search for each term
    - Append results to domain_notes
    """
    logger.info("  Enriching scene domain_notes with terminology lookup...")

    enriched_count = 0

    for scene in scenes:
        # Skip if domain_notes already substantial or scene is very short
        if len(scene.domain_notes) > 600:
            continue

        # Get scene's subtitle text
        scene_segments = [s for s in segments if scene.start_idx <= s.index <= scene.end_idx]
        if len(scene_segments) < 3:
            continue

        scene_text = " ".join(s.text for s in scene_segments[:20])  # Sample first 20 lines

        # Identify terms that need lookup
        context_info = f"Content: {profile.title or 'unknown'}\nGenre: {', '.join(profile.genre)}\n"
        if profile.domain_knowledge:
            context_info += "Series facts: " + " | ".join(profile.domain_knowledge[:6]) + "\n"
        # Find characters with habits that appear in the scene text — match by name substring
        char_habit_notes = []
        if profile.characters:
            scene_text_lower = scene_text.lower()
            for c in profile.characters:
                name = c.get("name", "")
                habits = c.get("habits", "")
                if habits and name and name.split()[0].lower() in scene_text_lower:
                    char_habit_notes.append(f"{name}: {habits}")
        if char_habit_notes:
            context_info += "Character traits: " + "; ".join(char_habit_notes) + "\n"
            # Also seed domain_notes directly so judge/agents always see it regardless of term selection
            seed = "Character context: " + "; ".join(char_habit_notes)
            if not scene.domain_notes:
                scene.domain_notes = seed
            elif seed not in scene.domain_notes:
                scene.domain_notes = seed + "; " + scene.domain_notes
        msg = f"""{context_info}
Scene theme: {scene.theme}
Scene tone: {scene.tone}
Scene description: {scene.description}

Subtitle sample:
{scene_text[:500]}

Identify domain-specific terms that need clarification for translation. Max 3 high-priority terms."""

        try:
            resp = claude.chat(DOMAIN_TERM_IDENTIFIER_SYSTEM, msg)
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            if not m:
                continue

            data = json.loads(m.group())
            terms = data.get("terms", [])

            # Filter high priority terms, limit to 3
            high_priority = [t for t in terms if t.get("priority") == "high"][:3]
            if not high_priority:
                high_priority = terms[:2]  # Fallback: take first 2

            if not high_priority:
                continue

            # Perform web_search for each term
            enrichment = []
            for term_info in high_priority:
                query = term_info.get("query", "")
                if not query:
                    continue

                logger.info(f"    Scene {scene.id}: searching '{query[:60]}'...")
                search_result = web_search_real(query, context=f"{scene.theme} scene, {scene.tone} tone")

                # Keep the full search result context (not just first line)
                # This preserves the "literal meaning vs speaker intent" distinction
                if search_result:
                    # Limit to 300 chars to avoid bloat, but keep context
                    insight = search_result[:300].replace("\n", " ")
                    enrichment.append(f"[{term_info['term']}] {insight}")

            # Append to domain_notes
            if enrichment:
                separator = "; " if scene.domain_notes else ""
                scene.domain_notes += separator + " | ".join(enrichment)
                enriched_count += 1

        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            logger.warning(f"    Scene {scene.id} enrichment failed: {e}")
            continue
        except Exception as e:
            logger.error(f"    Scene {scene.id} unexpected error: {e}")
            continue

    logger.info(f"  ✓ Enriched {enriched_count}/{len(scenes)} scenes with terminology lookup")
    return scenes


# ============================================================
# Claude Client
# ============================================================

def _make_client():
    """An Anthropic-compatible client, selected by PROVIDER.

    No credential is passed as an argument. Each client reads the environment the way its own SDK
    documents - ANTHROPIC_API_KEY for the direct API, the cloud provider's standard chain
    otherwise - which keeps keys out of every traceback and log line that renders call arguments,
    and keeps the deployment out of the code.

    `max_retries=0` is load-bearing. The retry policy that was measured is the explicit backoff in
    `Claude._invoke`; leaving the SDK's own default of 2 in place would silently triple the
    attempts behind each logged call and inflate the latency and cost figures the paper reports.
    """
    if PROVIDER == "bedrock":
        return anthropic.AnthropicBedrock(max_retries=0)
    if PROVIDER == "vertex":
        return anthropic.AnthropicVertex(max_retries=0)
    if PROVIDER != "anthropic":
        raise SystemExit(f"ANTHROPIC_PROVIDER={PROVIDER!r}: expected "
                         f"anthropic, bedrock or vertex")
    return anthropic.Anthropic(max_retries=0)


class Claude:
    # Worth retrying: 429 rate limiting, 529 overloaded, and transient 5xx. Anything else - 400 for
    # a malformed request, 401 for a bad key - is a bug or a misconfiguration that retrying only
    # delays, and burning eight attempts on it hides the cause behind four minutes of backoff.
    RETRY_STATUS = (429, 500, 502, 503, 504, 529)

    def __init__(self):
        self.client = _make_client()
        self.calls = 0

    def _invoke(self, body: dict) -> dict:
        delay = 30
        for attempt in range(8):
            try:
                resp = self.client.messages.create(model=MODEL_ID, **body)
                self.calls += 1
                # A plain dict, so everything downstream keeps indexing it the way it always did:
                # result["content"], block["name"], block["input"].
                #
                # `exclude_none` is not cosmetic. A bare model_dump() materialises every optional
                # field the response model declares, including the ones the wire omitted, as an
                # explicit null - `citations` and `caller` inside content blocks, seven of them
                # inside `usage`. Two things then go wrong. `tool_loop` appends this content
                # straight back onto `messages` and re-sends it, so those nulls become part of the
                # next request body, which is not something the API was ever sent before. And any
                # reader that walks `usage` gets null-valued token counts instead of absent keys.
                # Dropping the nulls reproduces the shape that came off the wire, which is what
                # leaving the downstream code untouched actually requires.
                return resp.model_dump(exclude_none=True)
            except anthropic.APIStatusError as e:
                if e.status_code not in self.RETRY_STATUS:
                    raise
                logger.warning(f"{type(e).__name__} {e.status_code}, waiting {delay}s "
                               f"(attempt {attempt+1}/8)...")
                time.sleep(delay)
                delay = min(delay * 2, 300)
            except anthropic.APIConnectionError as e:
                logger.warning(f"{type(e).__name__}, waiting {delay}s (attempt {attempt+1}/8)...")
                time.sleep(delay)
                delay = min(delay * 2, 300)
        raise RuntimeError("Exceeded max retries due to rate limiting or overload")

    def chat(self, system: str, user_msg: str, **_) -> str:
        body = {
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": user_msg}],
        }
        result = self._invoke(body)
        return self._get_text(result["content"])

    def tool_loop(self, system: str, user_msg: str, tools: list[dict],
                  executor: Callable,
                  max_turns: int = 6) -> tuple[str, list[dict]]:
        messages = [{"role": "user", "content": user_msg}]
        all_tool_calls = []
        content = []

        for _ in range(max_turns):
            body = {
                "max_tokens": MAX_TOKENS,
                "system": system,
                "messages": messages,
                "tools": tools,
            }
            result = self._invoke(body)
            stop_reason = result.get("stop_reason", "end_turn")
            content = result["content"]
            messages.append({"role": "assistant", "content": content})

            if stop_reason != "tool_use":
                return self._get_text(content), all_tool_calls

            tool_results_msg = []
            for block in content:
                if block.get("type") == "tool_use":
                    tool_name = block["name"]
                    tool_input = block["input"]
                    tool_use_id = block["id"]
                    output = executor(tool_name, tool_input)
                    all_tool_calls.append({
                        "tool": tool_name, "input": tool_input, "output": output
                    })
                    tool_results_msg.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(output, ensure_ascii=False),
                    })

            if tool_results_msg:
                messages.append({"role": "user", "content": tool_results_msg})
            else:
                return self._get_text(content), all_tool_calls

        # The tool budget is exhausted while the model still wants to call tools. At that point the
        # text in the last assistant message is only its preamble before calling a tool (e.g. "Let
        # me
        # do a final constraint check.") — the translation has not been written yet. Returning that
        # makes the agent abstain: translate_segment drops the candidate and the judge chooses from
        # a smaller pool.
        #
        # Measured on a 645-segment episode: 40 of 1290 candidates (3.1%) were wasted this way, and
        # 21 of 645 segments ended up with only one candidate. A typical consequence is idx 535: the
        # `natural` agent had the right idea (fold the next line's taunt in) but hit the cap and
        # returned English self-talk, collapsing the pool to one candidate, so the line that
        # shipped was a flat greeting instead of the taunting one the baseline produced.
        #
        # The fix is not to raise max_turns (that brings back the runaway tool loops it was added to
        # suppress) but to spend one more call with no tools attached and collect the answer it was
        # about to give. Only the ~3% that hit the cap trigger this.
        logger.warning(
            f"  tool budget exhausted after {max_turns} turns "
            f"({len(all_tool_calls)} tool calls) — asking for the final answer "
            f"without tools"
        )
        # The Messages API requires alternating user/assistant turns, so another user message cannot
        # be appended. On loop exit messages[-1] is always a user message carrying tool_result, so
        # the
        # nudge is appended to its content as a text block (text must come after tool_result).
        nudge = ("Your tool budget is used up — no more tool calls are available. "
                 "Give your final answer now, in exactly the output format "
                 "requested. Output only that, no commentary.")
        if messages and messages[-1]["role"] == "user" \
                and isinstance(messages[-1]["content"], list):
            messages[-1]["content"].append({"type": "text", "text": nudge})
        else:
            messages.append({"role": "user", "content": nudge})

        try:
            result = self._invoke({
                "max_tokens": MAX_TOKENS,
                "system": system,
                "messages": messages,
            })
            final = self._get_text(result["content"])
            if final:
                return final, all_tool_calls
            logger.warning("  final-answer call returned no text")
        except Exception as exc:
            logger.warning(f"  final-answer call failed: {exc}")

        return self._get_text(content), all_tool_calls

    @staticmethod
    def _get_text(content: list[dict]) -> str:
        parts = [b["text"] for b in content if b.get("type") == "text"]
        return "\n".join(parts).strip()


# ============================================================
# Web Search (Real Browser Scraping - No API Required)
# ============================================================

# Search backend priority
# 1. Selenium Google Search (preferred - real browser, most accurate results)
# 2. DuckDuckGo (lightweight fallback)

SEARCH_AVAILABLE = False
SEARCH_METHOD = "None"

# Allow skipping backend loading entirely via env var (for evaluation/offline runs, so a Selenium
# browser launch cannot hang startup)
if os.environ.get("SMART_NO_SEARCH") == "1":
    logger.info("SMART_NO_SEARCH=1 -> skipping search backend loading")
    def web_search_free(query, **kwargs):
        return "ERROR: search disabled (SMART_NO_SEARCH)"
    def cleanup_scraper():
        pass
    _SKIP_SEARCH_IMPORT = True
else:
    _SKIP_SEARCH_IMPORT = False

# Priority 1: Selenium Google Search (RECOMMENDED)
if _SKIP_SEARCH_IMPORT:
    pass
else:
  try:
    from web_search_selenium_google import web_search_free, cleanup_scraper
    SEARCH_AVAILABLE = True
    SEARCH_METHOD = "Selenium Google Search (real browser)"
    logger.info("✓ Using Selenium Google Search (real browser scraping)")
  except ImportError as e:
    logger.warning(f"Selenium Google Search not available: {e}")

    # Priority 2: DuckDuckGo fallback
    try:
        from web_search_requests import web_search_free
        SEARCH_AVAILABLE = True
        SEARCH_METHOD = "DuckDuckGo (fallback)"
        logger.info("Using DuckDuckGo search (fallback)")

        # Define dummy cleanup function
        def cleanup_scraper():
            pass
    except ImportError:
        try:
            from web_search_selenium import web_search_free
            SEARCH_AVAILABLE = True
            SEARCH_METHOD = "Selenium DuckDuckGo"
            logger.info("Using Selenium DuckDuckGo search")

            def cleanup_scraper():
                pass
        except ImportError:
            SEARCH_AVAILABLE = False
            SEARCH_METHOD = "None"
            logger.error("No search module available. Install: pip install selenium webdriver-manager")

            # Define dummy functions to avoid errors
            def web_search_free(query, **kwargs):
                return "ERROR: No search backend available"

            def cleanup_scraper():
                pass


def web_search_real(query: str, context: str = "", max_results: int = 3) -> str:
    """
    Web search using available search backend.

    Args:
        query: Search query
        context: Additional context to append to query
        max_results: Number of results to return (default: 3)

    Returns:
        Formatted search results or error message
    """
    if not SEARCH_AVAILABLE:
        error_msg = """ERROR: No search module available.

Install lightweight search (recommended for servers):
  pip install duckduckgo-search

Or install Selenium (requires Chrome browser):
  pip install selenium webdriver-manager
  bash install_chrome_no_sudo.sh"""
        logger.error("No search backend available")
        return error_msg

    try:
        # Only pass the query to the search engine, not the context (context causes long/noisy queries)
        logger.info(f"    🔍 Search ({SEARCH_METHOD}): '{query[:80]}'")
        result = web_search_free(query, max_results=max_results)
        logger.info(f"    📄 Search result:\n{result}")
        return result
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search error: {e}\n\nPlease check network connection and try again."


# ============================================================
# Tool Definitions (v4)
# ============================================================

TOOL_SCHEMAS = [
    {
        "name": "web_search",
        "description": (
            "Search for domain knowledge needed to translate accurately into German. Use when you encounter: "
            "(1) sports terminology or rules, "
            "(2) police/legal/medical jargon or ranks, "
            "(3) wordplay, puns, or cultural references, "
            "(4) character names or show-specific facts, "
            "(5) English slang whose meaning is unclear from context. "
            "Ask precise factual questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Factual question about meaning (e.g., 'What does top of the inning mean in baseball?')"
                },
                "context": {
                    "type": "string",
                    "description": "Brief context: who is speaking, what is the tone/scene"
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "terminology_lookup",
        "description": "Look up if an English name/term already has an established German translation. Use for character names, place names, recurring terms.",
        "input_schema": {
            "type": "object",
            "properties": {
                "term": {"type": "string", "description": "English term to look up"}
            },
            "required": ["term"],
        },
    },
    {
        "name": "terminology_register",
        "description": "Register a new English→German term pair. Call when translating a name/term for the first time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "English term"},
                "target": {"type": "string", "description": "German translation"},
                "category": {"type": "string", "enum": ["name", "place", "title", "phrase"]},
            },
            "required": ["source", "target", "category"],
        },
    },
    {
        "name": "constraint_check",
        "description": "Check if German translation fits subtitle limits (chars/sec ≤ 17, line ≤ 42 chars). Call to validate final translation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "German translation"},
                "duration": {"type": "number", "description": "Segment duration in seconds"},
            },
            "required": ["text", "duration"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search translation memory for similar previously-translated segments.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "English text to find similar past translations"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_context",
        "description": "Get surrounding subtitle lines, scene info, and domain notes for context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["before", "after", "both"]},
                "count": {"type": "integer", "description": "Number of lines (max 8)"},
            },
            "required": ["direction"],
        },
    },
    {
        "name": "idiom_lookup",
        "description": "Look up suitable German idiomatic expressions (Redewendungen, Sprichwörter, umgangssprachliche Ausdrücke) for a given emotional/situational context. Uses a content-specific expression bank built during Phase -1.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Emotional/situational category (e.g., anger, fear, sarcasm, wordplay, humor, urgency, etc.)",
                },
                "context": {
                    "type": "string",
                    "description": "Brief description of what you want to express (e.g., 'detective making a pun about humerus/humor')",
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "fluency_check",
        "description": "Check if a German translation sounds natural and fluent at the correct register. Catches anglicisms, register mismatches, Sie/du formality errors, and vulgarity calibration issues.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "German translation to check"},
                "source": {"type": "string", "description": "Original English text"},
                "tone": {"type": "string", "description": "Expected tone (casual, formal, tense, humorous, etc.)"},
            },
            "required": ["text", "source"],
        },
    },
]


# ============================================================
# Tool Executor
# ============================================================

class ToolExecutor:
    def __init__(self, claude_client=None):
        self.terminology: dict[str, dict] = {}
        self.memory: list[dict] = []
        self._context: Optional[Context] = None
        self._claude = claude_client
        self._dynamic_idiom_cache: dict[str, dict] = {}  # Cache for dynamic idiom lookups

    def set_context(self, ctx: Context):
        self._context = ctx

    def set_claude(self, claude):
        self._claude = claude

    def add_to_memory(self, source: str, target: str):
        self.memory.append({"source": source, "target": target})

    def seed_from_profile(self, profile: ContentProfile):
        """Pre-load high-confidence terminology from deep research."""
        for term, de in profile.terminology.items():
            if term and de and term.lower() not in (k.lower() for k in self.terminology):
                self.terminology[term] = {"target": de, "category": "phrase", "count": 0}

        for char in profile.characters:
            name = char.get("name", "")
            de = char.get("de", "")
            if name and de and name.lower() not in (k.lower() for k in self.terminology):
                self.terminology[name] = {"target": de, "category": "name", "count": 0}

        logger.info(f"  Pre-seeded {len(self.terminology)} terms from research")

    def execute(self, tool_name: str, tool_input: dict) -> dict:
        handlers = {
            "web_search": self._web_search,
            "terminology_lookup": self._terminology_lookup,
            "terminology_register": self._terminology_register,
            "constraint_check": self._constraint_check,
            "memory_search": self._memory_search,
            "get_context": self._get_context,
            "idiom_lookup": self._idiom_lookup,
            "fluency_check": self._fluency_check,
        }
        fn = handlers.get(tool_name)
        if not fn:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn(tool_input)
        except Exception as e:
            return {"error": str(e)}

    def _web_search(self, inp: dict) -> dict:
        query = inp["query"]
        context = inp.get("context", "")

        result = web_search_real(query, context)

        is_error = result.startswith("ERROR:") or result.startswith("No search results")

        if is_error:
            return {"query": query, "result": result, "source": "web_search", "error": True}

        interpretation = self._interpret_search_for_translation(query, result, context)

        return {
            "query": query,
            "result": f"{result}\n\n=== 翻译指导 ===\n{interpretation}",
            "source": "web_search_interpreted",
            "error": False,
        }

    def _interpret_search_for_translation(self, query: str, search_result: str, context: str) -> str:
        """
        Use Claude to interpret search results for translation.

        IMPORTANT: This is a GENERIC interpretation layer, with NO specific examples.
        Must work for any domain (sports, medical, legal, etc.) and any content.
        """
        if not self._claude:
            return ""

        # Get scene context if available
        scene_info = ""
        if self._context and self._context.scene:
            scene = self._context.scene
            scene_info = f"\nScene tone: {scene.tone}\nScene description: {scene.description}"

        system = """You are a linguistic consultant for subtitle translation.

CRITICAL RULE: The search results are AUTHORITATIVE. Your job is to explain what the search results mean for translation. DO NOT contradict or second-guess the search results based on context alone.

Given search results about a term, analyze:

1. **Domain meaning**: What do the search results say this term means? State it clearly.

2. **Conversational register**: Given the scene context, should the German translation use formal domain terminology or colloquial equivalent?
   - Formal context (broadcast, official) → domain terminology
   - Casual context (banter, chat) → colloquial equivalent
   - Interrupted speech → keep natural flow

Output format:
字面: [what the search results say this term means — follow the search results faithfully]
语境: [register guidance based on scene tone]
建议: [formal vs colloquial, and why]

Be concise (3-5 sentences). NO specific translation examples. Do NOT invent alternative meanings not supported by the search results."""

        msg = f"""Term/phrase: {query}

Search results:
{search_result}

Context: {context}{scene_info}

Provide translation guidance based on the search results above."""

        try:
            guidance = self._claude.chat(system, msg)
            return guidance
        except Exception as e:
            logger.warning(f"Search interpretation failed: {e}")
            return "(guidance unavailable)"

    def _terminology_lookup(self, inp: dict) -> dict:
        term = inp["term"].lower()
        for k, v in self.terminology.items():
            if k.lower() == term:
                return {"found": True, "source": k, **v}
        partials = [
            {"source": k, "target": v["target"], "category": v["category"]}
            for k, v in self.terminology.items()
            if term in k.lower() or k.lower() in term
        ]
        return {"found": False, "partial_matches": partials[:5]}

    def _terminology_register(self, inp: dict) -> dict:
        src, tgt = inp["source"], inp["target"]
        cat = inp.get("category", "phrase")
        key = src.lower()
        for k in self.terminology:
            if k.lower() == key:
                existing = self.terminology[k]["target"]
                if existing != tgt:
                    return {"registered": False, "conflict": True,
                            "existing_target": existing}
                self.terminology[k]["count"] += 1
                return {"registered": True, "already_existed": True}
        self.terminology[src] = {"target": tgt, "category": cat, "count": 1}
        return {"registered": True, "source": src, "target": tgt}

    def _constraint_check(self, inp: dict) -> dict:
        return length_report(inp["text"], inp["duration"])

    def _memory_search(self, inp: dict) -> dict:
        query = inp["query"].lower()
        query_words = set(query.split())
        scored = []
        for e in self.memory:
            src_words = set(e["source"].lower().split())
            overlap = len(query_words & src_words) / max(len(query_words), 1)
            if overlap > 0.2:
                scored.append((overlap, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return {"results": [
            {"source": e["source"], "target": e["target"], "similarity": round(s, 2)}
            for s, e in scored[:5]
        ]}

    def _get_context(self, inp: dict) -> dict:
        if not self._context:
            return {"error": "No context available"}
        direction = inp.get("direction", "both")
        count = min(inp.get("count", 5), 8)
        result = {}

        if self._context.scene:
            scene = self._context.scene
            result["scene"] = {
                "theme": scene.theme,
                "tone": scene.tone,
                "description": scene.description,
                "characters": scene.characters,
            }
            if scene.domain_notes:
                result["scene"]["domain_notes"] = scene.domain_notes

        if self._context.profile and self._context.profile.title:
            p = self._context.profile
            result["content"] = {
                "title": p.title,
                "genre": p.genre,
                "setting": p.setting,
            }

        if direction in ("before", "both"):
            segs = self._context.preceding_src[-count:]
            trans = self._context.preceding_tgt[-count:]
            result["before"] = [
                {"source": s.text, "translation": t}
                for s, t in zip(segs, trans[-len(segs):] if trans else [""] * len(segs))
            ]
        if direction in ("after", "both"):
            result["after"] = [
                {"source": s.text} for s in self._context.succeeding_src[:count]
            ]
        return result

    def _idiom_lookup(self, inp: dict) -> dict:
        """
        Hybrid idiom lookup: Phase -1 prebuilt bank + dynamic fallback.
        Strategy:
        1. First check content-specific idiom_bank from ContentProfile
        2. If not found, dynamically query Claude
        """
        category = inp["category"]
        context = inp.get("context", "")

        # Strategy 1: Check prebuilt idiom bank from ContentProfile
        if self._context and self._context.profile and self._context.profile.idiom_bank:
            idiom_bank = self._context.profile.idiom_bank

            # Exact category match
            if category in idiom_bank:
                cat_data = idiom_bank[category]
                result = {
                    "category": category,
                    "source": "content_profile",
                    "idioms": cat_data.get("idioms", []),
                    "colloquial_expressions": cat_data.get("colloquial", []),
                    "usage_note": cat_data.get("usage_note", "From content expression bank"),
                }
                return result

            # Fuzzy match: find similar categories
            similar = []
            for k, v in idiom_bank.items():
                if any(c in k.lower() for c in category.lower().split()) or \
                   any(c in category.lower() for c in k.lower().split()):
                    similar.append({
                        "category": k,
                        "idioms": v.get("idioms", [])[:3],
                        "colloquial": v.get("colloquial", [])[:3],
                    })

            if similar:
                return {
                    "category": category,
                    "source": "content_profile_fuzzy",
                    "similar_categories": similar[:3],
                    "note": f"No exact match for '{category}', showing similar categories from content bank",
                }

        # Strategy 2: Dynamic fallback - query Claude for expressions
        return self._dynamic_idiom_lookup(category, context)

    def _dynamic_idiom_lookup(self, category: str, context: str) -> dict:
        """Dynamic idiom lookup using Claude when prebuilt bank doesn't have the category.

        Uses caching to avoid repeated API calls for same category.
        """
        if not self._claude:
            return {"error": "Claude client not available and category not in prebuilt bank"}

        # Check cache first (category-based cache, context ignored for cache key)
        cache_key = category.lower().strip()
        if cache_key in self._dynamic_idiom_cache:
            cached = self._dynamic_idiom_cache[cache_key].copy()
            cached["source"] = "dynamic_query_cached"
            logger.info(f"    → Using cached dynamic lookup for '{category}'")
            return cached

        logger.info(f"    → Dynamic idiom lookup for '{category}' (not in prebuilt bank)")

        system = """You are a German language expert specializing in idiomatic expressions for subtitle translation.

Given a category and context, suggest suitable German expressions.

Provide:
- Redewendungen/Sprichwörter: idiomatic expressions for formal/expressive output
- umgangssprachliche Ausdrücke: casual expressions for everyday dialogue
- Usage notes: when to use each type

Output JSON:
{
  "idioms": ["Ausdruck1", "Ausdruck2", ...],
  "colloquial": ["umgangssprachlich1", "umgangssprachlich2", ...],
  "usage_note": "when to use these expressions"
}"""

        msg = f"""Category: {category}
Context: {context}

Suggest 5-8 suitable Chinese expressions (both formal idioms and colloquial) for this context."""

        try:
            resp = self._claude.chat(system, msg)
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            if m:
                data = json.loads(m.group())
                data["category"] = category
                data["source"] = "dynamic_query"

                # Cache the result
                self._dynamic_idiom_cache[cache_key] = data.copy()

                return data
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Dynamic idiom lookup failed: {e}")

        # Ultimate fallback: return empty with helpful message
        return {
            "category": category,
            "source": "fallback_empty",
            "idioms": [],
            "colloquial_expressions": [],
            "note": f"No expressions found for category '{category}'. Consider rephrasing the category or providing more context.",
        }

    def _fluency_check(self, inp: dict) -> dict:
        """Register-aware fluency check (v3's improved version)."""
        text = inp["text"]
        source = inp["source"]
        tone = inp.get("tone", "neutral")

        if not self._claude:
            return {"error": "Claude client not available"}

        system = """You are a German language expert checking subtitle translations for fluency and register.

Check for THREE failure modes:
1. Anglizismen (anglicisms) — unnatural borrowed English words/structure when a German equivalent exists; over-literal phrasing
2. Formality/register mismatch — wrong Sie/du choice for the relationship (Formality Error): formal Sie in casual dialogue, or informal du in formal/tense scenes
3. Vulgarity calibration — English slang/profanity should be rendered at the register appropriate for a professional German TV subtitle:
   - NOT maximally crude (literal English profanity transplanted)
   - NOT sanitized/euphemistic (losing the original edge)
   - Standard: "How would a professional German subtitle editor render this for a mainstream drama?"

Also check:
4. Grammar — case agreement (Nominativ/Akkusativ/Dativ/Genitiv), gender, and verb-final word order in subordinate clauses
5. Tone match — does register fit speaker's role and scene?
6. Expression opportunity — could a German Redewendung/Sprichwort improve it WITHOUT changing register?

Output JSON:
{"fluent": true/false, "score": 1-10, "issues": ["..."], "suggestion": "improved version or empty", "expression_opportunity": "Redewendung if any, else empty"}"""

        msg = f"""Source (English): {source}
Translation (Chinese): {text}
Expected tone: {tone}

JSON output:"""

        resp = self._claude.chat(system, msg)
        try:
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            if m:
                return json.loads(m.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {"fluent": True, "score": 7, "issues": [], "suggestion": ""}


# ============================================================
# Agent Definitions (v4 — v2 prompts, with domain notes awareness)
# ============================================================

DEFAULT_AGENTS = {
    "faithful": {
        "prompt": """You are a faithful English→German (de-DE) subtitle translator with cultural sensitivity.

PRIORITY: Semantic accuracy with natural German expression + character personality preservation.

WORKFLOW:
1. Check **Domain Notes** and **Scene** info in the prompt.
   Domain Notes provide UNDERSTANDING, not ready-made translations.
   - Read the literal domain meaning AND the speaker's conversational intent.
   - Don't use technical terminology if speaker tone is casual.
   - Incomplete/interrupted sentences → natural conversational flow.

2. Call terminology_lookup for any names/terms.

3. ONLY call web_search if you genuinely do NOT know what a word/phrase means.
   DO NOT search for things you already understand — trust your own knowledge first.
   Use only for: truly opaque jargon, cultural references you're unsure about, ambiguous slang.
   - Read the "Translation Guidance" section carefully.
   - Formal context → technical German; Casual → colloquial equivalents; Interrupted → natural flow.
   - Match register to scene tone and speaker relationship (Sie/du).

4. Call get_context if you need surrounding lines.

5. Translate preserving full meaning into natural German. Avoid anglicisms.
   - Use natural German word order (verb-final in subordinate clauses).
   - Casual speech → umgangssprachliches Register.
   - Choose Sie/du consistently with the speakers' relationship.
   - Profanity → match original register (professional subtitle standard).
   - Slang → German street-talk equivalents.

   **CREATIVE ADAPTATION (when appropriate):**
   - Wordplay/puns → create equivalent German wordplay if possible. Check get_context for the
     next line reacting to this — if so, wordplay MUST land in German.
   - Cultural references → if US-specific, consider German-speaking cultural equivalents.
   - Character personality → let word choice reflect speaker's voice.

6. If source has figurative/emotional language, call idiom_lookup.

7. Call constraint_check with your translation and duration.

8. If constraint fails, shorten ONCE and output. Do NOT call constraint_check more than twice total.

9. Call terminology_register for new names.

Output ONLY the final German translation.""",
        "tools": ["web_search", "terminology_lookup", "terminology_register", "constraint_check",
                  "memory_search", "get_context", "idiom_lookup"],
    },
    "natural": {
        "prompt": """You are a natural-sounding English→German (de-DE) subtitle translator with creative flair.

PRIORITY: Natural German at the correct register + character voice + cultural adaptation.
Goal: German audience feels what English audience feels.

WORKFLOW:
1. Check **Domain Notes** and **Scene** info.
2. Call terminology_lookup for names/terms.
3. ONLY call web_search if you genuinely don't know what a word/phrase means. Don't search things you already understand.
   - Formal context → technical German; Casual → colloquial equivalents.
4. Call get_context to confirm scene tone.
5. Translate at the register fitting the speaker and scene.
   - Avoid Anglizismen.
   - Choose Sie/du to match the speakers' relationship.
   - Different characters should "sound" different (older/younger, rough/refined).
   - Puns/wordplay → create German equivalent if possible.
   - Cultural humor → adapt function, not literal form.
   - Personality markers: gruff → direkt und schroff, sarcastic → ironisch, nervous → weitschweifig.
6. Call fluency_check to verify naturalness AND register.
7. If fluency check suggests improvements, revise ONCE.
8. Call constraint_check to verify length. If fails, shorten ONCE and output. Do NOT loop.

Key rules:
- Avoid Anglizismen (unnatural English-structure calques)
- Avoid over-formalization of casual dialogue
- Register: legal/court → formal; police briefing → professional; chat → natürlich-umgangssprachlich; street → derb
- Domain knowledge informs UNDERSTANDING, not direct word substitution.

Output ONLY the final German translation.""",
        "tools": ["web_search", "terminology_lookup", "terminology_register", "constraint_check",
                  "memory_search", "fluency_check", "get_context", "idiom_lookup"],
    },
    "expressive": {
        "prompt": """You are an expressive English→German (de-DE) subtitle translator for drama/crime shows.

PRIORITY: Capture emotion, character voice, and dramatic tension.

WORKFLOW:
1. Check **Domain Notes** and **Scene** info.
2. Call get_context to understand the emotional situation.
3. Call terminology_lookup for character names.
4. If cultural references, wordplay, or ambiguous phrases appear, call web_search.
5. Call idiom_lookup with the relevant emotional category.
6. Translate with emotionally-faithful German. Use Redewendungen or Ausdrücke when they enhance impact.
7. Call fluency_check to ensure natural sound.
8. Call constraint_check to verify length.

Key rules:
- Anger → forceful German (Register von Vorwurf/Beschimpfung)
- Tension → short, punchy sentences
- Sarcasm → natürliche Ironie/Sarkasmus in German
- Humor/wordplay → adapt to German humor patterns

Output ONLY the final German translation.""",
        "tools": ["web_search", "terminology_lookup", "terminology_register", "constraint_check",
                  "get_context", "idiom_lookup", "fluency_check"],
    },
    "length_aware": {
        "prompt": """You are a length-constrained English→German (de-DE) subtitle translator.

PRIORITY: Fit display constraints. Duration={duration}s, max {max_chars} chars.

WORKFLOW:
1. Call terminology_lookup for names.
2. If domain-specific terms appear, call web_search to understand the meaning first.
3. Translate concisely — target ≤{max_chars} characters. Avoid anglicisms.
4. Call constraint_check — if FAILS, shorten and re-check.

Tips:
- Drop redundant particles/fillers when meaning is clear from context
- Use shorter German equivalents (e.g., "schon" for "already", "ja" for confirmations)
- Use compact German compound nouns to combine ideas into a single word

Output ONLY the final German translation.""",
        "tools": ["web_search", "terminology_lookup", "constraint_check",
                  "terminology_register", "idiom_lookup"],
    },
    "colloquial": {
        "prompt": """You are a colloquial English→German (de-DE) subtitle translator for crime/cop dramas.

PRIORITY: Translate slang, profanity, and street talk into authentic German equivalents.

WORKFLOW:
1. Call terminology_lookup for character names.
2. If slang meaning is unclear, call web_search first to understand what it actually means.
3. Call idiom_lookup with category (threat, dismissal, complaint, etc.).
4. Translate matching register — use German slang/umgangssprachliche equivalents. Don't sanitize profanity.
5. Call fluency_check to make sure it sounds like real German street talk.
6. Call constraint_check to verify length.

Key rules:
- Target register: how would a professional German subtitle editor render this for mainstream streaming?
  Authentic to the original edge, calibrated for the medium — not maximally crude, not sanitized.
- Cop slang → Register deutscher Krimiserien
- Street talk → derbe/umgangssprachliche Straßensprache
- Regional German variants OK for character differentiation

Output ONLY the final German translation.""",
        "tools": ["web_search", "terminology_lookup", "terminology_register", "constraint_check",
                  "idiom_lookup", "fluency_check", "memory_search"],
    },
}

DEFAULT_POLICY = {
    "default": ["faithful", "natural"],
    "short": ["faithful", "natural"],
    "long": ["faithful", "natural", "length_aware"],
    "expressive": ["faithful", "natural", "expressive"],
    "colloquial": ["faithful", "natural", "colloquial"],
}


# ============================================================
# Router
# ============================================================

class Router:
    def __init__(self, config: dict = None):
        if config:
            self.policy = config.get("policy", DEFAULT_POLICY)
            self.agents = config.get("agents", DEFAULT_AGENTS)
        else:
            self.policy = dict(DEFAULT_POLICY)
            self.agents = dict(DEFAULT_AGENTS)
        self.log: list[dict] = []

    def classify(self, ctx: Context) -> str:
        text = ctx.segment.text
        word_count = len(text.split())

        has_slang = bool(re.search(
            r"\b(shit|fuck|damn|hell|ass|bastard|crap|bitch|goddamn|bullshit|motherfuck)\b",
            text, re.IGNORECASE
        ))
        has_emotion = bool(re.search(r"[!?]{2,}|\.{3}|[A-Z]{3,}", text))
        scene_tone = ctx.scene.tone if ctx.scene else ""

        ctx.features = {
            "word_count": word_count, "has_slang": has_slang,
            "has_emotion": has_emotion, "duration": ctx.segment.duration,
            "scene_tone": scene_tone,
        }

        if has_slang:
            return "colloquial"
        if has_emotion or scene_tone in ("tense", "emotional", "dramatic"):
            return "expressive"
        if word_count > 15:
            return "long"
        if word_count <= 3:
            return "short"
        return "default"

    def get_agents(self, category: str, ctx: Context) -> list[dict]:
        agent_names = self.policy.get(category, self.policy["default"])
        result = []
        for name in agent_names:
            agent_def = self.agents.get(name)
            if not agent_def:
                continue
            prompt = agent_def["prompt"]
            if name == "length_aware":
                dur = ctx.segment.duration
                max_c = int(dur * MAX_CPS)
                prompt = prompt.replace("{duration}", f"{dur:.1f}")
                prompt = prompt.replace("{max_chars}", str(max_c))
            result.append({
                "name": name,
                "prompt": prompt,
                "tools": agent_def["tools"],
            })
        return result

    def record(self, category: str, agent: str, score: float, features: dict, tools_used: list):
        self.log.append({
            "category": category, "agent": agent, "score": score,
            "features": features, "tools": tools_used,
        })

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"policy": self.policy, "agents": self.agents}, f,
                      ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "Router":
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return Router(config)


# ============================================================
# Judge Agent
# ============================================================

JUDGE_SYSTEM = """You are a subtitle translation quality judge (English→German, de-DE). Score candidates on:
- accuracy (1-10): meaning/intent preservation (not literal word match)
- fluency (1-10): natural German — penalize anglicisms AND over-formalization
- register_match (1-10): does the register fit the speaker, scene, AND medium?
  The medium is a professional German TV/streaming subtitle. Penalize when:
  (a) formal/professional dialogue sounds casual or flippant
  (b) casual dialogue sounds stiff
  (c) wrong Sie/du choice for the speakers' relationship (Formality Error)
  (d) English slang/profanity is rendered more crudely than a professional subtitle editor would
      — the standard is "authentic but not gratuitously coarse for the medium"
  (e) English slang is over-sanitized, losing the original edge entirely
- idiom_usage (1-10): appropriate German Redewendungen/umgangssprachliche Ausdrücke at the right register
- emotion (1-10): captures tone/feeling of the original
- consistency (1-10): consistent with context and terminology
- length (1-10): fits time constraint
- creativity (1-10): NEW DIMENSION for dialogue-heavy content
  * Does it preserve wordplay/humor/puns? (German equivalent > literal translation)
  * Cultural references adapted or lost? (German-speaking audience equivalent > direct translation)
  * Character personality expressed? (distinct voice > generic translation)
  * Clever adaptations that work BETTER than literal? (BONUS points)
  * Penalty for flattening humor, losing character voice, or missing cultural adaptation opportunities

Overall = weighted average. **Creativity, register_match, and accuracy are equally important for dialogue.**
When comparing candidates: a literal 7/10 that loses a pun vs creative 8/10 with German wordplay → favor creative.

Word-choice precision: penalize generic verbs/nouns when a more specific word exists.
- Consider the speaker's situation and the specific action/state: is there a German word that captures both the meaning AND the social/emotional context?
- Also penalize case-agreement or gender errors (Grammar) and wrong verb-final word order in subordinate clauses.
- Domain terms identified via web_search must be preserved exactly — do not reward paraphrases of technical terminology.
A competent-but-generic translation that misses the precise word scores ≤7, even if fluent.

Respond ONLY as a JSON array."""


def judge_candidates(claude: Claude, source: str, candidates: list[Candidate],
                     context_trans: list[str], duration: float,
                     scene: Optional[Scene] = None,
                     profile: Optional[ContentProfile] = None) -> list[Candidate]:
    ctx_str = " | ".join(context_trans[-3:]) if context_trans else "(start of video)"
    max_c = int(duration * MAX_CPS)
    scene_info = ""
    if scene:
        scene_info = f"\nScene: {scene.theme} (tone: {scene.tone}) — {scene.description}"
        if scene.domain_notes:
            scene_info += f"\nDomain: {scene.domain_notes}"
    profile_info = ""
    if profile and profile.title:
        profile_info = f"\nContent: {profile.title} ({', '.join(profile.genre)})"

    cands_text = "\n".join(
        f"{i+1}. [{c.agent}]: \"{c.text}\"" for i, c in enumerate(candidates)
    )

    msg = f"""Source (English): {source}
Duration: {duration:.1f}s (max ~{max_c} German chars){scene_info}{profile_info}
Recent translations: {ctx_str}

Candidates:
{cands_text}

Score each. Penalize anglicisms AND over-formalization. Penalize register mismatch.
IMPORTANT: If "Domain:" context is provided, a translation contradicting it must be penalized on accuracy.
Output JSON array:
[{{"id":1,"accuracy":N,"fluency":N,"register_match":N,"idiom_usage":N,"emotion":N,"consistency":N,"length":N,"creativity":N,"overall":N.N,"critique":"..."}}]"""

    resp = claude.chat(JUDGE_SYSTEM, msg)

    try:
        arr = json.loads(re.search(r"\[.*\]", resp, re.DOTALL).group())
        for entry in arr:
            idx = entry.get("id", 1) - 1
            if 0 <= idx < len(candidates):
                candidates[idx].score = float(entry.get("overall", 5.0))
                candidates[idx].detail = entry
    except Exception as e:
        logger.warning(f"Judge parse error: {e}")
        for c in candidates:
            c.score = 5.0

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates


# ============================================================
# Refiner Agent
# ============================================================

REFINER_SYSTEM = """You refine German (de-DE) subtitle translations based on judge critique with creative license.

Rules:
- Fix what's critiqued.
- If critique mentions Anglizismen or unnatural German, rewrite to sound native.
- If critique says "too literal" or "bland", add character voice/personality.
- If a German Redewendung/Sprichwort would improve it, use one.
- If wordplay in English was lost, CREATE a German equivalent if possible.
- Fix any Sie/du formality errors and case-agreement (Grammar) issues.
- Keep concise (subtitle format).

CREATIVE LICENSE:
- You MAY deviate from literal meaning to preserve humor/personality.
- You MAY add umgangssprachliche Ausdrücke that match character voice and scene tone.
- You MAY replace cultural references with German-speaking equivalents.
- Goal: German audience feels the same as English audience.

HARD CONSTRAINT — do NOT violate under any creative license:
- Domain-specific terms correctly translated by the agent MUST be preserved.
- If web_search determined a term's meaning, trust that result.
- Creative rewriting applies to STYLE and REGISTER, not to factual domain terminology.

REGISTER — one direction only:
- The Scene tone you are given is the target. Never make a line MORE formal or MORE
  literary than the scene calls for. In casual, taunting, playful or aggressive scenes
  that means: do not swap a spoken word for a written-register synonym
  (Leute→meine Herren, du→Sie, gehen→sich begeben, sagen→äußern and the like).
- Raising the register is only allowed when the critique explicitly asks for it.

NO CHANGE IS A VALID ANSWER:
- If the critique names no concrete, actionable defect — or concedes the line is already
  appropriate — echo the translation back unchanged. Do not swap synonyms just to show
  you did something; a needless rewrite is a regression.

Output ONLY the German translation (refined, or echoed unchanged), no labels, no quotes."""


def refine(claude: Claude, source: str, candidate: Candidate,
           context_trans: list[str], duration: float,
           scene_tone: str = "") -> str:
    if candidate.score >= 8.5:
        return candidate.text
    critique = candidate.detail.get("critique", "")
    if not critique:
        return candidate.text

    ctx = " | ".join(context_trans[-3:]) if context_trans else "(none)"
    max_c = int(duration * MAX_CPS)
    # scene_tone must be passed in: REFINER_SYSTEM asks twice for output that "matches character
    # voice and scene tone", yet this message used to carry no tone information at all — asking
    # the model to match something it was never told. idx 535 failed exactly this way: the scene
    # was tense/darkly humorous/bold and the critique asked for more mockery, but the refiner made
    # the line more literary instead.
    msg = f"""Source: {source}
Translation: {candidate.text}
Critique: {critique}
Score: {candidate.score}/10
Scene tone: {scene_tone or "(unknown)"}
Context: {ctx}
Max chars: {max_c}

Fix the concrete defect the critique names, keeping the line within the Scene tone.
If the critique names no actionable defect, echo the translation back unchanged.
If a German Redewendung or Ausdruck fits naturally, use it.
Output ONLY the refined German translation."""

    result = claude.chat(REFINER_SYSTEM, msg)
    return result.strip().strip('"').strip('"').strip("'")


# ============================================================
# Document Refiner
# ============================================================

DOC_REFINER_SYSTEM = """You are a German (de-DE) subtitle editor reviewing translations for consistency, coherence, and idiomatic quality.

Check THREE things:

## 1. TERMINOLOGY CONSISTENCY
Same English names/concepts → same German translation throughout.

## 2. CONVERSATIONAL COHERENCE
Fix translations that break dialogue logic. Ensure natural conversation flow.
- Question-answer pairs should match
- Pronouns and address consistent (Sie/du usage)
- Response logically follows what was just said

## 3. IDIOMATIC QUALITY
- Replace any remaining Anglizismen with natural German
- Where a Redewendung or umgangssprachlicher Ausdruck would sound better, suggest it
- Ensure register consistency within a scene

## What NOT to change:
- Lines already natural and consistent
- Working expressions — don't replace one good expression with another

Output: numbered list for ALL lines with ONLY the German translation. Unchanged lines repeated as-is."""


def _split_terminology(terminology: dict) -> tuple[list, list]:
    """Split the terminology table into a must-match tier and a reference-only tier.

    doc_refine used to inject the whole table as a hard constraint. The problem is that most
    entries have category "phrase", which is not terminology at all — it only records how one
    line happened to be rendered this time, including ordinary words. Treated as a global rule,
    such an entry drags correct renderings elsewhere off target: "overseas assets" recorded in
    its property sense (itself a mistranslation) rewrote "assets from the field" in another line
    from its correct operatives sense into the property sense.

    Only proper nouns (name / place / title) and explicit domain terms (term) need locking;
    phrase entries are reference-only, and the prompt says explicitly not to force alignment.
    """
    LOCKABLE = {"name", "place", "title", "term", "org", "organization"}
    locked, reference = [], []
    for k, v in (terminology or {}).items():
        target = v.get("target", "")
        if not target:
            continue
        if v.get("category", "phrase").lower() in LOCKABLE:
            locked.append((k, target, v.get("count", 0)))
        else:
            reference.append((k, target, v.get("count", 0)))

    # cross-episode memory can grow the table to thousands of entries and the prompt is finite: keep
    # the most stable ones by occurrence count.
    locked.sort(key=lambda e: -e[2])
    reference.sort(key=lambda e: -e[2])
    locked = [(k, t) for k, t, _ in locked[:DOC_REFINE_MAX_LOCKED_TERMS]]
    reference = [(k, t) for k, t, _ in reference[:DOC_REFINE_MAX_REFERENCE_TERMS]]
    return locked, reference


def _parse_doc_refine_response(resp: str, expected: int) -> tuple[dict, str]:
    """Parse a numbered-list response.

    Returns (idx -> translation, discard reason). The original per-line re.match could not
    detect a skipped, duplicated or missing number — the numbering then shifts and line A's
    translation is written onto line B. This validates the whole window first: leaving the window
    untouched is preferred over a misaligned write.
    """
    parsed, dupes, out_of_range = {}, [], []
    for line in resp.strip().split("\n"):
        m_line = re.match(r"(\d+)[\.\)]\s*(.+)", line.strip())
        if not m_line:
            continue
        idx = int(m_line.group(1)) - 1
        raw = m_line.group(2).strip()

        # the model sometimes echoes "[timestamp] "source" -> translation"; take the right side
        arrow = re.search(r"→\s*(.+)$", raw)
        if arrow:
            new_text = arrow.group(1).strip()
        else:
            new_text = re.sub(r"^\[[\d:,\.]+\]\s*", "", raw)
            new_text = re.sub(r'^"[^"]*"\s*→\s*', "", new_text)
            new_text = new_text.strip()
        new_text = _strip_quotes(new_text)
        if not new_text:
            continue

        if not (0 <= idx < expected):
            out_of_range.append(idx + 1)
            continue
        if idx in parsed:
            dupes.append(idx + 1)
            continue
        parsed[idx] = new_text

    if out_of_range:
        return {}, f"line numbers out of range: {sorted(set(out_of_range))[:5]}"
    if dupes:
        return {}, f"duplicate line numbers: {sorted(set(dupes))[:5]}"
    coverage = len(parsed) / max(expected, 1)
    if coverage < DOC_REFINE_MIN_COVERAGE:
        return {}, f"only parsed {len(parsed)}/{expected} lines ({coverage:.0%})"
    return parsed, ""


def _limit_hint(r: Result) -> str:
    """State this line's character budget in the doc_refine prompt.

    Very short / zero-duration entries (the source SRT really does contain start == end) get
    no numeric bound, otherwise a "(max 0 chars)" hint would push the model to empty output.
    """
    dur = r.segment.duration
    if dur < 1.5:
        # for a very short cue (including zero-duration entries) the computed budget is meaningless,
        # so give a non-numeric hint
        return "(short — meaning first, do not lengthen)"
    return f"(max {int(dur * MAX_CPS)} chars"


def _append_doc_trace(result: Result, record: dict):
    """Append one doc_refine record to the trace, merging duplicates into a count.

    Overlapping windows plus multiple passes mean the same subtitle is visited more than six
    times, and the same rewrite proposal is often rejected the same way repeatedly. Recording
    each one would bloat the trace, so identical (action, reason, before, after) tuples are kept
    once with an `occurrences` count.
    """
    key = ("doc_refine", record.get("action"), record.get("reason"),
           record.get("before"), record.get("after"))
    for prev in result.reasoning_trace:
        if prev.get("step") != "doc_refine":
            continue
        prev_key = ("doc_refine", prev.get("action"), prev.get("reason"),
                    prev.get("before"), prev.get("after"))
        if prev_key == key:
            prev["occurrences"] = prev.get("occurrences", 1) + 1
            prev.setdefault("rounds", [prev.get("round")])
            if record.get("round") not in prev["rounds"]:
                prev["rounds"].append(record.get("round"))
            return
    result.reasoning_trace.append(record)


def doc_refine(claude: Claude, results: list[Result], terminology: dict,
               scenes: list[Scene] = None,
               profile: ContentProfile = None,
               window_sizes: list[int] = None) -> list[Result]:
    if len(results) <= 3:
        return results

    if window_sizes is None:
        window_sizes = [5, 10, 20]

    refined = list(results)

    # Record the translation as it entered Phase 2 and emit it in the final JSONL, so that one
    # file alone distinguishes the per-segment pipeline's output from the sliding-window rewrite.
    for r in refined:
        r.translation_phase1 = r.translation

    # Per-subtitle rewrite history, used for oscillation detection and the rewrite cap.
    # Windows overlap (stride = window // 2) and several passes run, so one line is visited more
    # than six times; without this history it flips A->B->A and sets refined=True every time.
    seen_texts = [{r.translation} for r in refined]
    rewrite_counts = [0] * len(refined)

    locked, reference = _split_terminology(terminology)
    locked_str = "\n".join(f"  {k} → {t}" for k, t in locked) if locked else "  (none)"
    reference_str = ("\n".join(f"  {k} → {t}" for k, t in reference)
                     if reference else "  (none)")

    stats = {"applied": 0, "rejected_length": 0, "rejected_growth": 0,
             "rejected_oscillation": 0, "rejected_quota": 0, "windows_discarded": 0}

    for round_idx, window in enumerate(window_sizes):
        if window > len(refined):
            window = len(refined)
        stride = max(window // 2, 1)
        round_changes = 0
        logger.info(f"  Consistency round {round_idx+1}/{len(window_sizes)}: window={window}")

        for start in range(0, len(refined), stride):
            end = min(start + window, len(refined))
            current_batch = refined[start:end]
            if not current_batch:
                continue

            ctx_size = min(5, start)
            context_batch = refined[start - ctx_size:start]
            context_lines = "\n".join(
                f"  [{r.segment.start}] \"{r.source}\" → \"{r.translation}\""
                for r in context_batch
            ) if context_batch else "(start of document)"

            scene_info = ""
            if scenes:
                seg_idx = current_batch[0].segment.index
                for sc in scenes:
                    if sc.start_idx <= seg_idx <= sc.end_idx:
                        scene_info = f"\nScene: {sc.theme} (tone: {sc.tone})"
                        if sc.domain_notes:
                            scene_info += f"\nDomain: {sc.domain_notes}"
                        break

            profile_info = ""
            if profile and profile.title:
                profile_info = f"\nContent: {profile.title}"

            # put each line's character budget in front of the model instead of checking after the
            # fact
            src_tgt_lines = "\n".join(
                f"{i+1}. [{r.segment.start}] \"{r.source}\" → \"{r.translation}\""
                f"  {_limit_hint(r)}"
                for i, r in enumerate(current_batch)
            )

            msg = f"""## LOCKED TERMS (must match exactly)
{locked_str}

## REFERENCE TERMS (context-dependent — do NOT force alignment)
{reference_str}
{profile_info}

## LOCKED CONTEXT
{context_lines}
{scene_info}

## CURRENT WINDOW ({len(current_batch)} lines)
{src_tgt_lines}

---
Review for consistency and idiomatic German. Output {len(current_batch)} numbered lines:"""

            resp = claude.chat(DOC_REFINER_SYSTEM, msg)
            parsed, discard_reason = _parse_doc_refine_response(resp, len(current_batch))

            if discard_reason:
                stats["windows_discarded"] += 1
                logger.warning(
                    f"    window [{start}:{end}] discarded — {discard_reason}"
                )
                for i, r in enumerate(current_batch):
                    _append_doc_trace(r, {
                        "step": "doc_refine",
                        "round": round_idx + 1,
                        "window": [start, end],
                        "action": "window_discarded",
                        "reason": discard_reason,
                        "translation": r.translation,
                    })
                continue

            for idx, new_text in parsed.items():
                pos = start + idx
                r = refined[pos]
                old_text = r.translation
                if new_text == old_text:
                    continue

                record = {
                    "step": "doc_refine",
                    "round": round_idx + 1,
                    "window": [start, end],
                    "before": old_text,
                    "after": new_text,
                }

                # Guardrail 1: rewrite cap, so one subtitle is not overturned again and again
                if rewrite_counts[pos] >= DOC_REFINE_MAX_REWRITES:
                    stats["rejected_quota"] += 1
                    record.update(action="rejected",
                                  reason=f"rewrite quota reached "
                                         f"({DOC_REFINE_MAX_REWRITES})")
                    _append_doc_trace(r, record)
                    continue

                # Guardrail 2: oscillation detection — a value seen before means it is going in
                # circles
                if new_text in seen_texts[pos]:
                    stats["rejected_oscillation"] += 1
                    record.update(action="rejected",
                                  reason="oscillation — reverts to a previous version")
                    _append_doc_trace(r, record)
                    continue

                # Guardrail 3: length. Every Phase 1 line passed constraint_check, so a Phase 2
                # rewrite must
                # be re-checked, otherwise 12 characters become 33 crammed into 2 seconds.
                dur = r.segment.duration
                before_rep = length_report(old_text, dur)
                after_rep = length_report(new_text, dur)
                if after_rep["violations"] and not before_rep["violations"]:
                    stats["rejected_length"] += 1
                    record.update(action="rejected",
                                  reason="length constraint: "
                                         + "; ".join(after_rep["violations"]),
                                  constraint_before=before_rep,
                                  constraint_after=after_rep)
                    _append_doc_trace(r, record)
                    continue

                # Guardrail 4: no significant expansion. Short lines (<1.5s) hold the meaning-first
                # exemption
                # in constraint_check, so an absolute CPS bound will not stop 9 characters becoming
                # 17.
                grew = after_rep["chars"] - before_rep["chars"]
                if (grew > DOC_REFINE_GROWTH_GRACE_CHARS
                        and after_rep["chars"] >
                        before_rep["chars"] * DOC_REFINE_MAX_GROWTH_RATIO):
                    stats["rejected_growth"] += 1
                    record.update(action="rejected",
                                  reason=f"expansion: {before_rep['chars']}→"
                                         f"{after_rep['chars']} chars "
                                         f"(cps {before_rep['cps']}→{after_rep['cps']})",
                                  constraint_before=before_rep,
                                  constraint_after=after_rep)
                    _append_doc_trace(r, record)
                    continue

                r.translation = new_text
                r.refined = True
                r.refined_doc = True
                rewrite_counts[pos] += 1
                seen_texts[pos].add(new_text)
                round_changes += 1
                stats["applied"] += 1
                record.update(action="applied",
                              rewrite_count=rewrite_counts[pos],
                              constraint_after=after_rep)
                _append_doc_trace(r, record)

        logger.info(f"  Round {round_idx+1} done: {round_changes} changes")
        if round_changes == 0:
            break

    logger.info(
        f"  doc_refine summary: {stats['applied']} applied, "
        f"{stats['rejected_length']} rejected (length), "
        f"{stats['rejected_growth']} rejected (expansion), "
        f"{stats['rejected_oscillation']} rejected (oscillation), "
        f"{stats['rejected_quota']} rejected (quota), "
        f"{stats['windows_discarded']} windows discarded"
    )
    return refined


def _strip_quotes(text: str) -> str:
    """Remove surrounding/stray quote characters from translation output."""
    text = text.strip()
    for pair in ['""', '""', "''", '«»']:
        if len(text) >= 2 and text[0] == pair[0] and text[-1] == pair[1]:
            text = text[1:-1].strip()
    # Remove leading stray opening quotes
    if text and text[0] in ('"', '"', "'", '«'):
        text = text[1:].strip()
    return text


# Agent output that is commentary rather than a translation. Here
# the target is Latin script, so there is no "keep the non-Latin runs" rescue:
# when these fire the whole candidate is unusable and `_clean` returns "".
# A candidate reduced to "" is dropped from `valid` rather than published as a subtitle.
_LEAK_LABEL = re.compile(
    r"^\**\s*(Rationale|Reasoning|Note|Translation note|Analysis|Commentary)"
    r"\s*\**\s*[:：]", re.I)

_EN_PROSE = re.compile(
    r"\b(the|is|are|were|that|this|which|because|and|with|for|not|of|"
    r"to|he|she|they|would|could|should|from|here|there|what|when|"
    r"translation|register|line|meaning|tone|literal|choice|chosen|"
    r"English|speaker|delivery|beat|syllable|German)\b", re.I)
# note: 'was' is omitted on purpose - it is also a frequent word in the target language, and keeping
# it would reject legitimate translations


def _leak_kind(text: str) -> str:
    """Classify agent commentary that must not reach the subtitle. "" = clean.

    The rule was tuned on a full episode of real agent output: 43/43 true leaks caught, no misses,
    no false positives.

    The English function-word list deliberately omits words that are also frequent in the target
    language, so real translations do not trip it.
    """
    s = text.strip().lstrip("*_ ").lstrip()
    if s.startswith(">"):
        return "quote-block"
    if _LEAK_LABEL.match(s):
        return "label"
    if len(text) > 120 and len(_EN_PROSE.findall(text)) >= 4:
        return "en-prose"
    return ""


# ============================================================
# Series Memory  (cross-episode persistence)
# ============================================================

class SeriesMemory:
    """Persist cross-episode knowledge: character names, terminology, idiom bank.

    Stored as a single JSON file (default: series_memory_en2de.json).
    Load at episode start → inject into Pipeline → save at episode end.

    Structure:
    {
      "series": "<series id>",
      "terminology": {"<Character Name>": {"target": "<rendered name>", "category": "name", "count":
      12}, ...},
      "characters": [{"name": "...", "de": "...", "role": "..."}, ...],
      "domain_knowledge": ["...", ...],
      "idiom_bank": { <same structure as ContentProfile.idiom_bank> },
      "episodes_processed": ["S01E01", ...]
    }
    """

    def __init__(self, path: str = "series_memory_en2de.json"):
        self.path = path
        self.data: dict = {
            "series": "",
            "terminology": {},
            "characters": [],
            "domain_knowledge": [],
            "idiom_bank": {},
            "episodes_processed": [],
        }
        if os.path.exists(path):
            self._load()

    def _load(self):
        with open(self.path, encoding="utf-8") as f:
            saved = json.load(f)
        self.data.update(saved)
        logger.info(
            f"[SeriesMemory] Loaded from {self.path}: "
            f"{len(self.data['terminology'])} terms, "
            f"{len(self.data['characters'])} characters, "
            f"{len(self.data['idiom_bank'])} idiom categories, "
            f"episodes: {self.data['episodes_processed']}"
        )

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        logger.info(
            f"[SeriesMemory] Saved to {self.path}: "
            f"{len(self.data['terminology'])} terms, "
            f"{len(self.data['characters'])} characters"
        )

    def inject_into_executor(self, executor: "ToolExecutor"):
        """Pre-load all remembered terminology into a ToolExecutor."""
        count = 0
        for term, v in self.data["terminology"].items():
            key_lower = term.lower()
            if key_lower not in (k.lower() for k in executor.terminology):
                executor.terminology[term] = dict(v)
                count += 1
        logger.info(f"[SeriesMemory] Injected {count} terms into executor "
                    f"({len(executor.terminology)} total)")

    def inject_into_profile(self, profile: "ContentProfile"):
        """Enrich a ContentProfile with remembered characters, domain knowledge, idiom bank."""
        # Merge characters (avoid duplicates by name)
        existing_names = {c.get("name", "").lower() for c in profile.characters}
        added_chars = 0
        for char in self.data["characters"]:
            if char.get("name", "").lower() not in existing_names:
                profile.characters.append(char)
                existing_names.add(char.get("name", "").lower())
                added_chars += 1

        # Merge domain knowledge (dedup by text)
        existing_dk = set(profile.domain_knowledge)
        added_dk = 0
        for fact in self.data["domain_knowledge"]:
            if fact not in existing_dk:
                profile.domain_knowledge.append(fact)
                existing_dk.add(fact)
                added_dk += 1

        # Reuse idiom bank if episode didn't build one
        if not profile.idiom_bank and self.data["idiom_bank"]:
            profile.idiom_bank = dict(self.data["idiom_bank"])
            logger.info(f"[SeriesMemory] Reused idiom bank "
                        f"({len(profile.idiom_bank)} categories) — skipped rebuild")

        logger.info(f"[SeriesMemory] Injected into profile: "
                    f"+{added_chars} chars, +{added_dk} domain facts")

    def collect_from_executor(self, executor: "ToolExecutor"):
        """Merge terminology accumulated during this episode back into memory."""
        count = 0
        for term, v in executor.terminology.items():
            key_lower = term.lower()
            existing_key = next(
                (k for k in self.data["terminology"] if k.lower() == key_lower), None
            )
            if existing_key:
                # Accumulate count, keep existing translation unless count suggests update
                self.data["terminology"][existing_key]["count"] = (
                    self.data["terminology"][existing_key].get("count", 0) + v.get("count", 0)
                )
            else:
                self.data["terminology"][term] = dict(v)
                count += 1
        logger.info(f"[SeriesMemory] Collected +{count} new terms from episode "
                    f"({len(self.data['terminology'])} total)")

    def collect_from_profile(self, profile: "ContentProfile", episode_id: str = ""):
        """Merge characters, domain knowledge, and idiom bank from this episode."""
        # Characters
        existing_names = {c.get("name", "").lower() for c in self.data["characters"]}
        for char in profile.characters:
            if char.get("name", "").lower() not in existing_names:
                self.data["characters"].append(char)
                existing_names.add(char.get("name", "").lower())

        # Domain knowledge
        existing_dk = set(self.data["domain_knowledge"])
        for fact in profile.domain_knowledge:
            if fact not in existing_dk:
                self.data["domain_knowledge"].append(fact)
                existing_dk.add(fact)

        # Idiom bank: keep if memory is empty, or merge new categories
        if profile.idiom_bank:
            for cat, val in profile.idiom_bank.items():
                if cat not in self.data["idiom_bank"]:
                    self.data["idiom_bank"][cat] = val

        # Series title
        if not self.data["series"] and profile.title:
            self.data["series"] = profile.title

        # Episode tracking
        if episode_id and episode_id not in self.data["episodes_processed"]:
            self.data["episodes_processed"].append(episode_id)


# ============================================================
# Pipeline
# ============================================================

class Pipeline:
    def __init__(self, config_path: str = None, series_memory: "SeriesMemory" = None):
        self.claude = Claude()
        self.executor = ToolExecutor(claude_client=self.claude)
        if config_path and os.path.exists(config_path):
            self.router = Router.load(config_path)
            logger.info(f"Loaded config: {config_path}")
        else:
            self.router = Router()
            logger.info("Using default config")
        self.scenes: list[Scene] = []
        self.profile: Optional[ContentProfile] = None
        self.series_memory: Optional[SeriesMemory] = series_memory

    def translate_segment(self, ctx: Context) -> Result:
        source = ctx.segment.text
        self.executor.set_context(ctx)

        # Build reasoning trace
        reasoning_trace = []

        category = self.router.classify(ctx)
        agent_defs = self.router.get_agents(category, ctx)
        logger.info(f"[{ctx.segment.index}] \"{source[:40]}\" → {category} "
                    f"→ agents: {[a['name'] for a in agent_defs]}")

        reasoning_trace.append({
            "step": "classification",
            "category": category,
            "agents_selected": [a["name"] for a in agent_defs],
            "features": ctx.features
        })

        candidates = []
        for adef in agent_defs:
            user_msg = self._build_msg(ctx)
            tool_schemas = [s for s in TOOL_SCHEMAS if s["name"] in adef["tools"]]

            text, tools_called = self.claude.tool_loop(
                system=adef["prompt"],
                user_msg=user_msg,
                tools=tool_schemas,
                executor=self.executor.execute,
            )
            text = self._clean(text)
            candidates.append(Candidate(
                text=text, agent=adef["name"],
                tools_called=[t["tool"] for t in tools_called],
            ))
            logger.info(f"  [{adef['name']}] \"{text[:30]}\" "
                        f"tools={[t['tool'] for t in tools_called]}")

            # Record reasoning: what tools were called and what they returned
            reasoning_trace.append({
                "step": "agent_translation",
                "agent": adef["name"],
                "translation": text,
                "tools_called": tools_called  # Full tool call history with inputs & outputs
            })

        # Filter empty candidates
        valid = [c for c in candidates if c.text.strip()]
        if not valid:
            return Result(segment=ctx.segment, source=source,
                          translation="[translation failed]", agent="none",
                          reasoning_trace=reasoning_trace)
        candidates = valid

        candidates = judge_candidates(
            self.claude, source, candidates, ctx.preceding_tgt,
            ctx.segment.duration, ctx.scene, ctx.profile
        )
        best = candidates[0]
        logger.info(f"  → best: [{best.agent}] score={best.score}")

        # Low-score rescue: best < 6.0 → retry faithful agent with critique, re-judge
        if best.score < 6.0:
            logger.warning(f"  ⚠ Low score ({best.score:.1f}) — rescue retry")
            critique = best.detail.get("critique", "")
            faithful_def = next((a for a in agent_defs if a["name"] == "faithful"), agent_defs[0])
            rescue_msg = (self._build_msg(ctx) +
                          f"\n\n[PRIOR ATTEMPT SCORE: {best.score:.1f}/10]\n"
                          f"[CRITIQUE: {critique}]\n"
                          "Fix these issues. Produce a better translation.")
            rescue_schemas = [s for s in TOOL_SCHEMAS if s["name"] in faithful_def["tools"]]
            r_text, r_tools = self.claude.tool_loop(
                system=faithful_def["prompt"],
                user_msg=rescue_msg,
                tools=rescue_schemas,
                executor=self.executor.execute,
            )
            r_text = self._clean(r_text)
            if r_text.strip():
                rescue_cand = Candidate(text=r_text, agent="faithful_rescue",
                                        tools_called=[t["tool"] for t in r_tools])
                logger.info(f"  [faithful_rescue] \"{r_text[:30]}\"")
                reranked = judge_candidates(
                    self.claude, source, [rescue_cand] + candidates[:2],
                    ctx.preceding_tgt, ctx.segment.duration, ctx.scene, ctx.profile
                )
                best = reranked[0]
                logger.info(f"  → rescue best: [{best.agent}] score={best.score}")
                candidates = reranked

        reasoning_trace.append({
            "step": "judge_scoring",
            "best_agent": best.agent,
            "best_score": best.score,
            "all_scores": [(c.agent, c.score) for c in candidates],
            "critique": best.detail
        })

        final = refine(self.claude, source, best, ctx.preceding_tgt,
                       ctx.segment.duration,
                       scene_tone=ctx.scene.tone if ctx.scene else "")
        final = self._clean(final)

        # per-segment refine is length-checked too; if it grew, fall back to the judge's candidate
        if final != best.text:
            rep_before = length_report(best.text, ctx.segment.duration)
            rep_after = length_report(final, ctx.segment.duration)
            if rep_after["violations"] and not rep_before["violations"]:
                logger.warning(
                    f"  refine() rejected (length): "
                    f"{'; '.join(rep_after['violations'])}"
                )
                reasoning_trace.append({
                    "step": "refinement",
                    "before": best.text,
                    "after": best.text,
                    "refined": False,
                    "rejected_candidate": final,
                    "rejected_reason": "length constraint: "
                                       + "; ".join(rep_after["violations"]),
                })
                final = best.text
            else:
                reasoning_trace.append({
                    "step": "refinement",
                    "before": best.text,
                    "after": final,
                    "refined": True,
                })
        else:
            reasoning_trace.append({
                "step": "refinement",
                "before": best.text,
                "after": final,
                "refined": False,
            })

        self.executor.add_to_memory(source, final)
        self.router.record(category, best.agent, best.score,
                           ctx.features, best.tools_called)

        segment_refined = final != best.text
        return Result(
            segment=ctx.segment, source=source, translation=final,
            agent=best.agent, score=best.score, tools=best.tools_called,
            refined=segment_refined, refined_segment=segment_refined,
            reasoning_trace=reasoning_trace
        )

    # ---- Continuation-sentence grouping ----

    SEP_MARKER = "|||SEP|||"
    _MAX_GROUP_SIZE = 3

    @staticmethod
    def _is_continuation(seg1: Segment, seg2: Segment) -> bool:
        """Return True if seg2 continues an unfinished sentence from seg1.

        Signals:
        - seg1 ends with a comma
        - seg1 has no terminal punctuation (.!?…) AND seg2 starts with a lowercase letter
        """
        t1 = seg1.text.strip()
        t2 = seg2.text.strip()
        if not t1 or not t2:
            return False
        # Strip trailing close-quotes to expose the real terminal punctuation
        stripped = t1.rstrip('"\'”’»').strip()
        if not stripped:
            return False
        last_char = stripped[-1]
        first_char = t2[0]
        if last_char == ',':
            return True
        if last_char not in '.!?…' and first_char.islower():
            return True
        return False

    def _group_segments(self, segments: list[Segment]) -> list[list[Segment]]:
        """Group consecutive continuation segments (max _MAX_GROUP_SIZE per group)."""
        if not segments:
            return []
        groups: list[list[Segment]] = []
        current = [segments[0]]
        for i in range(1, len(segments)):
            if (len(current) < self._MAX_GROUP_SIZE
                    and self._is_continuation(segments[i - 1], segments[i])):
                current.append(segments[i])
            else:
                groups.append(current)
                current = [segments[i]]
        groups.append(current)
        merged = sum(1 for g in groups if len(g) > 1)
        logger.info(f"  Segment grouping: {len(segments)} segs → {len(groups)} groups "
                    f"({merged} multi-segment groups)")
        return groups

    def _translate_group(self, group: list[Segment], group_start_i: int,
                         all_segments: list[Segment],
                         past_tgt: list[str],
                         scene_map: dict) -> list[Result]:
        """Translate a multi-segment continuation group as one sentence, then split back."""
        first_seg, last_seg = group[0], group[-1]
        n = len(group)
        scene = scene_map.get(first_seg.index)

        # Virtual merged segment for Context (covers full span)
        combined_text = " ".join(seg.text for seg in group)
        virtual_seg = Segment(
            index=first_seg.index,
            start=first_seg.start,
            end=last_seg.end,
            text=combined_text,
            scene_id=first_seg.scene_id,
            scene_theme=first_seg.scene_theme,
        )
        i = group_start_i
        ctx = Context(
            segment=virtual_seg,
            preceding_src=all_segments[max(0, i - SLIDING_WINDOW):i],
            succeeding_src=all_segments[i + n:i + n + SLIDING_WINDOW],
            preceding_tgt=list(past_tgt[-SLIDING_WINDOW:]),
            scene=scene,
            profile=self.profile,
        )
        self.executor.set_context(ctx)

        user_msg = self._build_msg_for_group(group, ctx)
        category = self.router.classify(ctx)
        agent_defs = self.router.get_agents(category, ctx)
        adef = next((a for a in agent_defs if a["name"] == "faithful"), agent_defs[0])
        tool_schemas = [s for s in TOOL_SCHEMAS if s["name"] in adef["tools"]]

        raw_text, tools_called = self.claude.tool_loop(
            system=adef["prompt"],
            user_msg=user_msg,
            tools=tool_schemas,
            executor=self.executor.execute,
        )

        parts = self._parse_group_translation(raw_text, group)
        if parts is None:
            # The whole-sentence translation cannot be split into n parts (the model omitted the
            # separator, or the translation is too short to fill n subtitles). Forcing a split ships
            # duplicated or truncated subtitles, so fall back to translating each segment on its
            # own: cross-segment word-order optimisation is lost, but every subtitle gets real
            # content.
            logger.error(
                f"  [group ×{n}] split failed for \"{combined_text[:50]}\" — "
                f"falling back to per-segment translation"
            )
            fallback, local_tgt = [], list(past_tgt)
            for k, seg in enumerate(group):
                j = group_start_i + k
                seg_ctx = Context(
                    segment=seg,
                    preceding_src=all_segments[max(0, j - SLIDING_WINDOW):j],
                    succeeding_src=all_segments[j + 1:j + 1 + SLIDING_WINDOW],
                    preceding_tgt=list(local_tgt[-SLIDING_WINDOW:]),
                    scene=scene_map.get(seg.index),
                    profile=self.profile,
                )
                r = self.translate_segment(seg_ctx)
                fallback.append(r)
                local_tgt.append(r.translation)
            return fallback

        logger.info(f"  [group ×{n}] \"{combined_text[:50]}\" → {parts}")

        results = []
        for seg, de in zip(group, parts):
            self.executor.add_to_memory(seg.text, de)
            results.append(Result(
                segment=seg,
                source=seg.text,
                translation=de,
                agent=f"{adef['name']}_group",
                score=7.0,
                tools=[t["tool"] for t in tools_called],
            ))
        return results

    def _build_msg_for_group(self, group: list[Segment], ctx: Context) -> str:
        """Build translation prompt for a multi-segment continuation group."""
        combined = " ".join(seg.text for seg in group)
        boundary_view = self.SEP_MARKER.join(seg.text for seg in group)
        n = len(group)
        total_dur = Segment._sec(group[-1].end) - Segment._sec(group[0].start)

        parts = [
            f"Translate this subtitle passage into German:",
            f"**Source ({n} consecutive English segments forming one sentence):** {combined}",
            f"",
            f"These {n} subtitle lines are one sentence split across timestamps.",
            f"Original segment boundaries (marked with {self.SEP_MARKER}):",
            f"  {boundary_view}",
            f"",
            f"Translate the full sentence naturally into German, then SPLIT your translation",
            f"to match each original segment boundary using {self.SEP_MARKER} as the separator.",
            f"You MUST output exactly {n - 1} {self.SEP_MARKER} marker(s) in your output.",
            f"",
            f"**Time:** {group[0].start} → {group[-1].end} (total {total_dur:.2f}s)",
        ]

        if ctx.profile and ctx.profile.title:
            parts.append(f"**Content:** {ctx.profile.title} ({ctx.profile.media_type})")
            if ctx.profile.setting:
                parts.append(f"**Setting:** {ctx.profile.setting}")
            if ctx.profile.domain_knowledge:
                parts.append("**Series facts:** " + " | ".join(ctx.profile.domain_knowledge[:6]))

        if ctx.scene:
            parts.append(f"**Scene:** {ctx.scene.theme} (tone: {ctx.scene.tone})")
            if ctx.scene.description:
                parts.append(f"**Scene context:** {ctx.scene.description}")
            if ctx.scene.domain_notes:
                parts.append(f"**Domain Notes:** {ctx.scene.domain_notes}")

        if ctx.preceding_src:
            parts.append("\n**Previous lines (English):**")
            for s in ctx.preceding_src[-3:]:
                parts.append(f"  {s.text}")
        if ctx.preceding_tgt:
            parts.append("**Previous translations (German):**")
            for t in ctx.preceding_tgt[-3:]:
                parts.append(f"  → {t}")
        if ctx.succeeding_src:
            parts.append("**Next lines (English):**")
            for s in ctx.succeeding_src[:2]:
                parts.append(f"  {s.text}")

        return "\n".join(parts)

    def _parse_group_translation(self, raw: str, group: list[Segment]) -> list[str] | None:
        """Parse |||SEP||| delimited output; fall back to proportional split.

        Returns None when the group cannot be split, in which case the caller should translate
        segment by segment. This used to force a result out instead (see _proportional_split), which
        is worse than no result.
        """
        leak = _leak_kind(raw)
        if leak:
            logger.error(f"  [group] agent returned commentary ({leak}), "
                         f"output will be unreliable: {raw[:120]!r}")
        text = self._clean_group(raw)
        n = len(group)
        parts = [self._clean_group(p) for p in text.split(self.SEP_MARKER)]
        if len(parts) == n and all(parts):
            return parts
        logger.warning(
            f"Group split: expected {n} parts, got {len(parts)} "
            f"from '{text[:60]}'. Using proportional fallback."
        )
        # May return None — the caller then re-translates the group cue by cue.
        return self._proportional_split(text.replace(self.SEP_MARKER, " "), group)

    def _clean_group(self, text: str) -> str:
        """The lighter clean the group path needs: quotes, one label prefix, markdown.

        `_clean` is the wrong tool here, and the difference is not stylistic. `_clean` finishes by
        keeping only the last non-commentary LINE, which is correct for a single cue and wrong for
        a group answer: the separator-delimited parts of a group are not commentary, so that step
        can reduce a perfectly good part to "". `_parse_group_translation` then sees a falsy part,
        `all(parts)` fails, and a group that split correctly is thrown away — the caller
        re-translates it cue by cue, which costs one model call per cue and loses exactly the
        cross-cue coherence the group prompt exists to buy.

        So this strips the same wrappers and stops. It never returns "" for non-empty input, which
        matters because `_translate_group` publishes `parts` with no validity filter of its own.
        """
        text = text.strip()
        text = _strip_quotes(text)
        text = re.sub(r"^\*{1,2}([^*]{0,40}?)\*{1,2}\s*[:：]\s*", r"\1: ", text).strip()
        for p in ["Übersetzung:", "German:", "Deutsch:", "Output:", "Translation:",
                  "Final:", "Ergebnis:", "Translation note:", "Note:"]:
            if text.lower().startswith(p.lower()):
                text = text[len(p):].strip()
                break
        text = text.replace("**", "").strip()
        return _strip_quotes(text).strip()

    def _proportional_split(self, text: str, group: list[Segment]) -> list[str] | None:
        """Split into len(group) parts proportionally to source length; return None if not
        splittable.

        Two bugs used to stack here and disguise "not splittable" as "split successfully":

          1. _find_split_pos scanned 15 characters right of target looking for punctuation; on a
             short translation it ran to the end of the sentence, so the first part swallowed the
             whole line and the rest were empty strings.
          2. The closing `return [p or text for p in parts]` replaced each empty string with the
                ENTIRE sentence, so n parts became n identical subtitles.

        Observed case: segments 272-274 of one episode form a single sentence spanning three
        subtitles. The model emitted no separator and the translation collapsed to a 10-character
        clause, dropping two of the source clauses entirely. The old code published those 10
        characters three times over.

        The test is not a length threshold (arbitrary — 10 characters would still pass a 3-way
        split) but CLAUSE BOUNDARIES: a translation can only be spread over n subtitles without
        cutting a clause if it has enough internal punctuation.
        A single clause with no internal punctuation is a fragment however it is cut -> None.
        A sentence with two internal commas is exactly three clauses -> one clause per part.

        """
        if len(group) == 1:
            return [text]
        n = len(group)

        # cut points: positions after punctuation (excluding the final one — sentence end is not an
        # internal boundary)
        bounds = [i for i in range(1, len(text)) if text[i - 1] in ",;:.!?"]
        if len(bounds) < n - 1:
            logger.warning(
                f"Group split: only {len(bounds)} clause boundaries for {n} "
                f"segments — cannot split without cutting mid-clause: '{text[:40]}'"
            )
            return None

        src_lens = [len(seg.text) for seg in group]
        total_src = sum(src_lens)
        parts: list[str] = []
        pos = 0
        for k, seg in enumerate(group[:-1]):
            target = pos + round(len(text) * len(seg.text) / total_src)
            # (n-2-k) more cuts follow this one, so leave enough boundaries for them
            avail = bounds[:len(bounds) - (n - 2 - k)] if n - 2 - k else bounds
            avail = [b for b in avail if b > pos]
            if not avail:
                return None
            cut = min(avail, key=lambda b: abs(b - target))
            bounds = [b for b in bounds if b > cut]
            parts.append(text[pos:cut].strip())
            pos = cut
        parts.append(text[pos:].strip())

        if not all(parts) or len(set(parts)) != n:
            logger.warning(
                f"Group split: produced empty or duplicate parts for {n} "
                f"segments: {parts}"
            )
            return None
        return parts

    def translate_file(self, source_path: str, output_path: str = None,
                       max_segments: int = None,
                       episode_id: str = "") -> list[Result]:
        try:
            segments = parse_srt(source_path)
            logger.info(f"Parsed {len(segments)} segments from {Path(source_path).name}")
            if max_segments:
                segments = segments[:max_segments]

            # Phase -1: Deep content research
            self.profile = deep_research(self.claude, segments, source_path)

            # Inject cross-episode memory before seeding executor
            if self.series_memory:
                self.series_memory.inject_into_profile(self.profile)
                self.series_memory.inject_into_executor(self.executor)

            self.executor.seed_from_profile(self.profile)

            # Phase 0: Scene segmentation (with domain_notes)
            logger.info("Phase 0: Scene segmentation...")
            self.scenes = segment_into_scenes(self.claude, segments, self.profile)

            # Build scene lookup
            scene_map = {}
            for scene in self.scenes:
                for idx in range(scene.start_idx, scene.end_idx + 1):
                    scene_map[idx] = scene

            # Phase 1: Translate each segment (with continuation-sentence grouping)
            groups = self._group_segments(segments)
            results = []
            past_tgt = []
            jsonl_path = output_path.replace(".srt", ".jsonl") if output_path else None
            if jsonl_path:
                os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
                _jsonl_f = open(jsonl_path, "w", encoding="utf-8")
            else:
                _jsonl_f = None
            # Build index: segment → flat position in segments list
            seg_to_flat_idx = {seg.index: i for i, seg in enumerate(segments)}
            try:
                for group in groups:
                    group_start_i = seg_to_flat_idx[group[0].index]
                    if len(group) > 1:
                        group_results = self._translate_group(
                            group, group_start_i, segments, past_tgt, scene_map
                        )
                    else:
                        seg = group[0]
                        i = group_start_i
                        scene = scene_map.get(seg.index)
                        ctx = Context(
                            segment=seg,
                            preceding_src=segments[max(0, i - SLIDING_WINDOW):i],
                            succeeding_src=segments[i+1:i+1+SLIDING_WINDOW],
                            preceding_tgt=list(past_tgt[-SLIDING_WINDOW:]),
                            scene=scene,
                            profile=self.profile,
                        )
                        group_results = [self.translate_segment(ctx)]

                    for result in group_results:
                        results.append(result)
                        past_tgt.append(result.translation)
                        if _jsonl_f:
                            _jsonl_f.write(json.dumps({
                                "idx": result.segment.index,
                                "start": result.segment.start,
                                "end": result.segment.end,
                                "source": result.source,
                                "translation": result.translation,
                                "agent": result.agent,
                                "score": result.score,
                                "tools": result.tools,
                                "refined": result.refined,
                                "refined_segment": result.refined_segment,
                                "refined_doc": result.refined_doc,
                                "scene_theme": result.segment.scene_theme,
                                "reasoning_trace": result.reasoning_trace,
                            }, ensure_ascii=False) + "\n")
                            _jsonl_f.flush()
                    time.sleep(0.2)
            finally:
                if _jsonl_f:
                    _jsonl_f.close()

            # Phase 2: Document-level consistency
            logger.info("Phase 2: Document-level refinement...")
            results = doc_refine(
                self.claude, results, self.executor.terminology,
                self.scenes, self.profile
            )

            # Save cross-episode memory
            if self.series_memory:
                self.series_memory.collect_from_executor(self.executor)
                self.series_memory.collect_from_profile(self.profile, episode_id)
                self.series_memory.save()

            if output_path:
                self._write_srt(results, output_path)
            return results

        finally:
            # Cleanup: Close Selenium WebDriver if used
            logger.info("Cleaning up search backend...")
            cleanup_scraper()

    def __del__(self):
        """Cleanup when Pipeline is destroyed."""
        cleanup_scraper()

    def _build_msg(self, ctx: Context) -> str:
        parts = [f"Translate this subtitle line into German:\n**Source (English):** {ctx.segment.text}"]
        parts.append(f"**Time:** {ctx.segment.start} → {ctx.segment.end} "
                     f"(duration: {ctx.segment.duration:.2f}s)")

        if ctx.profile and ctx.profile.title:
            parts.append(f"**Content:** {ctx.profile.title} ({ctx.profile.media_type})")
            if ctx.profile.setting:
                parts.append(f"**Setting:** {ctx.profile.setting}")
            if ctx.profile.domain_knowledge:
                parts.append("**Series facts:** " + " | ".join(ctx.profile.domain_knowledge[:6]))

        if ctx.scene:
            parts.append(f"**Scene:** {ctx.scene.theme} (tone: {ctx.scene.tone})")
            if ctx.scene.description:
                parts.append(f"**Scene context:** {ctx.scene.description}")
            if ctx.scene.domain_notes:
                parts.append(f"**Domain Notes:** {ctx.scene.domain_notes}")

        if ctx.segment.speaker:
            parts.append(f"**Speaker:** {ctx.segment.speaker}")
        if ctx.preceding_src:
            parts.append("\n**Previous lines (English):**")
            for s in ctx.preceding_src[-3:]:
                parts.append(f"  {s.text}")
        if ctx.preceding_tgt:
            parts.append("**Previous translations (German):**")
            for t in ctx.preceding_tgt[-3:]:
                parts.append(f"  → {t}")
        if ctx.succeeding_src:
            parts.append("**Next lines (English):**")
            for s in ctx.succeeding_src[:2]:
                parts.append(f"  {s.text}")
        return "\n".join(parts)

    def _clean(self, text: str) -> str:
        text = text.strip()
        text = _strip_quotes(text)

        # Strip markdown emphasis around a leading label, e.g. "**Translation:**"
        text = re.sub(r"^\*{1,2}([^*]{0,40}?)\*{1,2}\s*[:：]\s*", r"\1: ", text).strip()

        for p in ["Übersetzung:", "German:", "Deutsch:", "Output:", "Translation:",
                  "Final:", "Ergebnis:", "Translation note:", "Note:"]:
            if text.lower().startswith(p.lower()):
                text = text[len(p):].strip()
                break

        # If agent leaked reasoning (English prose + German on a new line), take the
        # last non-empty line — skipping lines that are themselves commentary.
        if "\n" in text:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            keep = [l for l in lines if not _leak_kind(l)]
            if keep:
                text = keep[-1]
            elif lines:
                text = lines[-1]

        # Drop markdown-bold residue: it both shows up literally in the SRT and
        # inflates the char count into false CPS violations.
        text = text.replace("**", "").strip()
        text = _strip_quotes(text)
        text = text.rstrip('"').rstrip('"').strip()

        # Unrecoverable commentary: return "" and let translate_segment discard the
        # candidate rather than publish reasoning as a subtitle.
        kind = _leak_kind(text)
        if kind:
            logger.warning(f"  _clean dropped candidate ({kind}): {text[:80]!r}")
            return ""

        return text

    def _write_srt(self, results: list[Result], out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # Write the SRT file. Published text is line-wrapped; `translation` in the JSONL stays on a
        # single line verbatim (resume and diffing rely on that), and the `translation_srt` field
        # records the relationship between the two explicitly.
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"{r.segment.index}\n")
                f.write(f"{r.segment.start} --> {r.segment.end}\n")
                f.write(f"{wrap_subtitle(r.translation)}\n\n")

        # Write JSONL with full reasoning trace
        # Strip one trailing ".srt" rather than replacing the substring. `out_path`
        # need not end in one - `run_one_srt.py --out` takes any path - and
        # `str.replace` is then a no-op that points all three writers at the same
        # file, so each overwrites the last and only one artefact survives. The two
        # it loses are the per-segment record and the reasoning trace, which is to
        # say the entire evidence a run produces beyond the subtitles themselves.
        # `replace` is also unbounded: it rewrites ".srt" anywhere in the path,
        # including a parent directory named for the corpus it came from.
        stem = out_path[:-4] if out_path.endswith(".srt") else out_path
        jsonl = stem + ".jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps({
                    "idx": r.segment.index,
                    "start": r.segment.start,
                    "end": r.segment.end,
                    "source": r.source,
                    "translation": r.translation,
                    "translation_srt": wrap_subtitle(r.translation),
                    "translation_phase1": r.translation_phase1 or r.translation,
                    "agent": r.agent,
                    "score": r.score,
                    "tools": r.tools,
                    "refined": r.refined,
                    "refined_segment": r.refined_segment,
                    "refined_doc": r.refined_doc,
                    "scene_theme": r.segment.scene_theme,
                    "reasoning_trace": r.reasoning_trace,  # FULL reasoning trace
                }, ensure_ascii=False) + "\n")

        # Write human-readable reasoning trace
        trace_file = stem + "_reasoning.jsonl"
        with open(trace_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"\n{'='*80}\n")
                f.write(f"Segment {r.segment.index}: {r.source}\n")
                f.write(f"Translation: {r.translation}\n")
                if r.refined_doc and r.translation_phase1:
                    f.write(f"  (Phase 1 original: {r.translation_phase1})\n")
                f.write(f"{'='*80}\n\n")
                for step in r.reasoning_trace:
                    f.write(json.dumps(step, ensure_ascii=False, indent=2) + "\n\n")


        logger.info(f"Written: {out_path}, {jsonl}, {trace_file}")
        self._audit_traces(results)

    @staticmethod
    def _audit_traces(results: list):
        """Does the trace's last translation match the one that actually shipped?

        Phase 2 rewrites a translation after Phase 1 has already written its trace, so a trace can
        end on a string that was never published — which is the difference between a reasoning
        chain that documents the system and one that misrepresents it. Drift must be zero, and
        this is what says so in the log rather than leaving it to be assumed.
        """
        drift = []
        for r in results:
            last = None
            for step in r.reasoning_trace:
                if step.get("step") == "doc_refine":
                    if step.get("action") == "applied":
                        last = step.get("after")
                elif step.get("step") == "refinement":
                    last = step.get("after")
            if last is not None and last != r.translation:
                drift.append(r.segment.index)

        if drift:
            logger.warning(f"[trace audit] {len(drift)} segments whose trace does not "
                           f"match the shipped translation: {drift[:20]}")
        else:
            logger.info(f"[trace audit] OK — traces match shipped translations "
                        f"for all {len(results)} segments")
