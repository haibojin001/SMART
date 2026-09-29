"""Pairing a source episode with a candidate translation, and cutting it into judging windows.

SubMQM judges a *window of consecutive subtitles*, so before anything can be scored the source
and the candidate have to be put side by side. How that is done depends on where the candidate
came from, and the distinction is not cosmetic:

    Model hypotheses preserve the source segmentation. Every system under evaluation
    translates block n into block n, so the pairing is 1:1 and needs no matching at all. This
    is `block` mode.

    Human and community reference subtitles do not. They were authored independently, so one
    source block may correspond to half a reference block or to two of them. Charging that as
    Undertranslation would measure the segmentation, not the translation. So references are
    aligned to the source by *timestamp overlap* and judged at the *passage* level: the window
    is still a run of consecutive source cues, but the candidate side is the text that occupies
    the same stretch of time, presented as running text. This is `passage` mode.

    References also carry storage artefacts. Corpus-derived subtitles are often tokenised -
    `we don ' t` , `我 不 知 道` , a space before every comma - which a scorer would read as
    Spacing Error and Mispunctuation. Those are properties of the file format, not of the
    translation, so they are normalised away first. Normalisation is reported, never silent:
    `normalise_cues` returns how many cues it touched, and `--no-normalise` turns it off.

The technical dimension is the one place passage mode does not go coarse. Characters-per-line
and lines-per-box are properties of an actual displayed cue, so a passage-mode window carries
its candidate's own cues alongside the running text, and the judge is shown both.

This module deliberately parses SRT itself instead of importing a core module. Evaluation must
run on a machine with no model client and no credentials; importing a core module constructs
one at module scope.

    python3 align.py source.srt hypothesis.srt --mode block --window 20
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# Window geometry. The paper states that judging happens over a sliding window of consecutive
# subtitles but never states the size, so it is a config value here and every result file
# records what it actually used. Deliberately NOT 5/10/20-at-50%: those are the translation
# side's consistency editor, a different mechanism, and reusing its numbers here would imply a
# coupling the method does not have.
DEFAULT_WINDOW = 20
DEFAULT_OVERLAP = 0.0


# --------------------------------------------------------------------------------
# cues
# --------------------------------------------------------------------------------

@dataclass
class Cue:
    index: int
    start_ms: int
    end_ms: int
    lines: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.lines).strip()

    @property
    def block(self) -> str:
        """The cue as displayed, line breaks intact - what the technical checks look at."""
        return "\n".join(self.lines)

    @property
    def duration(self) -> float:
        return max(0.0, (self.end_ms - self.start_ms) / 1000.0)

    def overlaps(self, start_ms: int, end_ms: int) -> int:
        """Milliseconds of overlap with a time span. 0 means disjoint."""
        return max(0, min(self.end_ms, end_ms) - max(self.start_ms, start_ms))


_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")
_ARROW = re.compile(r"(\d+:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d+:\d{2}:\d{2}[,.]\d{1,3})")
# Inline markup an SRT may carry. Stripped for judging: a candidate is not penalised for
# italics, and a source is not judged on them either.
_TAGS = re.compile(r"</?[a-zA-Z][^>]*>|\{\\[^}]*\}")


def parse_timestamp(s: str) -> int:
    m = _TIME.fullmatch(s.strip())
    if not m:
        raise ValueError(f"not a timestamp: {s!r}")
    h, mi, sec, frac = m.groups()
    return ((int(h) * 60 + int(mi)) * 60 + int(sec)) * 1000 + int(frac.ljust(3, "0"))


def format_timestamp(ms: int) -> str:
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3_600_000)
    mi, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{mi:02d}:{s:02d},{msec:03d}"


def parse_srt(path: str | Path, *, strip_tags: bool = True) -> list[Cue]:
    """Tolerant SRT reader.

    Tolerant on purpose: the inputs are a mix of this system's own output, other systems'
    output and corpus-derived files, and a strict parser that rejects a missing index or a
    stray blank line would simply refuse to evaluate half the benchmark. It keys on the
    `-->` line rather than on the numbering, so a file with no indices, duplicated indices or
    Windows line endings still reads. A cue with no text is dropped - it carries nothing to
    judge - and cues are returned in presentation order.
    """
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: list[Cue] = []
    pending_lines: list[str] = []
    span: tuple[int, int] | None = None

    def flush() -> None:
        nonlocal span, pending_lines
        if span is not None:
            text = [ln for ln in pending_lines if ln.strip()]
            if text:
                cues.append(Cue(len(cues) + 1, span[0], span[1], text))
        span, pending_lines = None, []

    for i, line in enumerate(lines):
        m = _ARROW.search(line)
        if m:
            # The bare number directly above a timestamp is the *next* cue's index, so it has
            # been accumulating as the *previous* cue's last line of text. Drop it before
            # flushing. Keyed on adjacency to the arrow rather than on "looks like a number",
            # so a cue whose text genuinely is `1999` survives as long as it is not the line
            # immediately above the next cue's timestamp - and if it is, no parser can tell.
            if pending_lines and lines[i - 1].strip().isdigit():
                pending_lines.pop()
            flush()
            span = (parse_timestamp(m.group(1)), parse_timestamp(m.group(2)))
            continue
        if span is None:
            continue                          # the first cue's index line, and any preamble
        stripped = _TAGS.sub("", line) if strip_tags else line
        pending_lines.append(stripped.rstrip())
    flush()

    if not cues:
        raise ValueError(f"no cues parsed from {path}")
    cues.sort(key=lambda c: (c.start_ms, c.end_ms))
    for i, c in enumerate(cues, 1):
        c.index = i
    return cues


# --------------------------------------------------------------------------------
# reference normalisation
# --------------------------------------------------------------------------------

# Spaces that a tokeniser inserted, which the orthography does not have. Matched as `[ \t]`
# and never as `\s`, so a non-breaking space survives: French sets a narrow no-break space
# before `;:!?»` on purpose, and an ASCII space in the same position is the artefact. Keeping
# the two distinct is the difference between de-tokenising a file and re-typesetting it.
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?%)\]}»”’…])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[{«“‘])[ \t]+")
# Apostrophes get two narrow rules rather than one broad one. A broad `\s*'\s*` would also eat
# the spaces around a single-quoted phrase (`he said ' hello '`), turning a quotation into a
# contraction. These two match only the shapes a tokeniser actually produces: an English
# clitic split off its host, and a Romance one- or two-letter elision.
_SPACED_CLITIC = re.compile(r"(?<=\w)[ \t]*['´’][ \t]*(?=(?:s|t|ll|re|ve|m|d)\b)",
                            re.IGNORECASE)
_SPACED_ELISION = re.compile(r"\b(\w{1,2})[ \t]*['´’][ \t]+(?=\w)")
_SPACED_HYPHEN = re.compile(r"(?<=\w)[ \t]+-[ \t]+(?=\w)")
_MULTISPACE = re.compile(r"[ \t]{2,}")
# Two CJK characters separated by an ASCII space. Han and kana write no word spaces at all, so
# any space between two of them came from tokenisation. The class includes CJK punctuation and
# the fullwidth forms, because the artefact `我 不 知 道 ， 去 问` puts spaces around the comma
# too, and a rule that only knew about ideographs would leave `意思 ， 去问` - which then reads
# as a Spacing Error, the exact thing this is here to prevent.
#
# Hangul is excluded, deliberately. Korean *does* space between words, so stripping those
# spaces would manufacture the error rather than remove it; a tokeniser that split `사랑을` into
# `사랑 을` therefore survives normalisation. That is a stated limitation, not an oversight -
# telling a particle boundary from a word boundary needs a lexicon, and guessing wrong corrupts
# the reference in a way the judge would read as the translator's fault.
_CJK = ("\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
        "\uf900-\ufaff\ufe10-\ufe19\uff01-\uff60\U00020000-\U0002ffff")
_SPACED_CJK = re.compile(f"(?<=[{_CJK}])[ \t]+(?=[{_CJK}])")
# A dialogue dash opening a line is orthography, not an artefact, and must survive the
# hyphen rule. Matched before it, so it is protected.
_LEADING_DASH = re.compile(r"^\s*[-–—]\s*")


def normalise_text(text: str) -> str:
    """Remove storage artefacts from one line of reference text.

    Order matters: the leading dialogue dash is protected before the word-internal hyphen rule
    runs, and the CJK rule runs before whitespace collapsing so `我 不 知 道` is joined rather
    than merely single-spaced.
    """
    lead = _LEADING_DASH.match(text)
    prefix, body = (lead.group(0), text[lead.end():]) if lead else ("", text)
    body = unicodedata.normalize("NFC", body)
    while True:      # `我 不 知 道`: one pass may leave alternate gaps, so iterate to a fixpoint
        joined = _SPACED_CJK.sub("", body)
        if joined == body:
            break
        body = joined
    body = _SPACE_BEFORE_PUNCT.sub(r"\1", body)
    body = _SPACE_AFTER_OPEN.sub(r"\1", body)
    body = _SPACED_CLITIC.sub("'", body)
    body = _SPACED_ELISION.sub(r"\1'", body)
    body = _SPACED_HYPHEN.sub("-", body)
    body = _MULTISPACE.sub(" ", body)
    return (prefix + body).strip()


def normalise_cues(cues: list[Cue]) -> tuple[list[Cue], int]:
    """Normalise every cue's text. Returns the new cues and how many changed.

    The count is returned rather than logged because it belongs in the result file: a run where
    normalisation touched 90% of the reference's cues is telling you something about the corpus,
    and a run where it touched none means the flag did nothing.
    """
    out, changed = [], 0
    for c in cues:
        lines = [normalise_text(ln) for ln in c.lines]
        lines = [ln for ln in lines if ln]
        if lines != c.lines:
            changed += 1
        out.append(Cue(c.index, c.start_ms, c.end_ms, lines or c.lines))
    return out, changed


# --------------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------------

@dataclass
class Unit:
    """One source cue and whatever of the candidate corresponds to it."""
    source: Cue
    targets: list[Cue] = field(default_factory=list)


@dataclass
class WindowSpec:
    """One judging window: a run of consecutive source cues and the candidate text for it."""
    idx: int
    first: int                     # 1-based source cue index, inclusive
    last: int                      # 1-based source cue index, inclusive
    units: list[Unit]
    mode: str                      # block | passage

    @property
    def start_ms(self) -> int:
        return self.units[0].source.start_ms

    @property
    def end_ms(self) -> int:
        return max(u.source.end_ms for u in self.units)

    @property
    def source_text(self) -> str:
        return " ".join(u.source.text for u in self.units).strip()

    @property
    def target_cues(self) -> list[Cue]:
        """The candidate's own cues for this window, de-duplicated, in presentation order.

        De-duplication matters in passage mode: one long reference cue can overlap several
        source cues, and counting it once per source cue would inflate the passage.
        """
        seen, out = set(), []
        for u in self.units:
            for c in u.targets:
                key = (c.start_ms, c.end_ms, c.block)
                if key not in seen:
                    seen.add(key)
                    out.append(c)
        return out

    @property
    def target_text(self) -> str:
        return " ".join(c.text for c in self.target_cues).strip()

    def render(self) -> str:
        """The window as the judge is shown it.

        Block mode numbers the pairs, because a 1:1 pairing is the information the judge needs
        to localise an error. Passage mode shows two passages plus the candidate's raw cues:
        the passages are what Accuracy and Fluency are judged on, the raw cues are what
        Characters-per-Line and Lines-per-Box are judged on, and conflating them is exactly the
        re-segmentation penalty the protocol is designed to avoid.
        """
        if self.mode == "block":
            rows = []
            for u in self.units:
                tgt = u.targets[0].block if u.targets else ""
                rows.append(f"[{u.source.index}] "
                            f"({format_timestamp(u.source.start_ms)} --> "
                            f"{format_timestamp(u.source.end_ms)}, "
                            f"{u.source.duration:.1f}s)\n"
                            f"  SOURCE: {u.source.block}\n"
                            f"  TARGET: {tgt}")
            return "\n".join(rows)
        cues = self.target_cues
        blocks = "\n".join(f"  [{i}] {c.block}" for i, c in enumerate(cues, 1)) or "  (none)"
        return (f"SOURCE PASSAGE (cues {self.first}-{self.last}, "
                f"{format_timestamp(self.start_ms)} --> {format_timestamp(self.end_ms)}):\n"
                f"  {self.source_text}\n\n"
                f"TARGET PASSAGE (same time span, independently segmented):\n"
                f"  {self.target_text or '(empty)'}\n\n"
                f"TARGET CUES AS DISPLAYED (for the technical checks only):\n{blocks}")

    def stats(self) -> dict:
        cues = self.target_cues
        return {
            "window": self.idx, "first_cue": self.first, "last_cue": self.last,
            "source_cues": len(self.units), "target_cues": len(cues),
            "start": format_timestamp(self.start_ms), "end": format_timestamp(self.end_ms),
            "empty_targets": sum(1 for u in self.units if not u.targets),
        }


def pair_blockwise(source: list[Cue], target: list[Cue]) -> tuple[list[Unit], dict]:
    """1:1 pairing for a model hypothesis, which preserves the source segmentation.

    A count mismatch is reported rather than raised. It is a real and informative failure - a
    system that dropped its last 40 cues should be judged as having dropped them, scoring
    Undertranslation 10 on those windows, not excluded from the table - so the surplus source
    cues get an empty target and the discrepancy goes into the report.
    """
    units = [Unit(s, [target[i]] if i < len(target) else []) for i, s in enumerate(source)]
    return units, {
        "mode": "block", "source_cues": len(source), "target_cues": len(target),
        "paired": sum(1 for u in units if u.targets),
        "unpaired_source": sum(1 for u in units if not u.targets),
        "surplus_target": max(0, len(target) - len(source)),
        "count_match": len(source) == len(target),
    }


def pair_by_overlap(source: list[Cue], target: list[Cue],
                    min_overlap_ms: int = 1) -> tuple[list[Unit], dict]:
    """Timestamp-overlap pairing for an independently segmented reference.

    Every target cue that shares screen time with a source cue is attached to it, so a
    one-to-many or many-to-one correspondence is represented as such instead of being forced
    into a 1:1 shape. A target cue may be attached to more than one source cue; `target_cues`
    de-duplicates at the window level, which is where it matters.

    Linear, not quadratic: both sides are in presentation order, so a moving start index is
    enough. On a 600-cue episode the difference is irrelevant, but the alignment runs once per
    (system, direction, episode) across thirty directions and it costs nothing to not be
    quadratic.
    """
    units, j = [], 0
    for s in source:
        while j < len(target) and target[j].end_ms <= s.start_ms:
            j += 1
        k, hits = j, []
        while k < len(target) and target[k].start_ms < s.end_ms:
            if target[k].overlaps(s.start_ms, s.end_ms) >= min_overlap_ms:
                hits.append(target[k])
            k += 1
        units.append(Unit(s, hits))
    attached = {id(c) for u in units for c in u.targets}
    return units, {
        "mode": "passage", "source_cues": len(source), "target_cues": len(target),
        "paired": sum(1 for u in units if u.targets),
        "unpaired_source": sum(1 for u in units if not u.targets),
        # A target cue matched by nothing is usually a sign, forced subtitle or credit that the
        # source does not have. Reported because a large number of them means the two files are
        # not the same episode, which is the failure worth catching before spending judge calls.
        "unmatched_target": sum(1 for c in target if id(c) not in attached),
        "multi_target_source": sum(1 for u in units if len(u.targets) > 1),
        "min_overlap_ms": min_overlap_ms,
    }


def make_windows(units: list[Unit], mode: str, size: int = DEFAULT_WINDOW,
                 overlap: float = DEFAULT_OVERLAP) -> list[WindowSpec]:
    """Cut the paired units into windows of `size` consecutive source cues.

    `overlap` is a fraction of the window. The default is 0 - contiguous, non-overlapping
    windows - because the episode score is an unweighted mean over windows, so overlapping
    them would count the shared cues twice and quietly weight the middle of an episode above
    its ends. Overlap is available because the paper leaves the geometry to the config, and
    whatever is used is recorded in the result file.
    """
    if size < 1:
        raise ValueError(f"window size must be >= 1, got {size}")
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    stride = max(1, int(round(size * (1 - overlap))))
    out: list[WindowSpec] = []
    for start in range(0, len(units), stride):
        chunk = units[start:start + size]
        if not chunk:
            break
        out.append(WindowSpec(len(out) + 1, chunk[0].source.index, chunk[-1].source.index,
                              chunk, mode))
        if start + size >= len(units):
            break                    # the last window already reaches the end; do not re-cut it
    return out


def prepare(source_path: str | Path, target_path: str | Path, *, mode: str = "block",
            window: int = DEFAULT_WINDOW, overlap: float = DEFAULT_OVERLAP,
            normalise: bool | None = None) -> tuple[list[WindowSpec], dict]:
    """Read both files and return the judging windows plus an alignment report.

    `normalise` defaults to whatever the mode implies: references (passage mode) are
    normalised, model hypotheses (block mode) are not. A hypothesis is this system's own
    output, so cleaning it up before scoring would be scoring a file the system did not
    produce - if it emits a space before every comma, that is a Spacing Error and the table
    should say so.
    """
    if mode not in ("block", "passage"):
        raise ValueError(f"mode must be 'block' or 'passage', got {mode!r}")
    source = parse_srt(source_path)
    target = parse_srt(target_path)
    if normalise is None:
        normalise = (mode == "passage")
    normalised = 0
    if normalise:
        target, normalised = normalise_cues(target)
    units, report = (pair_blockwise(source, target) if mode == "block"
                     else pair_by_overlap(source, target))
    windows = make_windows(units, mode, window, overlap)
    report.update({
        "source_file": str(source_path), "target_file": str(target_path),
        "normalised": normalise, "normalised_cues": normalised,
        "window_size": window, "window_overlap": overlap, "windows": len(windows),
    })
    return windows, report


# --------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------

def self_test() -> int:
    """Check the parser, the normaliser, both pairing modes and the window geometry.

    Every case here is one that was wrong at some point during development, so the list is a
    regression suite rather than a demonstration. The two that cost the most: the parser used to
    append the *next* cue's index number to the previous cue's text, and `_SPACED_CJK` used to
    miss full-width punctuation, so a normalised reference still read as a Spacing Error.
    """
    import tempfile
    failures: list[str] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        if not cond:
            failures.append(f"{label}{': ' + detail if detail else ''}")

    def write(name: str, text: str) -> str:
        path = Path(tempfile.mkdtemp()) / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    # -- the parser -----------------------------------------------------------------
    # Cue 3's text is two lines; cue 4's index sits directly above its timestamp. A cue whose
    # text genuinely ends in a number ("1999.") must survive, which is why the fix keys on
    # adjacency to the arrow rather than on "the last line looks like an index".
    src = write("src.srt", """1
00:00:01,000 --> 00:00:03,000
It was <i>1999</i>.

2
00:00:03,500 --> 00:00:05,000
1999.

3
00:00:05,500 --> 00:00:08,000
She said she would call
but she never did.

4
00:00:08,500 --> 00:00:10,000
Ask the prosecutor.
""")
    cues = parse_srt(src)
    check("parsed the wrong number of cues", len(cues) == 4, str(len(cues)))
    check("html tags not stripped", cues[0].text == "It was 1999.", repr(cues[0].text))
    check("a cue whose text is a bare number was eaten",
          cues[1].text == "1999.", repr(cues[1].text))
    check("the next cue's index leaked into this cue's text",
          cues[2].text == "She said she would call but she never did.", repr(cues[2].text))
    check("multi-line cue lost its line break in .block",
          cues[2].block == "She said she would call\nbut she never did.", repr(cues[2].block))
    check("timestamps not parsed", (cues[3].start_ms, cues[3].end_ms) == (8500, 10000),
          f"{cues[3].start_ms}/{cues[3].end_ms}")
    check("duration wrong", abs(cues[0].duration - 2.0) < 1e-9, str(cues[0].duration))
    check("index not taken from the file", [c.index for c in cues] == [1, 2, 3, 4],
          str([c.index for c in cues]))

    # -- normalisation --------------------------------------------------------------
    for raw, want, why in [
        ("我 不 知 道 他 什 么 意 思 ， 去 问 。", "我不知道他什么意思，去问。",
         "CJK text and full-width punctuation spaced apart"),
        ("こ れ は 何 で す か ？", "これは何ですか？", "kana"),
        ("we don ' t care", "we don't care", "English clitic"),
        ("l ' homme qu ' il aime", "l'homme qu'il aime", "French elision"),
        ("Hello , world !", "Hello, world!", "space before punctuation"),
        ("( aside ) done", "(aside) done", "space after an opening bracket"),
        ("a well - known fact", "a well-known fact", "spaced hyphen"),
        ("too    many spaces", "too many spaces", "runs of spaces"),
        # Left alone on purpose:
        ("- She said ' hello ' to me", "- She said ' hello ' to me",
         "quotation marks must not be turned into contractions"),
        ("사랑 을 해", "사랑 을 해",
         "Korean spaces between words are real; stripping them would manufacture the error"),
        ("Ne pas !", "Ne pas !",
         "French narrow no-break space (U+202F) is correct typography"),
        ("Ne pas !", "Ne pas !", "no-break space (U+00A0) likewise"),
    ]:
        got = normalise_text(raw)
        check(f"normalise: {why}", got == want, f"{raw!r} -> {got!r}, want {want!r}")

    # -- block mode -----------------------------------------------------------------
    # 4 source cues, 3 target cues. A system that dropped a cue must be *reported* as having
    # dropped it, not rejected with an exception: the omission is the finding.
    tgt3 = write("t3.srt", """1
00:00:01,000 --> 00:00:03,000
Era il 1999.

2
00:00:03,500 --> 00:00:05,000
1999.

3
00:00:05,500 --> 00:00:08,000
Disse che avrebbe chiamato.
""")
    windows, report = prepare(src, tgt3, mode="block", window=20)
    check("block mode did not report the count mismatch",
          report["count_match"] is False and report["unpaired_source"] == 1,
          f"{report['count_match']}/{report['unpaired_source']}")
    check("block mode should not normalise the hypothesis", report["normalised"] is False)
    check("block mode produced the wrong number of windows", len(windows) == 1)
    check("the unpaired source cue lost its slot", len(windows[0].units) == 4)
    check("the dropped cue rendered as translated",
          "  TARGET: \n" in windows[0].render() + "\n", "cue 4 should render an empty TARGET")

    # -- passage mode ---------------------------------------------------------------
    # An independently segmented reference: 2 cues covering the same span as 4 source cues,
    # with the spacing defects a real reference file has.
    ref = write("ref.srt", """1
00:00:00,800 --> 00:00:05,200
Era il 1999 .
1999 .

2
00:00:05,300 --> 00:00:10,000
Disse che avrebbe chiamato ,
ma non lo fece . Chiedi al PM .
""")
    pwin, preport = prepare(src, ref, mode="passage", window=20)
    check("passage mode did not normalise the reference",
          preport["normalised"] is True and preport["normalised_cues"] == 2,
          f"{preport['normalised']}/{preport['normalised_cues']}")
    check("a source cue went unmatched in passage mode",
          preport["unmatched_target"] == 0 and len(pwin) == 1,
          f"unmatched={preport['unmatched_target']}, windows={len(pwin)}")
    rendered = pwin[0].render()
    for want in ("SOURCE PASSAGE", "TARGET PASSAGE", "TARGET CUES AS DISPLAYED"):
        check(f"passage render lost the {want!r} section", want in rendered)
    check("normalisation did not reach the rendered passage",
          "1999 ." not in rendered and "chiamato ," not in rendered,
          "spaced punctuation survived into the judge's view")
    check("the two reference cues were not de-duplicated across 4 source cues",
          len(pwin[0].target_cues) == 2, str(len(pwin[0].target_cues)))

    # -- window geometry ------------------------------------------------------------
    units = [Unit(source=Cue(index=i, start_ms=i * 1000, end_ms=i * 1000 + 900, lines=["x"]))
             for i in range(1, 11)]
    for size, overlap, want, why in [
        (5, 0.0, [(1, 5), (6, 10)], "contiguous, non-overlapping"),
        (4, 0.0, [(1, 4), (5, 8), (9, 10)], "a short final window is kept, not padded"),
        (4, 0.5, [(1, 4), (3, 6), (5, 8), (7, 10)], "50% overlap: stride is half the size"),
        # An odd size at 50% has no integer stride. 5*(1-0.5) = 2.5 rounds to 2 under
        # Python's round-half-to-even, i.e. slightly *more* overlap than asked for, not
        # less - which is the safe direction, since no cue can fall between windows.
        (5, 0.5, [(1, 5), (3, 7), (5, 9), (7, 10)], "odd size at 50% overlap"),
        (20, 0.0, [(1, 10)], "size larger than the episode gives one window"),
        (1, 0.0, [(i, i) for i in range(1, 11)], "size 1"),
    ]:
        got = [(w.first, w.last) for w in make_windows(units, "block", size, overlap)]
        check(f"windows({size}, {overlap}) - {why}", got == want, f"{got} != {want}")
    for size, overlap in ((0, 0.0), (5, 1.0), (5, -0.1)):
        try:
            make_windows(units, "block", size, overlap)
            check(f"make_windows({size}, {overlap}) should have raised", False)
        except ValueError:
            pass
    check("window indices are not 1-based and consecutive",
          [w.idx for w in make_windows(units, "block", 4, 0.0)] == [1, 2, 3])

    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"PASSED: parser handles tags, multi-line cues, a numeric-only cue and the index-line "
          f"leak; 12 normalisation cases including the four that must be left alone; block "
          f"mode reports a dropped cue instead of raising; passage mode normalises, "
          f"de-duplicates and renders all three sections; window geometry correct for 6 "
          f"size/overlap combinations and rejects 3 invalid ones")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Align a source and a candidate, show the windows.")
    if "--self-test" in (argv if argv is not None else sys.argv[1:]):
        return self_test()
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--self-test", action="store_true",
                    help="run the built-in regression checks and exit")
    ap.add_argument("--mode", choices=("block", "passage"), default="block")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    ap.add_argument("--no-normalise", action="store_true",
                    help="skip reference normalisation even in passage mode")
    ap.add_argument("--show", type=int, default=1, help="print the first N windows verbatim")
    args = ap.parse_args(argv)

    windows, report = prepare(args.source, args.target, mode=args.mode, window=args.window,
                              overlap=args.overlap,
                              normalise=False if args.no_normalise else None)
    for k, v in report.items():
        print(f"  {k}: {v}")
    for w in windows[:max(0, args.show)]:
        print("\n" + "-" * 70)
        print(f"window {w.idx}/{len(windows)}  cues {w.first}-{w.last}")
        print(w.render())
    if not report.get("count_match", True):
        print(f"\nWARNING: block mode with {report['source_cues']} source cues and "
              f"{report['target_cues']} target cues; {report['unpaired_source']} source cue(s) "
              f"have no translation and will be judged as omissions.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
