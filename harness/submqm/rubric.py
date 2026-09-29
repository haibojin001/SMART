"""The SubMQM rubric: seven dimensions, nineteen error types, and the scoring formula.

SubMQM is a subtitle-adapted MQM protocol. An evaluator assigns a penalty in {0, 5, 10} to
each of nineteen error types for each window of an episode; dimension scores aggregate their
own error types, and Overall weights the dimensions to emphasise semantic fidelity. Every
number is a *penalty*, so lower is better throughout - there is no inversion anywhere.

This module is data plus arithmetic and nothing else. It makes no model calls and imports
nothing outside the standard library, which is deliberate: the rubric is what the judge
prompt is built from (`judge.py`), what the aggregation uses (`evaluate.py`) and what the
LaTeX column order comes from (`tables.py`), and all three must agree by construction rather
than by three hand-maintained copies.

Two orders exist and they are not interchangeable:

  * `DIMENSIONS` is the *structural* order - the one the rubric table and the fine-grained
    result tables use, and the one to emit in JSON.
  * `SUMMARY_ORDER` is descending by weight, which is how the summary table presents the
    same seven dimensions.

`self_check()` runs on import: the weights must sum to exactly 1, the error-type counts must
sum to 19, and every abbreviation must be unique. A rubric that has silently lost an error
type would otherwise still produce plausible-looking penalties.
"""

from __future__ import annotations

# --------------------------------------------------------------------------------
# the nineteen error types, in structural order
# --------------------------------------------------------------------------------
# Descriptions are the rubric's own wording, because they are what the evaluator is shown.
# Abbreviations are the result tables' column headers.

DIMENSIONS: dict[str, list[str]] = {
    "terminology": ["name_inconsistency", "term_inconsistency"],
    "accuracy": ["mistranslation", "undertranslation", "overtranslation"],
    "fluency": ["coherence", "naturalness", "vividness"],
    "linguistic_conventions": ["mispunctuation", "miscapitalization", "grammar",
                               "spacing_error"],
    "technical": ["incorrect_line_breaking", "exceeding_characters_per_line",
                  "exceeding_lines_per_box"],
    "locale_conventions": ["localization_error", "language_detection_error"],
    "audience_appropriateness": ["profanity", "formality_error"],
}

WEIGHTS: dict[str, float] = {
    "accuracy": 0.30,
    "terminology": 0.20,
    "fluency": 0.20,
    "audience_appropriateness": 0.12,
    "linguistic_conventions": 0.08,
    "technical": 0.06,
    "locale_conventions": 0.04,
}

# Dimension labels for the aggregate table, in its column order.
DIMENSION_LABELS: dict[str, str] = {
    "terminology": "Term.",
    "accuracy": "Acc.",
    "fluency": "Flu.",
    "linguistic_conventions": "Ling.",
    "technical": "Tech.",
    "locale_conventions": "Locale",
    "audience_appropriateness": "Audience",
}

# Full dimension names, for the judge prompt and the rubric table.
DIMENSION_NAMES: dict[str, str] = {
    "terminology": "Terminology",
    "accuracy": "Accuracy",
    "fluency": "Fluency",
    "linguistic_conventions": "Linguistic Conventions",
    "technical": "Technical",
    "locale_conventions": "Locale Conventions",
    "audience_appropriateness": "Audience Appropriateness",
}

# The summary table's presentation order: descending weight.
SUMMARY_ORDER: list[str] = ["accuracy", "terminology", "fluency", "audience_appropriateness",
                            "linguistic_conventions", "technical", "locale_conventions"]

ERROR_TYPES: dict[str, tuple[str, str, str]] = {
    # key: (display name, table abbreviation, description shown to the evaluator)
    "name_inconsistency": (
        "Name Inconsistency", "NameInc",
        "Inconsistent translations or spellings of recurring proper names, such as "
        "characters, places, and entities."),
    "term_inconsistency": (
        "Term Inconsistency", "TermInc",
        "Inconsistent translations of recurring domain-specific or contextual terms and "
        "phrases."),
    "mistranslation": (
        "Mistranslation", "MisTrans",
        "Translation that incorrectly changes the meaning of the source."),
    "undertranslation": (
        "Undertranslation", "UndTrans",
        "Source content that should be translated is omitted."),
    "overtranslation": (
        "Overtranslation", "OvrTrans",
        "Unsupported information or specificity is introduced into the translation."),
    "coherence": (
        "Coherence", "Coher",
        "The translation lacks continuity with the surrounding scene or discourse."),
    "naturalness": (
        "Naturalness", "Natur",
        "Awkward or translation-like phrasing that does not resemble native usage."),
    "vividness": (
        "Vividness", "Vivid",
        "Emotional nuance, humor, wordplay, or stylistic expressiveness is weakened."),
    "mispunctuation": (
        "Mispunctuation", "MisPunc",
        "Missing, incorrect, or improperly formatted punctuation."),
    "miscapitalization": (
        "Miscapitalization", "MisCap",
        "Incorrect capitalization of sentence-initial words, proper nouns, or other forms."),
    "grammar": (
        "Grammar", "Gram",
        "Grammatical errors in agreement, morphology, syntax, or related constructions."),
    "spacing_error": (
        "Spacing Error", "Space",
        "Missing, extra, or duplicated spaces around words or punctuation."),
    "incorrect_line_breaking": (
        "Incorrect Line Breaking", "LineBrk",
        "Line breaks improperly separate tightly coupled linguistic units."),
    "exceeding_characters_per_line": (
        "Exceeding Characters per Line", "CharLim",
        "A subtitle line exceeds the predefined character-per-line constraint."),
    "exceeding_lines_per_box": (
        "Exceeding Lines per Box", "LineLim",
        "A subtitle event exceeds the predefined maximum number of lines."),
    "localization_error": (
        "Localization Error", "LocErr",
        "Units, currencies, dates, or culturally specific expressions are improperly "
        "localized."),
    "language_detection_error": (
        "Language Detection Error", "LangDet",
        "The detected language does not match the expected source or target language."),
    "profanity": (
        "Profanity", "Profan",
        "Profanity is unjustifiably strengthened, weakened, introduced, or removed."),
    "formality_error": (
        "Formality Error", "Formal",
        "The register or level of formality is inappropriate or inconsistent with the "
        "scene."),
}

# The only legal per-error-type values. 0 doubles as "not applicable": a window with no
# profanity in it and a window whose profanity is handled correctly both score 0, which is
# why MisCap is exactly 0.00 down every en->zh column - Chinese has no letter case.
ALLOWED_SCORES: tuple[int, ...] = (0, 5, 10)
SEVERITY: dict[int, str] = {0: "no error or not applicable", 5: "minor", 10: "severe"}

# Flat structural order of all nineteen keys.
ALL_TYPES: list[str] = [t for subs in DIMENSIONS.values() for t in subs]

# The two fine-grained result tables' column sets, in their published order.
SEMANTIC_TYPES: list[str] = (DIMENSIONS["terminology"] + DIMENSIONS["accuracy"]
                             + DIMENSIONS["fluency"])
FORM_TYPES: list[str] = (DIMENSIONS["linguistic_conventions"] + DIMENSIONS["technical"]
                         + DIMENSIONS["locale_conventions"]
                         + DIMENSIONS["audience_appropriateness"])

DIMENSION_OF: dict[str, str] = {t: d for d, subs in DIMENSIONS.items() for t in subs}


# --------------------------------------------------------------------------------
# the scoring formula
# --------------------------------------------------------------------------------
# A "window" here is one dict of {error_type: score}. Both means are unweighted, which has a
# consequence worth stating: a dimension with four error types dilutes a single severe error
# more than a dimension with two, so Linguistic Conventions is structurally harder to move
# than Terminology. That is the rubric's design, not an artefact.

Window = dict[str, int]


def normalise_window(raw: dict) -> Window:
    """One evaluator response as a complete, validated window.

    Missing error types default to 0 - explicitly sanctioned, since an evaluator that saw no
    units in a window has nothing to say about Localization Error. Anything the evaluator
    reports outside {0, 5, 10} is a protocol violation and raises rather than being rounded,
    because silently snapping 7 to 5 would make a broken judge look like a working one.
    """
    out: Window = {}
    for t in ALL_TYPES:
        v = raw.get(t, 0)
        if v is None:
            v = 0
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if v not in ALLOWED_SCORES:
            raise ValueError(f"{t}: {v!r} is not one of {ALLOWED_SCORES}")
        out[t] = int(v)
    unknown = set(raw) - set(ALL_TYPES)
    if unknown:
        raise ValueError(f"unknown error types in judge response: {sorted(unknown)}")
    return out


def type_penalty(windows: list[Window], error_type: str) -> float:
    """One error type's penalty: its mean over the episode's windows."""
    if not windows:
        return 0.0
    return sum(w.get(error_type, 0) for w in windows) / len(windows)


def dimension_penalty(windows: list[Window], dimension: str) -> float:
    """One dimension's penalty: mean over its error types, then mean over windows."""
    subs = DIMENSIONS[dimension]
    if not windows:
        return 0.0
    per_window = [sum(w.get(c, 0) for c in subs) / len(subs) for w in windows]
    return sum(per_window) / len(per_window)


def overall_penalty(windows: list[Window]) -> float:
    """The weighted average of the seven dimension penalties."""
    return sum(WEIGHTS[d] * dimension_penalty(windows, d) for d in DIMENSIONS)


def score_episode(windows: list[Window]) -> dict:
    """Every number the result tables need, for one (system, direction, episode).

    Returns per-error-type, per-dimension and Overall penalties, plus the window count, so a
    later aggregation across episodes can weight correctly if it wants to. The paper's own
    aggregation does not: it means over episodes unweighted, the same way it means over
    windows unweighted.
    """
    return {
        "windows": len(windows),
        "types": {t: type_penalty(windows, t) for t in ALL_TYPES},
        "dimensions": {d: dimension_penalty(windows, d) for d in DIMENSIONS},
        "overall": overall_penalty(windows),
    }


def mean_episodes(per_episode: list[dict]) -> dict:
    """Aggregate `score_episode` results across episodes, unweighted.

    Deliberately not a re-derivation from the pooled windows. Pooling would weight a
    900-segment episode more heavily than a 300-segment one; the protocol averages over
    episodes, so an episode is one vote regardless of length.
    """
    scored = [e for e in per_episode if e.get("windows")]
    if not scored:
        return {"episodes": 0, "windows": 0,
                "types": {t: 0.0 for t in ALL_TYPES},
                "dimensions": {d: 0.0 for d in DIMENSIONS}, "overall": 0.0}
    n = len(scored)
    return {
        "episodes": n,
        "windows": sum(e["windows"] for e in scored),
        "types": {t: sum(e["types"][t] for e in scored) / n for t in ALL_TYPES},
        "dimensions": {d: sum(e["dimensions"][d] for e in scored) / n for d in DIMENSIONS},
        "overall": sum(e["overall"] for e in scored) / n,
    }


# --------------------------------------------------------------------------------
# the judge-facing rendering of the rubric
# --------------------------------------------------------------------------------

def rubric_text() -> str:
    """The rubric as the evaluator is shown it: seven headed dimensions, nineteen items.

    Built from the tables above rather than written out as a prose constant, so an error type
    cannot exist in the scoring code and be absent from the prompt. That failure mode is
    quiet and expensive: the evaluator never scores the type, every window defaults it to 0,
    and the column reads as a perfect score in the paper.
    """
    lines = []
    for dim, subs in DIMENSIONS.items():
        lines.append(f"{DIMENSION_NAMES[dim]} ({len(subs)} error types):")
        for t in subs:
            name, _abbrev, desc = ERROR_TYPES[t]
            lines.append(f"  - {t} ({name}): {desc}")
    return "\n".join(lines)


def json_skeleton() -> str:
    """The exact JSON object the evaluator must return, with every key present.

    Showing the skeleton rather than describing it is what keeps the response parseable; the
    keys are generated, so the skeleton and `normalise_window`'s expectations are the same
    list by construction.
    """
    body = ",\n".join(f'    "{t}": 0' for t in ALL_TYPES)
    return "{\n" + body + "\n}"


# --------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------

def self_check() -> None:
    """Structural invariants of the rubric. Runs on import."""
    assert len(DIMENSIONS) == 7, f"expected 7 dimensions, got {len(DIMENSIONS)}"
    assert len(ALL_TYPES) == 19, f"expected 19 error types, got {len(ALL_TYPES)}"
    assert len(set(ALL_TYPES)) == 19, "an error type is listed under two dimensions"
    counts = [len(v) for v in DIMENSIONS.values()]
    assert counts == [2, 3, 3, 4, 3, 2, 2], f"error-type counts drifted: {counts}"
    assert set(ERROR_TYPES) == set(ALL_TYPES), "ERROR_TYPES and DIMENSIONS disagree"
    assert set(WEIGHTS) == set(DIMENSIONS), "WEIGHTS and DIMENSIONS disagree"
    assert set(DIMENSION_LABELS) == set(DIMENSIONS) == set(DIMENSION_NAMES)
    assert sorted(SUMMARY_ORDER) == sorted(DIMENSIONS), "SUMMARY_ORDER is not a permutation"
    # Exact, not approximate: these are two-decimal constants, and a weight vector that does
    # not sum to 1 would silently rescale every Overall in the paper.
    total = round(sum(WEIGHTS.values()), 10)
    assert total == 1.0, f"weights sum to {total}, not 1"
    abbrevs = [v[1] for v in ERROR_TYPES.values()]
    assert len(set(abbrevs)) == 19, "two error types share a table abbreviation"
    assert len(SEMANTIC_TYPES) == 8 and len(FORM_TYPES) == 11, (
        f"result-table column split drifted: {len(SEMANTIC_TYPES)} + {len(FORM_TYPES)}")
    assert set(SEMANTIC_TYPES) | set(FORM_TYPES) == set(ALL_TYPES)
    assert not set(SEMANTIC_TYPES) & set(FORM_TYPES)
    # The formula itself: an all-severe episode must score exactly 10, an all-clean one 0.
    worst = [{t: 10 for t in ALL_TYPES}]
    assert abs(overall_penalty(worst) - 10.0) < 1e-9, overall_penalty(worst)
    assert overall_penalty([{t: 0 for t in ALL_TYPES}]) == 0.0


self_check()


def self_test() -> int:
    """The behaviour `self_check` does not reach.

    `self_check` asserts the rubric's *shape* - seven dimensions, nineteen types, weights summing
    to one - and it runs on import, so any module that imports this one already enforces it. What
    it does not touch is what happens to an actual judge response on the way to a number, and that
    is where a silent corruption would live: a penalty outside {0, 5, 10} quietly rounded, an
    unrecognised key quietly dropped, or episodes pooled instead of averaged. Each of those
    produces a plausible table with wrong numbers in it, which is the worst failure mode available.

    Wired to `--self-test` so the loop the README documents is honest for all five modules.
    """
    failures = []

    def check(label, got, want):
        if got != want:
            failures.append(f"{label}: got {got!r}, expected {want!r}")

    # --- normalise_window: what reaches the arithmetic ------------------------------------
    full = normalise_window({t: 0 for t in ALL_TYPES})
    check("a complete clean window keeps all 19 keys", len(full), 19)

    # A missing type defaults to 0. Sanctioned: a window with no proper nouns in it has nothing
    # to say about NameInc, and forcing the judge to invent a score would be worse.
    check("a missing error type defaults to 0", normalise_window({})["name_inconsistency"], 0)

    # 5.0 is 5; 5.5 is not a severity. The first is a JSON artefact, the second is a broken judge.
    check("an integral float is accepted", normalise_window({"name_inconsistency": 10.0})["name_inconsistency"], 10)
    for bad in (7, 1, -5, 11, 5.5, "5"):
        try:
            normalise_window({"name_inconsistency": bad})
            failures.append(f"penalty {bad!r} was accepted; only {ALLOWED_SCORES} are severities, "
                            f"and rounding one would make a broken judge look like a working one")
        except ValueError:
            pass

    # An unknown key means the judge answered a different rubric than the one it was shown.
    # Dropping it silently would score that response as if the missing dimension were perfect.
    try:
        normalise_window({"NotAType": 5})
        failures.append("an unknown error type was accepted; the response would be scored as if "
                        "the type it actually reported did not exist")
    except ValueError:
        pass

    # --- the arithmetic ------------------------------------------------------------------
    clean, worst = [{t: 0 for t in ALL_TYPES}], [{t: 10 for t in ALL_TYPES}]
    check("a clean episode scores 0", overall_penalty(clean), 0.0)
    if abs(overall_penalty(worst) - 10.0) > 1e-9:
        failures.append(f"an all-severe episode scores {overall_penalty(worst)}, not 10")

    # Lower is better, and one error in one dimension must move Overall by that dimension's
    # weight - not by more, and not by nothing.
    one = [dict({t: 0 for t in ALL_TYPES}, name_inconsistency=10)]
    expected = WEIGHTS["terminology"] * (10 / len(DIMENSIONS["terminology"]))
    if abs(overall_penalty(one) - expected) > 1e-9:
        failures.append(f"a single name_inconsistency moved Overall to {overall_penalty(one)}, expected "
                        f"{expected} - the dimension weight is not being applied as documented")

    # A window mean, not a sum: two windows, one clean, one severe, is the midpoint.
    mixed = [{t: 0 for t in ALL_TYPES}, {t: 10 for t in ALL_TYPES}]
    if abs(overall_penalty(mixed) - 5.0) > 1e-9:
        failures.append(f"two windows averaged to {overall_penalty(mixed)}, not 5.0")

    check("no windows scores 0 rather than raising", overall_penalty([]), 0.0)

    # --- mean_episodes: one episode, one vote ---------------------------------------------
    # The protocol averages over episodes unweighted. Pooling windows instead would let a long
    # episode outvote a short one, so a 1-window clean episode and a 9-window severe episode
    # must average to 5, not to 9.
    a = score_episode([{t: 0 for t in ALL_TYPES}])
    b = score_episode([{t: 10 for t in ALL_TYPES}] * 9)
    agg = mean_episodes([a, b])
    check("episodes counted", agg["episodes"], 2)
    check("windows still reported", agg["windows"], 10)
    if abs(agg["overall"] - 5.0) > 1e-9:
        failures.append(f"a 1-window and a 9-window episode aggregated to {agg['overall']}, not "
                        f"5.0 - episodes are being pooled by length instead of averaged")

    # An episode with no windows is not a perfect episode; it must not drag an average down to 0.
    agg2 = mean_episodes([b, {"windows": 0, "types": {}, "dimensions": {}, "overall": 0.0}])
    if abs(agg2["overall"] - 10.0) > 1e-9:
        failures.append(f"an empty episode was counted as a clean one: {agg2['overall']}")
    check("nothing to aggregate yields 0 episodes", mean_episodes([])["episodes"], 0)

    # --- the prompt and the scoring code are the same list -------------------------------
    import json
    # This is the quiet one: a type present in the code and absent from the prompt is never
    # scored, defaults to 0 in every window, and publishes as a perfect column.
    text = rubric_text()
    missing = [t for t in ALL_TYPES if f"  - {t} (" not in text]
    if missing:
        failures.append(f"{len(missing)} error type(s) are scored but never shown to the judge, "
                        f"so they would publish as perfect columns: {missing}")
    try:
        skeleton = json.loads(json_skeleton())
        check("the skeleton offers exactly the 19 keys", sorted(skeleton), sorted(ALL_TYPES))
    except ValueError as e:
        failures.append(f"json_skeleton is not valid JSON, so no judge could follow it: {e}")

    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"PASSED: rubric structure ({len(DIMENSIONS)} dimensions, {len(ALL_TYPES)} types, "
          f"weights sum to 1) is asserted on import; penalties outside {ALLOWED_SCORES} and "
          f"unknown error types are rejected rather than coerced, missing types default to 0, "
          f"one error moves Overall by exactly its dimension weight, windows and episodes are "
          f"averaged rather than pooled, an empty episode is not a clean one, and all 19 types "
          f"appear in both the judge prompt and the JSON skeleton")
    return 0


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(self_test())

    print(f"SubMQM rubric: {len(DIMENSIONS)} dimensions, {len(ALL_TYPES)} error types, "
          f"penalties in {ALLOWED_SCORES} (lower is better)\n")
    for dim in SUMMARY_ORDER:
        subs = DIMENSIONS[dim]
        print(f"  {DIMENSION_NAMES[dim]:<26} w={WEIGHTS[dim]:.2f}  {len(subs)} types: "
              + ", ".join(ERROR_TYPES[t][1] for t in subs))
    print(f"\n  {'total':<26} w={sum(WEIGHTS.values()):.2f}  {len(ALL_TYPES)} types")
    print("\nsemantic table columns:", " ".join(ERROR_TYPES[t][1] for t in SEMANTIC_TYPES))
    print("form table columns:    ", " ".join(ERROR_TYPES[t][1] for t in FORM_TYPES))
