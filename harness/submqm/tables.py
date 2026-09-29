"""Turn a results file into the LaTeX the paper's result tables are made of.

Three table shapes, all built from `rubric`'s own ordering so a column cannot drift from the
scorer that filled it:

    semantic, 11 columns   Dir. | Method | NameInc TermInc | MisTrans UndTrans OvrTrans |
                           Coher Natur Vivid | Overall
    form, 14 columns       Dir. | Method | MisPunc MisCap Gram Space | LineBrk CharLim LineLim |
                           LocErr LangDet | Profan Formal | Overall
    aggregate, 9 columns   Method | Term. Acc. Flu. Ling. Tech. Locale Audience | Overall

`Overall` appears in both fine-grained tables and is the same number in each, which is a useful
internal check when reading the output: if the two tables disagree, the results file was edited
between renders.

Highlighting is per direction and per column: the best value in a column within one direction's
block gets `\\bestcell`, the second best `\\secondcell`. Every number is a penalty, so best means
*lowest*. Two rules keep a highlight from claiming more than the data does. Ties share a rank
instead of being broken by row order, which would put the mark on whichever system the job file
happened to list first; and a column where every system scored the same is left unshaded, since
"best" is meaningless there - without this, the many form columns where nothing goes wrong come
out shaded end to end.

    python3 tables.py results/results.json -o tables/
    python3 tables.py results/results.json --group out_en --kind semantic   # to stdout
    python3 tables.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__:
    from . import rubric
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rubric

# What the tables need in a preamble. Emitted with the output so a rendered table compiles on
# its own, rather than depending on definitions living in someone's main.tex.
PREAMBLE = r"""% SubMQM result tables.
% Requires: \usepackage{booktabs,multirow,xcolor} and \usepackage[table]{xcolor} for
% \cellcolor (colortbl). \multirow spans the Dir. column over one direction's systems.
% Penalties, lower is better. Best per column within a direction is shaded blue-grey,
% second best warm beige.
\definecolor{bestbg}{RGB}{222,231,240}
\definecolor{secondbg}{RGB}{247,240,226}
\newcommand{\bestcell}[1]{\cellcolor{bestbg}\textbf{#1}}
\newcommand{\secondcell}[1]{\cellcolor{secondbg}#1}"""

SEMANTIC_GROUPS = [("Terminology", "terminology"), ("Accuracy", "accuracy"),
                   ("Fluency", "fluency")]
FORM_GROUPS = [("Linguistic Conventions", "linguistic_conventions"),
               ("Technical", "technical"), ("Locale Conventions", "locale_conventions"),
               ("Audience Appropriateness", "audience_appropriateness")]

SEMANTIC_FOOTER = (
    r"\textbf{Terminology} (NameInc = Name Inconsistency; TermInc = Term Inconsistency). "
    r"\textbf{Accuracy} (MisTrans = Mistranslation; UndTrans = Undertranslation; "
    r"OvrTrans = Overtranslation). "
    r"\textbf{Fluency} (Coher = Coherence; Natur = Naturalness; Vivid = Vividness).")
FORM_FOOTER = (
    r"\textbf{Linguistic Conventions} (MisPunc = Mispunctuation; MisCap = Miscapitalization; "
    r"Gram = Grammar; Space = Spacing Error). "
    r"\textbf{Technical} (LineBrk = Incorrect Line Breaking; CharLim = Exceeding Characters "
    r"per Line; LineLim = Exceeding Lines per Box). "
    r"\textbf{Locale Conventions} (LocErr = Localization Error; LangDet = Language Detection "
    r"Error). "
    r"\textbf{Audience Appropriateness} (Profan = Profanity; Formal = Formality Error).")
AGGREGATE_FOOTER = (
    r"\textbf{Abbreviations.} Term. = Terminology; Acc. = Accuracy; Flu. = Fluency; "
    r"Ling. = Linguistic Conventions; Tech. = Technical; Locale = Locale Conventions; "
    r"Audience = Audience Appropriateness. Overall is the weighted penalty "
    r"($0.30$ Accuracy, $0.20$ Terminology, $0.20$ Fluency, $0.12$ Audience, "
    r"$0.08$ Linguistic, $0.06$ Technical, $0.04$ Locale).")


def escape(text: str) -> str:
    """LaTeX-escape a system or direction label.

    System names come from the results file, which came from a job file someone wrote by hand,
    so a name containing `_` or `&` is a question of when and not whether.
    """
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        text = text.replace(a, b)
    return text


def direction_label(direction: str) -> str:
    """`en2zh` as the `Dir.` column shows it. Falls back to the raw name off-tree."""
    try:
        harness = Path(__file__).resolve().parent.parent
        if str(harness) not in sys.path:
            sys.path.insert(0, str(harness))
        import directions
        d = directions.resolve(direction)
        return f"{directions.LOCALES[d.src].code}$\\rightarrow${directions.LOCALES[d.tgt].code}"
    except Exception:
        return escape(direction)


def rank_marks(values: list[float | None]) -> list[str]:
    """`best`, `second` or `` for each value in one column of one direction's block.

    Ranks run over *distinct* values, and a shade has to mean "stands out from the field", so
    two thresholds apply. `best` needs two distinct values - with every system equal, or only
    one system present, there is nothing to be best at, and shading a uniform column of 0.00
    reads as a result. `second` needs three, because with only two distinct values the runner-up
    is also the worst, and calling the worst result second-best is worse than saying nothing.
    Ties share whichever rank they land on rather than being broken by row order, which would
    put the mark on whichever system the job file happened to list first.
    """
    live = sorted({v for v in values if v is not None})
    if len(live) < 2:
        return [""] * len(values)
    best = live[0]
    second = live[1] if len(live) >= 3 else None
    out = []
    for v in values:
        if v is None:
            out.append("")
        elif v == best:
            out.append("best")
        elif second is not None and v == second:
            out.append("second")
        else:
            out.append("")
    return out


# Shrink a table only if it is too wide, leaving a narrower one at its natural size. Plain
# `\resizebox{\textwidth}{!}` scales to *exactly* \textwidth, which blows a small table up and
# makes its font larger than the body text. Measured in `article` at 10pt - the narrowest
# column any of these will meet - the semantic table runs ~178pt over and the aggregate ~45pt,
# so this is not a hypothetical.
FIT_OPEN = r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%"
FIT_CLOSE = "}"

# `article` sets \belowcaptionskip to 0pt, so a caption above the table sits flush against the
# \toprule. Venue styles that already leave a gap just get 3pt more, which is not visible.
CAPTION_GAP = r"\vspace{3pt}"


def footer_block(text: str) -> list[str]:
    """The abbreviation note under a table.

    A `\\parbox` rather than `\\\\[2pt] \\footnotesize ...`: outside the tabular the float is in
    vertical mode, where `\\\\` raises "There's no line here to end" and the table does not
    compile at all. The parbox is exactly `\\linewidth`, so the enclosing `\\centering` has
    nothing to centre, and `\\raggedright` keeps the note from being justified into rivers.
    """
    return [r"\vspace{2pt}",
            r"\parbox{\linewidth}{\footnotesize\raggedright %s}" % text]


def cell(value: float | None, mark: str, digits: int = 2) -> str:
    if value is None:
        return "--"
    text = f"{value:.{digits}f}"
    if mark == "best":
        return r"\bestcell{%s}" % text
    if mark == "second":
        return r"\secondcell{%s}" % text
    return text


# --------------------------------------------------------------------------------
# selecting rows
# --------------------------------------------------------------------------------

def group_of(direction: str) -> str:
    return "out_en" if direction.startswith("en2") else "into_en"


def select(results: dict, group: str | None = None,
           systems: list[str] | None = None) -> tuple[list[str], list[str], dict]:
    """(directions, systems, {(direction, system): aggregate}) for one table.

    Direction order follows `directions.TARGETS`, which is the order the paper's tables use, so
    the rendered table can be dropped in without reordering rows. System order follows
    `--systems` when given and otherwise first-appearance in the results file, because the
    conventional layout puts the proposed system last.
    """
    rows = [r for r in results["rows"]
            if r["aggregate"].get("episodes") and (group is None
                                                   or group_of(r["direction"]) == group)]
    seen_dirs, seen_sys = [], []
    table: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r["direction"] not in seen_dirs:
            seen_dirs.append(r["direction"])
        if r["system"] not in seen_sys:
            seen_sys.append(r["system"])
        table[(r["direction"], r["system"])] = r["aggregate"]

    try:
        harness = Path(__file__).resolve().parent.parent
        if str(harness) not in sys.path:
            sys.path.insert(0, str(harness))
        import directions as dirmod
        order = {d.name: i for i, d in enumerate(dirmod.DIRECTIONS)}
        seen_dirs.sort(key=lambda n: order.get(n, len(order)))
    except Exception:
        seen_dirs.sort()

    if systems:
        missing = [s for s in systems if s not in seen_sys]
        if missing:
            raise SystemExit(f"no results for system(s): {', '.join(missing)}. "
                             f"Present: {', '.join(seen_sys)}")
        seen_sys = list(systems)
    return seen_dirs, seen_sys, table


# --------------------------------------------------------------------------------
# fine-grained tables
# --------------------------------------------------------------------------------

def fine_table(results: dict, kind: str, group: str | None = None,
               systems: list[str] | None = None, label: str | None = None,
               caption: str | None = None, fit: bool | None = None) -> str:
    """The semantic or the form table, for one direction group.

    `fit` wraps the tabular in the shrink-if-too-wide `\\resizebox` above, on by default for both
    kinds. Neither fits a single-column page at `\\small`, and an overfull hbox in a submission
    is a worse outcome than a slightly shrunk table. Pass `fit=False` to get the raw tabular and
    size it by hand.
    """
    if kind == "semantic":
        groups, types, footer = SEMANTIC_GROUPS, rubric.SEMANTIC_TYPES, SEMANTIC_FOOTER
    elif kind == "form":
        groups, types, footer = FORM_GROUPS, rubric.FORM_TYPES, FORM_FOOTER
    else:
        raise ValueError(f"kind must be 'semantic' or 'form', got {kind!r}")

    dirs, syss, table = select(results, group, systems)
    if not dirs:
        return f"% no rows for group={group}, kind={kind}\n"

    if fit is None:
        fit = True
    colspec = "ll" + "r" * (len(types) + 1)
    head_groups = ["", ""] + [r"\multicolumn{%d}{c}{%s}" % (len(rubric.DIMENSIONS[key]), name)
                              for name, key in groups] + [""]
    # cmidrules under each dimension group, so the reader can see which columns belong together.
    rules, col = [], 3
    for _name, key in groups:
        n = len(rubric.DIMENSIONS[key])
        rules.append(r"\cmidrule(lr){%d-%d}" % (col, col + n - 1))
        col += n
    head_cols = [r"\textbf{Dir.}", r"\textbf{Method}"] \
        + [r"\textbf{%s}" % rubric.ERROR_TYPES[t][1] for t in types] + [r"\textbf{Overall}"]

    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             # Caption above the table, abbreviation note below: the placement ICLR and most
             # ML venues use, and the only order in which the note reads as a note.
             r"\caption{%s}" % (caption or default_caption(kind, group, results)),
             r"\label{%s}" % (label or f"tab:submqm_{group or 'all'}_{kind}"),
             CAPTION_GAP]
    lines += [FIT_OPEN] if fit else []
    lines += [r"\begin{tabular}{%s}" % colspec, r"\toprule",
             " & ".join(head_groups) + r" \\",
             "".join(rules),
             " & ".join(head_cols) + r" \\", r"\midrule"]

    for di, direction in enumerate(dirs):
        present = [s for s in syss if (direction, s) in table]
        marks = {}
        for t in types + ["overall"]:
            vals = [(table[(direction, s)]["types"][t] if t != "overall"
                     else table[(direction, s)]["overall"]) for s in present]
            marks[t] = dict(zip(present, rank_marks(vals)))
        for si, system in enumerate(present):
            a = table[(direction, system)]
            first = (r"\multirow{%d}{*}{%s}" % (len(present), direction_label(direction))
                     if si == 0 else "")
            cells = [cell(a["types"][t], marks[t][system]) for t in types]
            cells.append(cell(a["overall"], marks["overall"][system]))
            lines.append(" & ".join([first, escape(system)] + cells) + r" \\")
        if di < len(dirs) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [FIT_CLOSE] if fit else []
    lines += footer_block(footer)
    lines += [r"\end{table}"]
    return "\n".join(lines) + "\n"


def default_caption(kind: str, group: str | None, results: dict) -> str:
    what = ("semantic error types (Terminology, Accuracy, Fluency)" if kind == "semantic"
            else "form error types (Linguistic Conventions, Technical, Locale Conventions, "
                 "Audience Appropriateness)")
    where = {"out_en": "translation out of English", "into_en": "translation into English",
             None: "all directions"}.get(group, str(group))
    judge = results.get("judge", {})
    return (f"SubMQM {what}, {where}. All values are penalties averaged over episodes; "
            f"lower is better. Best per column within a direction is shaded; second best "
            f"lightly shaded. Judge: \\texttt{{{escape(judge.get('model', 'n/a'))}}}; "
            f"window size {results.get('window_size', 'n/a')} subtitles.")


# --------------------------------------------------------------------------------
# aggregate table
# --------------------------------------------------------------------------------

def aggregate_table(results: dict, groups: list[str] | None = None,
                    systems: list[str] | None = None, label: str | None = None,
                    caption: str | None = None, fit: bool = True) -> str:
    """Per-dimension means over the directions of each group - the summary table.

    Reads `results["by_system"]`, which `evaluate.py` computed as a mean over directions, rather
    than re-averaging the per-direction rows here. One place computes the aggregate; this only
    formats it.
    """
    by = results.get("by_system") or {}
    if not by:
        return "% results file has no by_system aggregates\n"
    groups = groups or ["out_en", "into_en"]
    syss = systems or list(by)
    missing = [s for s in syss if s not in by]
    if missing:
        raise SystemExit(f"no aggregate for system(s): {', '.join(missing)}")

    dims = [d for d in rubric.DIMENSIONS]          # structural order: Term. Acc. Flu. Ling. ...
    head = [r"\textbf{Method}"] + [r"\textbf{%s}" % rubric.DIMENSION_LABELS[d] for d in dims] \
        + [r"\textbf{Overall}"]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{%s}" % (caption or
                                "SubMQM dimension penalties, averaged over directions. "
                                "Lower is better."),
             r"\label{%s}" % (label or "tab:submqm_aggregate"),
             CAPTION_GAP]
    lines += [FIT_OPEN] if fit else []
    lines += [r"\begin{tabular}{l%s}" % ("r" * (len(dims) + 1)), r"\toprule",
              " & ".join(head) + r" \\"]

    group_titles = {"out_en": "English $\\rightarrow$ 15 locales",
                    "into_en": "15 locales $\\rightarrow$ English",
                    "all": "all 30 directions"}
    for group in groups:
        rows = [(s, by[s].get(group, {})) for s in syss]
        rows = [(s, a) for s, a in rows if a.get("directions")]
        if not rows:
            continue
        n = rows[0][1]["directions"]
        lines += [r"\midrule",
                  r"\multicolumn{%d}{l}{\textit{%s (%d direction%s)}} \\" %
                  (len(dims) + 2, group_titles.get(group, group), n, "" if n == 1 else "s")]
        marks = {}
        for d in dims + ["overall"]:
            vals = [(a["dimensions"][d] if d != "overall" else a["overall"]) for _s, a in rows]
            marks[d] = dict(zip([s for s, _a in rows], rank_marks(vals)))
        for system, a in rows:
            cells = [cell(a["dimensions"][d], marks[d][system]) for d in dims]
            cells.append(cell(a["overall"], marks["overall"][system]))
            lines.append(" & ".join([escape(system)] + cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [FIT_CLOSE] if fit else []
    lines += footer_block(AGGREGATE_FOOTER)
    lines += [r"\end{table}"]
    return "\n".join(lines) + "\n"


def render_all(results: dict, out_dir: Path, systems: list[str] | None = None) -> list[Path]:
    """Every table the result sections need, one file each, plus the preamble."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "preamble.tex"]
    written[0].write_text(PREAMBLE + "\n", encoding="utf-8")
    for group in ("out_en", "into_en"):
        for kind in ("semantic", "form"):
            body = fine_table(results, kind, group, systems,
                              label=f"tab:submqm_{group}_{kind}")
            p = out_dir / f"submqm_{group}_{kind}.tex"
            p.write_text(body, encoding="utf-8")
            written.append(p)
    p = out_dir / "submqm_aggregate.tex"
    p.write_text(aggregate_table(results, systems=systems), encoding="utf-8")
    written.append(p)
    return written


# --------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------

def self_test() -> int:
    failures: list[str] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        if not cond:
            failures.append(f"{label}{': ' + detail if detail else ''}")

    def agg(overall: float, bump: dict | None = None) -> dict:
        types = {t: 0.0 for t in rubric.ALL_TYPES}
        types.update(bump or {})
        dims = {d: sum(types[t] for t in subs) / len(subs)
                for d, subs in rubric.DIMENSIONS.items()}
        return {"episodes": 2, "windows": 20, "types": types, "dimensions": dims,
                "overall": overall}

    results = {
        "judge": {"backend": "anthropic", "model": "claude-sonnet-4-6"},
        "window_size": 20,
        "rows": [
            {"direction": "en2zh", "system": "Base_line", "mode": "block",
             "aggregate": agg(2.0, {"mistranslation": 4.0, "miscapitalization": 0.0})},
            {"direction": "en2zh", "system": "Other", "mode": "block",
             "aggregate": agg(1.5, {"mistranslation": 3.0, "miscapitalization": 0.0})},
            {"direction": "en2zh", "system": "SMART", "mode": "block",
             "aggregate": agg(1.0, {"mistranslation": 2.0, "miscapitalization": 0.0})},
            {"direction": "en2de", "system": "Base_line", "mode": "block",
             "aggregate": agg(3.0, {"mistranslation": 5.0, "miscapitalization": 1.0})},
            {"direction": "en2de", "system": "SMART", "mode": "block",
             "aggregate": agg(1.2, {"mistranslation": 2.0, "miscapitalization": 1.0})},
            {"direction": "zh2en", "system": "SMART", "mode": "block",
             "aggregate": agg(0.9, {"naturalness": 1.0})},
            {"direction": "zh2en", "system": "Base_line", "mode": "block",
             "aggregate": agg(1.4, {"naturalness": 2.0})},
            {"direction": "en2ko", "system": "SMART", "mode": "block",
             "aggregate": {"episodes": 0}},        # a direction with nothing scored
        ],
        "by_system": {
            "SMART": {"out_en": {"directions": 2, "episodes": 4, "windows": 40,
                                 "types": {t: 0.0 for t in rubric.ALL_TYPES},
                                 "dimensions": {d: 1.0 for d in rubric.DIMENSIONS},
                                 "overall": 1.1},
                      "into_en": {"directions": 1, "episodes": 2, "windows": 20,
                                  "types": {t: 0.0 for t in rubric.ALL_TYPES},
                                  "dimensions": {d: 0.9 for d in rubric.DIMENSIONS},
                                  "overall": 0.9}},
            "Base_line": {"out_en": {"directions": 2, "episodes": 4, "windows": 40,
                                     "types": {t: 0.0 for t in rubric.ALL_TYPES},
                                     "dimensions": {d: 2.0 for d in rubric.DIMENSIONS},
                                     "overall": 2.5},
                          "into_en": {"directions": 1, "episodes": 2, "windows": 20,
                                      "types": {t: 0.0 for t in rubric.ALL_TYPES},
                                      "dimensions": {d: 1.4 for d in rubric.DIMENSIONS},
                                      "overall": 1.4}},
        },
    }

    sem = fine_table(results, "semantic", "out_en", ["Base_line", "Other", "SMART"])
    form = fine_table(results, "form", "out_en", ["Base_line", "Other", "SMART"])
    into = fine_table(results, "semantic", "into_en")
    aggr = aggregate_table(results, systems=["Base_line", "SMART"])

    # Column counts, from the header row.
    def header_width(tex: str) -> int:
        for line in tex.splitlines():
            if r"\textbf{Dir.}" in line or r"\textbf{Method}" in line:
                return line.count("&") + 1
        return -1

    check("semantic table is not 11 columns", header_width(sem) == 11, str(header_width(sem)))
    check("form table is not 14 columns", header_width(form) == 14, str(header_width(form)))
    check("aggregate table is not 9 columns", header_width(aggr) == 9, str(header_width(aggr)))

    # Column order must match the published header exactly.
    want_sem = "NameInc TermInc MisTrans UndTrans OvrTrans Coher Natur Vivid"
    want_form = ("MisPunc MisCap Gram Space LineBrk CharLim LineLim LocErr LangDet "
                 "Profan Formal")
    got_sem = " ".join(rubric.ERROR_TYPES[t][1] for t in rubric.SEMANTIC_TYPES)
    got_form = " ".join(rubric.ERROR_TYPES[t][1] for t in rubric.FORM_TYPES)
    check("semantic column order", got_sem == want_sem, got_sem)
    check("form column order", got_form == want_form, got_form)
    for abbrev in want_sem.split():
        check(f"semantic header lost {abbrev}", f"\\textbf{{{abbrev}}}" in sem)
    for abbrev in want_form.split():
        check(f"form header lost {abbrev}", f"\\textbf{{{abbrev}}}" in form)

    # Highlighting, read out of the actual cells rather than searched for in the whole table:
    # `2.00` is the worst Overall in en2zh but the *best* MisTrans, so a string search over the
    # table cannot tell a correct highlight from a misplaced one.
    def column(tex: str, system: str, index: int) -> str:
        """The `index`-th numeric cell of `system`'s row (0 = the first error type)."""
        for line in tex.splitlines():
            cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
            if len(cells) > 2 and cells[1] == escape(system):
                return cells[2 + index]
        return "<no row>"

    last = len(rubric.SEMANTIC_TYPES)                      # the Overall column
    mistrans = rubric.SEMANTIC_TYPES.index("mistranslation")
    check("best not given to the lowest Overall",
          column(sem, "SMART", last) == r"\bestcell{1.00}", column(sem, "SMART", last))
    check("second not given to the runner-up Overall",
          column(sem, "Other", last) == r"\secondcell{1.50}", column(sem, "Other", last))
    check("the worst Overall was highlighted",
          column(sem, "Base_line", last) == "2.00", column(sem, "Base_line", last))
    check("2.00 is the lowest MisTrans in this block and should be best",
          column(sem, "SMART", mistrans) == r"\bestcell{2.00}",
          column(sem, "SMART", mistrans))
    check("a column where every system is equal should be unshaded",
          column(sem, "SMART", 0) == "0.00" and column(sem, "Base_line", 0) == "0.00",
          f"{column(sem, 'SMART', 0)} / {column(sem, 'Base_line', 0)}")
    check("a direction with no scored episodes appeared", "en2ko" not in sem)
    check("into_en table picked up an out_en direction",
          "zh" in into and r"en$\rightarrow$de" not in into)
    check("out_en table picked up an into_en direction",
          "zh$\\rightarrow$en" not in sem)

    # Ranks run over distinct values: a tie for the lead marks everyone who tied, the next
    # distinct value is still second, and a column with nothing to compare gets no marks.
    for values, want in [
        ([1.0, 2.0, 3.0], ["best", "second", ""]),
        ([1.0, 1.0, 2.0, 3.0], ["best", "best", "second", ""]),
        ([1.0, 2.0, 2.0, 3.0], ["best", "second", "second", ""]),
        ([None, 3.0, 2.0, 1.0], ["", "", "second", "best"]),
        ([1.0, 1.0, 2.0], ["best", "best", ""]),      # 2 distinct: runner-up is also worst
        ([1.0, 2.0], ["best", ""]),                   # two systems: only the winner is shaded
        ([0.0, 0.0, 0.0], ["", "", ""]),              # nothing to distinguish
        ([1.5], [""]),                                # a lone system is not "best"
        ([None, None], ["", ""]),
        ([], []),
    ]:
        got = rank_marks(values)
        check(f"rank_marks({values})", got == want, f"got {got}, want {want}")

    # Escaping, multirow arity, group headers, footers.
    check("underscore not escaped", r"Base\_line" in sem and "Base_line" not in
          sem.replace(r"Base\_line", ""))
    check("multirow arity wrong for a 3-system direction",
          r"\multirow{3}{*}" in sem and r"\multirow{2}{*}" in sem)
    for name, _k in SEMANTIC_GROUPS:
        check(f"semantic group header lost {name}", name in sem)
    for name, _k in FORM_GROUPS:
        check(f"form group header lost {name}", name in form)
    check("semantic footer missing", "NameInc = Name Inconsistency" in sem)
    check("form footer missing", "LineLim = Exceeding Lines per Box" in form)
    check("aggregate footer missing", "Ling. = Linguistic Conventions" in aggr)
    check("judge model not in the caption", "claude-sonnet-4-6" in sem)
    check("window size not in the caption", "window size 20" in sem)
    check("aggregate lost a group heading",
          "English $\\rightarrow$ 15 locales" in aggr and "(2 directions)" in aggr)
    check("aggregate did not highlight the better system", r"\bestcell{1.10}" in aggr)

    # The ways a generated table fails to compile: unbalanced braces, rows of differing width,
    # a `\\` in vertical mode after \caption, an unmatched \resizebox, and environments that do
    # not close.
    for name, tex in (("semantic", sem), ("form", form), ("into_en semantic", into),
                      ("aggregate", aggr)):
        check(f"{name}: unbalanced braces", tex.count("{") == tex.count("}"),
              f"{tex.count('{')} vs {tex.count('}')}")
        body = [l for l in tex.splitlines()
                if l.endswith(r"\\") and "multicolumn" not in l and "cmidrule" not in l]
        widths = {l.count("&") + 1 for l in body}
        check(f"{name}: rows have differing column counts", len(widths) == 1, str(widths))
        check(f"{name}: tabular not closed",
              tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}") == 1)
        check(f"{name}: table env not closed",
              tex.count(r"\begin{table}") == tex.count(r"\end{table}") == 1)
        # Every `\\` must be inside the tabular. One after \caption is a compile error
        # ("There's no line here to end"), which is why the footer is a \parbox.
        after = tex.split(r"\end{tabular}")[-1]
        check(f"{name}: a line break survives after the tabular", r"\\" not in after,
              after.strip()[:80])
        check(f"{name}: footer is not a parbox", r"\parbox{\linewidth}{\footnotesize" in tex)
        # Caption above the table, note below - and each exactly once.
        check(f"{name}: caption is not above the tabular",
              tex.index(r"\caption{") < tex.index(r"\begin{tabular}")
              < tex.index(r"\parbox{\linewidth}"))
        check(f"{name}: caption or label emitted twice",
              tex.count(r"\caption{") == tex.count(r"\label{") == 1)
    # All three tables overflow a single-column page, so all three are wrapped - but in the
    # shrink-only form, which leaves a table that already fits at its natural size. A plain
    # \resizebox{\textwidth} would enlarge a small table past the body font.
    for name, tex in (("semantic", sem), ("form", form), ("aggregate", aggr)):
        check(f"{name}: not wrapped for width", FIT_OPEN in tex)
        check(f"{name}: wrapper would enlarge a narrow table",
              r"\resizebox{\textwidth}" not in tex)
    check("fit=False still emitted a resizebox",
          r"\resizebox" not in fine_table(results, "form", "out_en", fit=False)
          and r"\resizebox" not in aggregate_table(results, fit=False))

    import tempfile
    written = render_all(results, Path(tempfile.mkdtemp()) / "tables")
    check("render_all wrote the wrong number of files", len(written) == 6, str(len(written)))
    check("preamble does not define the highlight macros",
          r"\newcommand{\bestcell}" in written[0].read_text(encoding="utf-8"))

    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"PASSED: semantic 11 cols, form 14 cols, aggregate 9 cols; column order matches "
          f"the published headers; shared-rank highlighting, escaping, multirow arity, group "
          f"filtering, footers and brace balance all check out; render_all wrote "
          f"{len(written)} files")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render SubMQM result tables as LaTeX.")
    ap.add_argument("results", nargs="?", help="results.json from evaluate.py")
    ap.add_argument("-o", "--out", default=None, help="write every table into this directory")
    ap.add_argument("--kind", choices=("semantic", "form", "aggregate"), default=None)
    ap.add_argument("--group", choices=("out_en", "into_en", "all"), default=None)
    ap.add_argument("--systems", default=None,
                    help="comma-separated row order, e.g. Online,GPT,TransAgent,SMART")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.results:
        ap.error("a results.json is required (or use --self-test)")
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    systems = args.systems.split(",") if args.systems else None

    if args.out:
        written = render_all(results, Path(args.out), systems)
        for p in written:
            print(f"  {p}")
        return 0
    group = None if args.group in (None, "all") else args.group
    if args.kind == "aggregate":
        print(aggregate_table(results, systems=systems))
    elif args.kind:
        print(fine_table(results, args.kind, group, systems))
    else:
        print(PREAMBLE)
        for kind in ("semantic", "form"):
            print(fine_table(results, kind, group, systems))
        print(aggregate_table(results, systems=systems))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
