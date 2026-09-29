"""The thirty translation directions, and how to load the core module for one.

SMART ships one self-contained core module per direction, `mas_core_<direction>_v4.py`. That
is a deliberate choice rather than an accident of history: each file carries its own agent
prompts, its own judge criteria, its own cue lists and its own display constraints, all in
the target language's terms, so a direction can be edited or ablated without touching the
other twenty-nine. The cost is thirty files; the benefit is that nothing in a run is
parameterised by a locale variable that could be wrong.

This module is the index over them. It does three things:

  * names the thirty directions and the sixteen locales they are built from;
  * loads one direction's module by path, so the caller never has to know the filename;
  * reads a module's display constraints *without importing it*, so `--list` works on a
    machine with no model client installed and no credentials configured.

The last point is why `constraints()` uses `ast` rather than `import`. Importing a core
module constructs a model client at module scope, which fails without credentials; listing
what directions exist should not require being able to run them.

Usage:  python3 directions.py              # the thirty directions, with their limits
        python3 directions.py --check      # every direction has a module, and it parses
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

def _find_root() -> Path:
    """Where the thirty core modules live.

    Two layouts are supported because two exist. The release is flat - the modules sit next to
    this file - while the tree they are generated in keeps them in a `mas_core/` directory. The
    harness is developed against the second and shipped in the first, so it resolves either
    rather than being correct in one of them. `SMART_CORE_DIR` overrides both, for a run that
    keeps the modules somewhere else entirely.
    """
    here = Path(__file__).resolve().parent
    env = os.environ.get("SMART_CORE_DIR")
    candidates = [Path(env)] if env else []
    candidates += [here, here / "mas_core", here.parent / "mas_core"]
    for c in candidates:
        if (c / "mas_core_en2zh_v4.py").exists():
            return c
    return Path(env) if env else here      # nothing found: report against the default


ROOT = _find_root()


# --------------------------------------------------------------------------------
# locales
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Locale:
    key: str            # profile key, e.g. "es_419"
    code: str           # reporting code, e.g. "es-419"
    short: str          # two-letter code the research phase fills in, e.g. "es"
    name: str           # English name as it appears inside prompts
    qualified: str      # disambiguated name, where the region matters
    tag: str            # the direction-name fragment, e.g. "es_419" -> "es_419"; "zh" -> "zh"

    @property
    def is_english(self) -> bool:
        return self.code == "en"


# `tag` is what appears in a filename. Where one language has two locales the tag keeps the
# region (es_419/es_ES, pt_BR/pt_PT) because both ship; where it does not, the bare language
# code is used, which is the author's own convention in the eight files that predate this.
LOCALES: dict[str, Locale] = {l.key: l for l in [
    Locale("en",     "en",     "en", "English",    "English",                      "en"),
    Locale("zh_CN",  "zh-CN",  "zh", "Chinese",    "Mandarin Chinese",             "zh"),
    Locale("ko_KR",  "ko-KR",  "ko", "Korean",     "Korean",                       "ko"),
    Locale("de_DE",  "de-DE",  "de", "German",     "German",                       "de"),
    Locale("fr_FR",  "fr-FR",  "fr", "French",     "French",                       "fr"),
    Locale("it_IT",  "it-IT",  "it", "Italian",    "Italian",                      "it"),
    Locale("es_419", "es-419", "es", "Spanish",    "Latin American Spanish",       "es_419"),
    Locale("es_ES",  "es-ES",  "es", "Spanish",    "European (Peninsular) Spanish", "es_ES"),
    Locale("pt_BR",  "pt-BR",  "pt", "Portuguese", "Brazilian Portuguese",         "pt_BR"),
    Locale("pt_PT",  "pt-PT",  "pt", "Portuguese", "European Portuguese",          "pt_PT"),
    Locale("nl_NL",  "nl-NL",  "nl", "Dutch",      "Dutch",                        "nl"),
    Locale("sv_SE",  "sv-SE",  "sv", "Swedish",    "Swedish",                      "sv"),
    Locale("da_DK",  "da-DK",  "da", "Danish",     "Danish",                       "da"),
    Locale("no_NO",  "no-NO",  "no", "Norwegian",  "Bokmål Norwegian",        "no"),
    Locale("ro_RO",  "ro-RO",  "ro", "Romanian",   "Romanian",                     "ro"),
    Locale("tr_TR",  "tr-TR",  "tr", "Turkish",    "Turkish",                      "tr"),
]}

# The fifteen target locales, in the order the result tables use.
TARGETS: list[str] = ["zh_CN", "ko_KR", "de_DE", "fr_FR", "it_IT", "es_419", "es_ES",
                      "pt_BR", "pt_PT", "nl_NL", "sv_SE", "da_DK", "no_NO", "ro_RO",
                      "tr_TR"]


# --------------------------------------------------------------------------------
# directions
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Direction:
    name: str           # e.g. "en2zh", "es2en_419"
    src: str            # source locale key
    tgt: str            # target locale key

    @property
    def module_name(self) -> str:
        return f"mas_core_{self.name}_v4"

    @property
    def filename(self) -> str:
        return f"{self.module_name}.py"

    @property
    def group(self) -> str:
        """`out_en` or `into_en` - the two halves the result tables are split into."""
        return "out_en" if LOCALES[self.src].is_english else "into_en"

    @property
    def label(self) -> str:
        """The `Dir.` column, e.g. `en->zh-CN`."""
        return f"{LOCALES[self.src].code}->{LOCALES[self.tgt].code}"

    def path(self, root: Path | None = None) -> Path:
        return (root or ROOT) / self.filename


def _build_directions() -> list[Direction]:
    out = []
    for t in TARGETS:
        out.append(Direction(f"en2{LOCALES[t].tag}", "en", t))
    for t in TARGETS:
        # Into English the tag goes *after* the `2en`, which is the author's convention:
        # `es2en_419`, not `es_4192en`. It keeps the direction readable and the filename sane.
        tag = LOCALES[t].tag
        name = f"{tag}2en" if "_" not in tag else f"{tag.split('_')[0]}2en_{tag.split('_')[1]}"
        out.append(Direction(name, t, "en"))
    return out


DIRECTIONS: list[Direction] = _build_directions()
BY_NAME: dict[str, Direction] = {d.name: d for d in DIRECTIONS}

assert len(DIRECTIONS) == 30, f"expected 30 directions, got {len(DIRECTIONS)}"
assert len(BY_NAME) == 30, "two directions share a name"
assert len(TARGETS) == 15 and len(set(TARGETS)) == 15
assert set(TARGETS) | {"en"} == set(LOCALES), "TARGETS and LOCALES disagree"


def resolve(name: str) -> Direction:
    """A direction by name, accepting the unambiguous shorthands.

    `en2es` and `es2en` are ambiguous - two Spanish locales ship - so they are rejected with
    the alternatives named rather than silently resolved to es-419. Same for Portuguese.
    """
    if name in BY_NAME:
        return BY_NAME[name]
    near = sorted(n for n in BY_NAME if n.startswith(name))
    if len(near) == 1:
        return BY_NAME[near[0]]
    if near:
        raise SystemExit(f"{name!r} is ambiguous: {', '.join(near)}")
    raise SystemExit(f"unknown direction {name!r}. Known: {', '.join(sorted(BY_NAME))}")


# --------------------------------------------------------------------------------
# loading a core module
# --------------------------------------------------------------------------------

_LOADED: dict[str, object] = {}


def load(direction: str | Direction, root: Path | None = None):
    """Import one direction's core module and return it.

    Loaded by file path rather than by `import mas_core_en2zh_v4`, so the harness works from
    any working directory. The module's own directory goes on `sys.path` first, because a
    core module imports its search backend by bare name (`from web_search_requests import
    ...`) - the flat layout the cores were written against, preserved here so the thirty files
    are byte-identical to what a reviewer would get from the paper.

    `web_search/` goes on too, and this matters more than it looks. In a layout where the
    backends sit in their own directory, that bare-name import raises ImportError, and the cores
    answer an ImportError by falling through to a stub that returns "ERROR: No search backend
    available". Nothing crashes and nothing warns above WARNING: the run completes with the
    research tool silently dead, which is a different system from the one being measured. So the
    directory is added whenever it exists, and if research is meant to be off it should be turned
    off explicitly with SMART_NO_SEARCH=1.
    """
    d = direction if isinstance(direction, Direction) else resolve(direction)
    if d.name in _LOADED:
        return _LOADED[d.name]
    path = d.path(root)
    if not path.exists():
        raise SystemExit(f"{d.name}: no module at {path}")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    for candidate in (path.parent / "web_search", path.parent.parent / "web_search"):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break
    spec = importlib.util.spec_from_file_location(d.module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[d.module_name] = mod
    spec.loader.exec_module(mod)
    _LOADED[d.name] = mod
    return mod


# Module-scope constants worth reading without paying for an import.
_WANTED = ("MAX_CPS", "MAX_LINE", "MAX_TOKENS", "SLIDING_WINDOW", "RESEARCH_SAMPLE_SIZE",
           "DOC_REFINE_MAX_REWRITES", "DOC_REFINE_MIN_COVERAGE")


def constraints(direction: str | Direction, root: Path | None = None) -> dict:
    """A direction's display and budget constants, read by parsing rather than importing.

    Only plain literals are resolved. `MODEL_ID` is deliberately absent: in the shipped files
    it is an `os.environ.get(...)` call, so its value is a runtime fact and reporting a parsed
    default here would be reporting something that may not be what ran.
    """
    d = direction if isinstance(direction, Direction) else resolve(direction)
    tree = ast.parse(d.path(root).read_text(encoding="utf-8"))
    out: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in _WANTED:
            try:
                out[target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def main() -> int:
    missing = [d.name for d in DIRECTIONS if not d.path().exists()]
    if "--check" in sys.argv:
        bad = [f"no module: {n}" for n in missing]
        for d in DIRECTIONS:
            if d.name in missing:
                continue
            try:
                ast.parse(d.path().read_text(encoding="utf-8"))
            except SyntaxError as e:
                bad.append(f"{d.name}: line {e.lineno}: {e.msg}")
        if bad:
            print(f"FAILED ({len(bad)})")
            for b in bad:
                print(f"  {b}")
            return 1
        print(f"PASSED: all {len(DIRECTIONS)} direction modules present and parse")
        return 0

    print(f"{len(DIRECTIONS)} directions, {len(LOCALES)} locales\n")
    print(f"  {'direction':<11} {'dir.':<14} {'group':<8} {'cps':>4} {'line':>5}  module")
    for d in DIRECTIONS:
        if d.name in missing:
            print(f"  {d.name:<11} {d.label:<14} {d.group:<8} {'?':>4} {'?':>5}  "
                  f"MISSING {d.filename}")
            continue
        c = constraints(d)
        print(f"  {d.name:<11} {d.label:<14} {d.group:<8} {c.get('MAX_CPS', '?'):>4} "
              f"{c.get('MAX_LINE', '?'):>5}  {d.filename}")
    if missing:
        print(f"\n{len(missing)} module(s) missing")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
