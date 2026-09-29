#!/usr/bin/env python3
"""Gate the release tree: does every file compile, and is anything identifying left in it?

Run this before publishing, and again after any hand edit:

    python3 harness/tools/check_release.py                      # the tree this file lives in
    python3 harness/tools/check_release.py --root DIR
    python3 harness/tools/check_release.py --denylist FILE      # also scan for private strings

Four checks, each of which has caught something real during the build:

  compile     every `.py` parses. A release nobody can import is worse than no release.
  tokens      no credential - as a literal assignment, and as the shape of an AWS key id or an
              sk- key - no concrete `/home/<someone>` path, which names its owner, and no trace
              of the one cloud deployment the work happened to run on, which is a fingerprint
              rather than a secret and is not needed by the method.
  names       no series or character name from the private corpus. The benchmark is not
              published, so the code must not name it either. This one needs `--denylist`,
              because the names are exactly what this file must not contain; see below.
  comments    no *unquoted* non-Latin script in a comment or in shipped prose. Three things
              have to be told apart here, and a whole-file scan conflates all of them:
                - a string literal - the cores must contain Chinese, Korean and Turkish inside
                  prompts and punctuation tables, because that text is what the system does;
                - a quoted example - a comment explaining why `我 不 知 道` must have its spaces
                  stripped cannot do so without naming the artefact, and a diff quoting `」`
                  in a report is the same case. Backtick spans and fenced blocks are exempt;
                - prose in the author's own language - which identifies the author as surely
                  as a signature. That is the only one of the three this rejects.
  data        no subtitle corpus or answer key. Only `harness/examples/` may hold an `.srt`.

Exit status is 0 when the tree is publishable and 1 when it is not, so it works as a CI step.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import token as T
import tokenize
from pathlib import Path

# ------------------------------------------------------------------------------------
# what must never ship
#
# Everything here is a *generic* pattern. That is a deliberate constraint, and it is the one
# design decision in this file worth arguing about: an anonymity checker that carries a literal
# list of the things it is hiding discloses them to everyone who reads the checker. A built-in
# list of the corpus's series names would tell a reviewer which shows the unpublished benchmark is
# drawn from, and a built-in home directory would name the author - in the file whose entire
# purpose is to prevent exactly that. The build enforces this: it refuses to publish a tree whose
# checker contains any of the strings, which is how this very paragraph got rewritten.
#
# So the private strings do not live here. They live in the build tree, which is not published,
# and `_build/assemble.py` passes them in with `--denylist` when it gates the assembled release.
# A reviewer running this file with no `--denylist` still gets every check that does not require
# a secret to perform, which is all of them except the corpus-name scan.
# ------------------------------------------------------------------------------------

FORBIDDEN_TOKENS = [
    'AWS_ACCESS_KEY = "',       # the reference assigned both credentials as literals
    'AWS_SECRET_KEY = "',
    "chinese_fluency_check",    # a pre-rename tool name; names no person and no corpus
    "IDIOM_DB_DEPRECATED",      # a dead table of idiom entries, likewise

    # The deployment the work happened to run on. These are not secrets - they are a fingerprint.
    # The method needs one Anthropic API key and nothing else; the cores were originally written
    # against `boto3` and `bedrock-runtime` only because the authors' cluster reaches Claude
    # through one particular cloud, and leaving that in would both name the cloud and make
    # `MAS_AWS_ACCESS_KEY` look like part of the system's interface. `ANTHROPIC_PROVIDER=bedrock`
    # is the supported way to get the same transport back, so nothing is lost by banning the
    # hard-wired form - and a future edit that reintroduces it fails here instead of shipping.
    #
    # Note what is *not* banned: the bare word "bedrock". It is a legal value of
    # ANTHROPIC_PROVIDER and appears in both the README and the judge, which is the point.
    # What is banned is the machinery that makes it the only option.
    "MAS_AWS_",                 # the old credential env prefix
    "MAS_MODEL_ID",             # the old model env var
    "import boto3",             # the old transport
    "bedrock-runtime",          # the old client name
    "bedrock-2023-05-31",       # the API version pin the SDK now sets itself
    "us.anthropic.",            # a cloud-qualified model id; the plain id is portable
]

# An AWS access key id is AKIA followed by 16 uppercase alphanumerics. Matched as a pattern
# rather than as a prefix so that the word AKIA in a comment - including this one - is not a hit,
# while any actual key is, whatever it is assigned to.
CREDENTIAL_PATTERNS = [
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "an AWS access key id"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "an API key in the sk- family"),
    (re.compile(r"aws_secret_access_key\s*[=:]\s*['\"][A-Za-z0-9/+=]{30,}"), "an AWS secret key"),
]

# A concrete personal home directory, which names its owner. `/home/<user>` and `/Users/<user>`
# only - `/opt`, `/data`, `/mnt` and the like say nothing about who ran the code. Placeholders are
# exempt because the documentation needs to be able to show the shape of a path.
PERSONAL_PATH = re.compile(
    r"/(?:home|Users)/"
    r"(?!<|\$|\{|user\b|USER\b|username\b|youruser\b|someuser\b|me\b|ubuntu\b|runner\b)"
    r"[A-Za-z][A-Za-z0-9._-]{2,}")

# Filled from --denylist. Kept empty in the shipped file on purpose; see the note above.
FORBIDDEN_NAMES: list[str] = []

# CJK, Hangul, Kana and the CJK punctuation block. Not a general "non-ASCII" test: the cores
# are full of accented Latin, the em dash and the narrow no-break space, all of which are
# meaningful to a subtitle system and none of which says anything about who wrote it.
NON_LATIN = re.compile(
    "[　-〿"      # CJK punctuation
    "぀-ヿ"       # Hiragana, Katakana
    "㐀-䶿"       # CJK extension A
    "一-鿿"       # CJK unified ideographs
    "가-힯"       # Hangul syllables
    "＀-￯]"      # halfwidth and fullwidth forms
)

# Directories that are build scaffolding or caches, never part of the tree under test. `_build`
# and `specs` quote the author's original source verbatim - credentials, home paths and corpus
# names included - which is exactly why they are scaffolding and do not ship.
SKIP_DIRS = {"__pycache__", ".git", "_build", "specs", "release", ".ipynb_checkpoints"}

# The one place a subtitle file is allowed: a short public-domain sample so the harness can be
# run end to end without the benchmark.
DATA_SUFFIXES = {".srt", ".vtt", ".ass", ".sub"}
EXAMPLES_DIR = "harness/examples"


def files_under(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if p.suffix in {".pyc", ".pdf", ".zip"}:
            continue
        out.append(p)
    return out


# A fenced block, then an inline backtick span. Fences first, because a fenced block may well
# contain a stray single backtick that would otherwise swallow half the file.
_FENCE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1", re.S | re.M)
_INLINE = re.compile(r"`[^`\n]*`")


def strip_quoted(text: str) -> str:
    """Blank out fenced blocks and backtick spans, keeping line numbers intact.

    Replacement is space-for-character rather than deletion so the line a problem is reported on
    is still the line it is on in the file.
    """
    def blank(m: re.Match) -> str:
        return "".join("\n" if c == "\n" else " " for c in m.group(0))
    return _INLINE.sub(blank, _FENCE.sub(blank, text))


def comment_text(source: str) -> list[tuple[int, str]]:
    """Every comment in `source` as (line number, text).

    Docstrings are `STRING` tokens, not `COMMENT`, and are left out on purpose: a target-language
    example inside a prompt docstring is part of the system's behaviour.
    """
    out = []
    try:
        for t in tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__):
            if t.type == T.COMMENT:
                out.append((t.start[0], t.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # A file that will not tokenise fails the compile check; do not report it twice.
        pass
    return out


def check(root: Path, verbose: bool = False) -> list[str]:
    problems: list[str] = []
    paths = files_under(root)
    if not paths:
        return [f"{root}: nothing to check - is the path right?"]

    # This file necessarily contains every string it bans, so scanning it would always fail.
    # Matched by resolved path rather than by name: a copy under another name still gets checked.
    me = Path(__file__).resolve()

    n_py = 0
    for path in paths:
        rel = path.relative_to(root).as_posix()
        if path.resolve() == me:
            continue

        if path.suffix in DATA_SUFFIXES and not rel.startswith(EXAMPLES_DIR):
            problems.append(f"{rel}: benchmark data - the release is code only "
                            f"(only {EXAMPLES_DIR}/ may hold a subtitle file)")
            continue

        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{rel}: not UTF-8 text; a release tree should not carry binaries")
            continue

        if path.suffix == ".py":
            n_py += 1
            try:
                ast.parse(source, filename=rel)
            except SyntaxError as e:
                problems.append(f"{rel}:{e.lineno}: does not compile: {e.msg}")

        # Tokens, patterns and names, by line, so the report points at something.
        for i, line in enumerate(source.splitlines(), 1):
            for bad in FORBIDDEN_TOKENS:
                if bad in line:
                    problems.append(f"{rel}:{i}: forbidden token {bad!r}")
            for pattern, what in CREDENTIAL_PATTERNS:
                if pattern.search(line):
                    problems.append(f"{rel}:{i}: looks like {what}")
            if (m := PERSONAL_PATH.search(line)):
                problems.append(f"{rel}:{i}: personal home directory {m.group(0)!r}")
            for name in FORBIDDEN_NAMES:
                if re.search(rf"\b{re.escape(name)}\b" if name.isascii() else re.escape(name),
                             line):
                    # The name itself is not echoed: this report may be pasted somewhere.
                    problems.append(f"{rel}:{i}: a denylisted corpus name "
                                    f"(entry {FORBIDDEN_NAMES.index(name) + 1} "
                                    f"of {len(FORBIDDEN_NAMES)})")

        if path.suffix == ".py":
            for lineno, text in comment_text(source):
                if NON_LATIN.search(strip_quoted(text)):
                    problems.append(f"{rel}:{lineno}: unquoted non-Latin script in a comment: "
                                    f"{text.strip()[:60]}")
        elif path.suffix in {".md", ".txt"}:
            for i, line in enumerate(strip_quoted(source).splitlines(), 1):
                if NON_LATIN.search(line):
                    problems.append(f"{rel}:{i}: unquoted non-Latin script in shipped prose")

    if verbose:
        print(f"  checked {len(paths)} files ({n_py} Python) under {root}")
    return problems


def self_test() -> int:
    """Each case is a way the gate could be wrong, not a way the tree could be.

    A gate that passes everything is indistinguishable from no gate, so the cases that matter
    most are the ones that must *fail*.
    """
    import tempfile

    cases: list[tuple[str, str, bool, str]] = [
        # (filename, content, should_pass, why)
        ("ok.py", "X = 1  # a plain comment\n", True, "clean file"),
        ("key.py", 'AWS_ACCESS_KEY = "redacted"\n', False, "credential assigned as a literal"),
        ("akia.py", 'K = "AKIAQ4SZX7NRTVBWCDEF"\n', False,
         "a full-length AWS key id, whatever it is assigned to"),
        ("akia_word.py", "# an AKIA prefix named in prose is not a key\n", True,
         "matching the pattern rather than the prefix keeps this file from failing itself"),
        ("sk.py", 'K = "sk-abcdefghijklmnopqrstuvwxyz123"\n', False, "an sk- family API key"),
        ("path.py", "P = '/home/adevname/data'\n", False, "a personal home directory"),
        ("path_ph.py", "P = '/home/<user>/data'  # and /Users/$USER, /home/someuser\n", True,
         "placeholders must stay writable; documentation needs the shape of a path"),
        ("path_other.py", "P = '/opt/smart/data'\n", True,
         "/opt and /data name no one and must not be flagged"),
        ("zh_comment.py", "X = 1  # 这是中文注释\n", False, "Chinese comment"),
        ("zh_quoted.py", "X = 1  # strip the spaces in `我 不 知 道` first\n", True,
         "a backticked example is the subject matter; the comment cannot be written without it"),
        ("zh_half.py", "X = 1  # `我 不 知 道` 然后呢\n", False,
         "quoting one example does not license prose around it"),
        ("md_fenced.md", "A diff:\n\n```\n- t.rstrip('」')\n```\n", True,
         "a fenced block in a report is quoted code"),
        ("md_prose.md", "# Notes\n\n这是中文说明\n", False, "Chinese prose in a shipped document"),
        ("md_line.md", "ok\nok\n这里\n", False,
         "the report must name the offending line, not just the file"),
        ("zh_string.py", 'PUNCT = "，。！"  # target punctuation\n', True,
         "target-language text in a literal is the system's behaviour, not an author trace"),
        ("zh_doc.py", '"""Prompt: 翻译成中文."""\n', True,
         "a docstring is not a comment; prompts legitimately hold target text"),
        ("broken.py", "def f(:\n", False, "syntax error"),
        ("accents.py", "S = 'crème brûlée — naïve'  # Latin-1 and an em dash\n", True,
         "accented Latin and punctuation must not be mistaken for a non-Latin script"),
        ("nbsp.py", "S = 'Ne pas !'\n", True, "narrow no-break space survives"),
        ("old.py", "def chinese_fluency_check():\n    pass\n", False, "pre-rename symbol"),

        # The deployment fingerprint. Each of these shipped at some point and each is now banned.
        ("aws_env.py", 'K = os.environ.get("MAS_AWS_ACCESS_KEY")\n', False,
         "the old credential env prefix"),
        ("aws_model.py", 'M = os.environ.get("MAS_MODEL_ID")\n', False, "the old model env var"),
        ("aws_import.py", "import boto3\n", False, "the old transport"),
        ("aws_client.py", 'c = client("bedrock-runtime")\n', False, "the old client name"),
        ("aws_version.py", 'B = {"anthropic_version": "bedrock-2023-05-31"}\n', False,
         "the API version pin the SDK sets itself"),
        ("aws_model_id.py", 'M = "us.anthropic.claude-sonnet-4-6"\n', False,
         "a cloud-qualified model id"),
        ("provider_ok.py", 'P = os.environ.get("ANTHROPIC_PROVIDER", "anthropic")\n', True,
         "the generic form must stay writable"),
        ("provider_bedrock.py", 'if provider == "bedrock":\n    pass\n', True,
         "bedrock as a *value* is a supported option, not a fingerprint; banning the word would "
         "make the portable form undocumentable"),
    ]

    failures = []
    for fname, content, should_pass, why in cases:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / fname).write_text(content, encoding="utf-8")
            got = not check(Path(td))
        if got != should_pass:
            failures.append(f"{fname}: expected {'pass' if should_pass else 'FAIL'}, "
                            f"got {'pass' if got else 'FAIL'} - {why}")

    # The denylist is supplied, not built in, so the mechanism has to be tested with a supplied
    # one. A neutral placeholder stands in for the private names, which is the whole point.
    global FORBIDDEN_NAMES
    saved_names = FORBIDDEN_NAMES
    FORBIDDEN_NAMES = ["Zephyrine"]
    try:
        for content, should_pass, why in [
            ('SERIES = "Zephyrine"\n', False, "a denylisted name is caught"),
            ("# Zephyrines are fine\n", True, "the match is word-bounded"),
            ('S = "Quillfeather"\n', True,
             "nothing private is built in; an unsupplied name cannot be checked for"),
        ]:
            with tempfile.TemporaryDirectory() as td:
                (Path(td) / "d.py").write_text(content, encoding="utf-8")
                got = check(Path(td))
            if bool(got) == should_pass:
                failures.append(f"denylist: {why} - got {got}")
            elif got and "Zephyrine" in got[0]:
                failures.append("the report echoed the denylisted name it is meant to withhold")
    finally:
        FORBIDDEN_NAMES = saved_names

    # `strip_quoted` blanks character-for-character rather than deleting, so that a problem after
    # a fenced block is still reported on its real line. Nothing else checks that.
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "shift.md").write_text("intro\n\n```\n」\n」\n```\n\n这里\n", encoding="utf-8")
        got = check(Path(td))
        if len(got) != 1 or not got[0].startswith("shift.md:8:"):
            failures.append(f"line numbers shifted past a fenced block: expected one hit on "
                            f"line 8, got {got}")

    # Data placement: an .srt is a failure anywhere but harness/examples/.
    for rel, should_pass in [("harness/examples/example_en.srt", True),
                             ("episode.srt", False),
                             ("data/season1/ep01.srt", False)]:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")
            got = not check(Path(td))
        if got != should_pass:
            failures.append(f"{rel}: expected {'pass' if should_pass else 'FAIL'}")

    # An empty tree must not silently pass: "nothing found" would look like "nothing wrong".
    with tempfile.TemporaryDirectory() as td:
        if not check(Path(td)):
            failures.append("an empty directory passed; a no-op gate reads as a clean gate")

    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"PASSED: {len(cases) + 9} gate cases - credential literals and key patterns, personal "
          f"home directories, supplied denylist names, pre-rename symbols, the six traces of the "
          f"authors' cloud deployment and unquoted Chinese prose are caught; path placeholders, "
          f"non-personal absolute paths, target-language literals, docstrings, backticked "
          f"examples, fenced diffs, accented Latin, no-break spaces and the portable provider "
          f"form are not; the denylist is not built in and is never echoed back; line numbers "
          f"survive a fenced block; misplaced subtitle data is caught and the example is not; "
          f"an empty tree fails")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="tree to check (default: the repository this file is in)")
    ap.add_argument("-q", "--quiet", action="store_true", help="print only the verdict")
    ap.add_argument("--denylist", default=None,
                    help="file of private strings, one per line, to scan for as well "
                         "(the build passes the corpus names here; see the note in this file "
                         "on why they are not built in)")
    ap.add_argument("--self-test", action="store_true", help="check the gate itself")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.denylist:
        path = Path(args.denylist)
        if not path.exists():
            raise SystemExit(f"no such denylist: {path}")
        global FORBIDDEN_NAMES
        FORBIDDEN_NAMES = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
                           if l.strip() and not l.startswith("#")]
        if not args.quiet:
            print(f"  denylist: {len(FORBIDDEN_NAMES)} private string(s) loaded from {path.name}")

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[2]
    problems = check(root, verbose=not args.quiet)
    if problems:
        print(f"NOT PUBLISHABLE ({len(problems)} problem"
              f"{'' if len(problems) == 1 else 's'})")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"PUBLISHABLE: {root} compiles clean and carries no credential, author path, corpus "
          f"name, pre-rename symbol, non-Latin comment or benchmark data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
