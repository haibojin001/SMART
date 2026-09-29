"""Score many (system, direction, episode) combinations and aggregate them into the tables.

`judge.py` scores one episode. This is the sweep: it walks a job description, judges every
episode of every system in every direction, and writes the aggregates the result tables are
built from. It is the step that has to be re-runnable, because the tables get rebuilt every time
an aggregation detail changes, and re-judging 30 directions from scratch each time is not
affordable. Hence the disk cache in `judge.py` and hence this file's rule that nothing is
overwritten in place.

Three decisions worth stating, because each is a place where a plausible alternative would
quietly bias the numbers:

    A system's alignment mode is declared, not guessed. Systems that translate the source
    block-by-block are judged block-by-block; independently segmented references are judged at
    the passage level. Guessing from cue counts would silently switch a system to the lenient
    mode whenever it happened to drop a cue, so the job file says which, and the result records
    which was used.

    Aggregation is a mean over episodes, then a mean over directions. Not a pooled mean over
    windows: a 900-segment episode would otherwise outvote a 300-segment one, and a direction
    with more test episodes would outvote the rest of the table. `rubric.mean_episodes` does
    the first; `aggregate_directions` does the second.

    A run with too many unjudgeable windows is not published. Windows the judge could not be
    made to score are dropped rather than zeroed - a zero is indistinguishable from a perfect
    window, so dropping is the only honest option - but dropping enough of them changes what is
    being measured. Above `--max-failure-rate` the record is marked `excluded` and kept out of
    the aggregates, with the reason in the output rather than in a log nobody reads.

    python3 evaluate.py --job jobs/full_sweep.json -o results/
    python3 evaluate.py --direction en2zh --source bench/en/ --system SMART=runs/smart/en2zh \\
                        --system Online=bench/online/zh --mode Online=passage -o results/
    python3 evaluate.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

if __package__:
    from . import align, judge, rubric
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import align
    import judge
    import rubric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("submqm.evaluate")

# Above this fraction of unjudgeable windows an episode is excluded from the aggregates.
DEFAULT_MAX_FAILURE_RATE = 0.05

_SXXEYY = re.compile(r"[sS](\d{1,2})[ ._-]?[eE](\d{1,3})")
_NxM = re.compile(r"(?<!\d)(\d{1,2})x(\d{1,3})(?!\d)")


def episode_key(path: Path) -> str:
    """A stable id that survives differing filenames across systems.

    Two systems' output for the same episode rarely share a filename - one is
    `S01E03.srt`, another `TheSeries.S01E03.1080p.zh.srt` - so pairing has to go through the
    season/episode numbers. Where they are absent the stem is used, which pairs only files that
    were named the same way; that is reported as a match failure rather than assumed to be fine.
    """
    m = _SXXEYY.search(path.name) or _NxM.search(path.name)
    if m:
        return f"S{int(m.group(1)):02d}E{int(m.group(2)):02d}"
    return path.stem


def index_episodes(directory: str | Path) -> dict[str, Path]:
    d = Path(directory)
    if d.is_file():
        return {episode_key(d): d}
    out: dict[str, Path] = {}
    for p in sorted(d.rglob("*.srt")):
        out.setdefault(episode_key(p), p)
    return out


# --------------------------------------------------------------------------------
# one (system, direction)
# --------------------------------------------------------------------------------

def evaluate_system(direction: str, system: str, source_index: dict[str, Path],
                    target_dir: str | Path, *, mode: str, backend: judge.Backend,
                    window: int, overlap: float, cache_dir: Path | None,
                    raw_root: Path | None, max_failure_rate: float,
                    normalise: bool | None = None, limit: int | None = None) -> dict:
    """Judge every episode this system has in this direction, and aggregate them."""
    targets = index_episodes(target_dir)
    shared = [k for k in sorted(source_index) if k in targets]
    missing = [k for k in sorted(source_index) if k not in targets]
    extra = [k for k in sorted(targets) if k not in source_index]
    if limit:
        shared = shared[:limit]
    logger.info("  %s / %s: %d episode(s) matched, %d source-only, %d target-only",
                direction, system, len(shared), len(missing), len(extra))

    episodes, excluded = [], []
    for key in shared:
        rec = judge.judge_episode(
            source_index[key], targets[key], direction=direction, backend=backend, mode=mode,
            window=window, overlap=overlap, normalise=normalise,
            cache_dir=cache_dir,
            raw_dir=(raw_root / direction / system / key) if raw_root else None)
        rec["episode"] = key
        rec["system"] = system
        total = rec["windows_total"] or 1
        rate = rec["windows_failed"] / total
        rec["failure_rate"] = round(rate, 4)
        if rec["windows_scored"] == 0 or rate > max_failure_rate:
            rec["excluded"] = (f"{rec['windows_failed']}/{rec['windows_total']} windows "
                               f"unjudgeable ({rate:.1%} > {max_failure_rate:.0%})")
            excluded.append(rec)
            logger.warning("    %s excluded: %s", key, rec["excluded"])
        else:
            episodes.append(rec)
        logger.info("    %s: Overall %.2f over %d/%d windows%s", key,
                    rec["score"]["overall"], rec["windows_scored"], rec["windows_total"],
                    " [EXCLUDED]" if "excluded" in rec else "")

    agg = rubric.mean_episodes([e["score"] for e in episodes])
    return {
        "direction": direction, "system": system, "mode": mode,
        "window_size": window, "window_overlap": overlap,
        "judge": backend.describe(),
        "episodes_matched": len(shared), "episodes_scored": len(episodes),
        "episodes_excluded": len(excluded),
        "source_only": missing, "target_only": extra,
        "aggregate": agg,
        "per_episode": [{"episode": e["episode"], "windows": e["windows_scored"],
                         "overall": e["score"]["overall"],
                         "dimensions": e["score"]["dimensions"],
                         "types": e["score"]["types"],
                         "alignment": e["alignment"]} for e in episodes],
        "excluded": [{"episode": e["episode"], "reason": e["excluded"],
                      "failed": e["failed"]} for e in excluded],
    }


def aggregate_directions(rows: list[dict]) -> dict:
    """Mean over directions for one system, unweighted.

    The paper's headline numbers are means over the fifteen directions of a group, so a
    direction is one vote regardless of how many test episodes it has. Directions the system
    scored nothing in are left out entirely rather than counted as zero - a system that failed
    on en->ko must not be rewarded with a 0.00 penalty there.
    """
    live = [r for r in rows if r["aggregate"]["episodes"]]
    if not live:
        return {"directions": 0}
    n = len(live)
    return {
        "directions": n,
        "direction_names": [r["direction"] for r in live],
        "episodes": sum(r["aggregate"]["episodes"] for r in live),
        "windows": sum(r["aggregate"]["windows"] for r in live),
        "types": {t: sum(r["aggregate"]["types"][t] for r in live) / n
                  for t in rubric.ALL_TYPES},
        "dimensions": {d: sum(r["aggregate"]["dimensions"][d] for r in live) / n
                       for d in rubric.DIMENSIONS},
        "overall": sum(r["aggregate"]["overall"] for r in live) / n,
    }


# --------------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------------

def run_job(job: dict, out_dir: Path) -> dict:
    """Judge everything the job describes. Writes as it goes, so a crash keeps what ran."""
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = judge.make_backend(job.get("judge", {}).get("backend", "anthropic"),
                                 job.get("judge", {}).get("model"))
    window = int(job.get("window", align.DEFAULT_WINDOW))
    overlap = float(job.get("overlap", align.DEFAULT_OVERLAP))
    cache_dir = Path(job["cache_dir"]) if job.get("cache_dir") else out_dir / "cache"
    raw_root = Path(job["raw_dir"]) if job.get("raw_dir") else None
    max_fr = float(job.get("max_failure_rate", DEFAULT_MAX_FAILURE_RATE))
    limit = job.get("episodes_per_direction")

    results = {
        "judge": backend.describe(),
        "window_size": window, "window_overlap": overlap,
        "max_failure_rate": max_fr,
        "episodes_per_direction": limit,
        "cache_dir": str(cache_dir),
        "rows": [],
    }
    out_path = out_dir / "results.json"

    for direction, spec in job["directions"].items():
        source_index = index_episodes(spec["source_dir"])
        logger.info("%s: %d source episode(s) in %s", direction, len(source_index),
                    spec["source_dir"])
        for system, sysspec in spec["systems"].items():
            row = evaluate_system(
                direction, system, source_index, sysspec["dir"],
                mode=sysspec.get("mode", "block"), backend=backend, window=window,
                overlap=overlap, cache_dir=cache_dir, raw_root=raw_root,
                max_failure_rate=max_fr,
                normalise=sysspec.get("normalise"), limit=limit)
            results["rows"].append(row)
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    results["by_system"] = {}
    systems = sorted({r["system"] for r in results["rows"]})
    for system in systems:
        rows = [r for r in results["rows"] if r["system"] == system]
        results["by_system"][system] = {
            "all": aggregate_directions(rows),
            # The paper reports the two halves separately, because translating out of English
            # and into English are different problems; a combined mean hides that.
            "out_en": aggregate_directions([r for r in rows
                                            if r["direction"].startswith("en2")]),
            "into_en": aggregate_directions([r for r in rows
                                             if not r["direction"].startswith("en2")]),
        }
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(results, out_dir / "summary.csv")
    logger.info("results: %s", out_path)
    return results


def write_csv(results: dict, path: Path) -> Path:
    """A flat per-(system, direction) table: every number the LaTeX tables use, plus counts.

    Written alongside the JSON because refilling a table by hand from nested JSON is where
    transcription errors come from, and because a spreadsheet is the fastest way to notice that
    one direction is an order of magnitude off.
    """
    cols = (["system", "direction", "mode", "episodes", "windows"]
            + [rubric.ERROR_TYPES[t][1] for t in rubric.ALL_TYPES]
            + [rubric.DIMENSION_LABELS[d] for d in rubric.DIMENSIONS] + ["Overall"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results["rows"]:
            a = r["aggregate"]
            if not a["episodes"]:
                continue
            w.writerow([r["system"], r["direction"], r["mode"], a["episodes"], a["windows"]]
                       + [f"{a['types'][t]:.2f}" for t in rubric.ALL_TYPES]
                       + [f"{a['dimensions'][d]:.2f}" for d in rubric.DIMENSIONS]
                       + [f"{a['overall']:.2f}"])
        for system, groups in results.get("by_system", {}).items():
            for group, a in groups.items():
                if not a.get("directions"):
                    continue
                w.writerow([system, f"MEAN[{group}]", f"{a['directions']} dirs", a["episodes"],
                            a["windows"]]
                           + [f"{a['types'][t]:.2f}" for t in rubric.ALL_TYPES]
                           + [f"{a['dimensions'][d]:.2f}" for d in rubric.DIMENSIONS]
                           + [f"{a['overall']:.2f}"])
    return path


def job_from_args(args) -> dict:
    """Build a one-direction job from the CLI flags, for a single evaluation."""
    modes = dict(kv.split("=", 1) for kv in args.mode or [])
    systems = {}
    for entry in args.system:
        if "=" not in entry:
            raise SystemExit(f"--system wants NAME=path, got {entry!r}")
        name, path = entry.split("=", 1)
        systems[name] = {"dir": path, "mode": modes.get(name, "block")}
    return {
        "judge": {"backend": args.judge, "model": args.judge_model},
        "window": args.window, "overlap": args.overlap,
        "cache_dir": args.cache_dir, "raw_dir": args.raw_dir,
        "max_failure_rate": args.max_failure_rate,
        "episodes_per_direction": args.limit,
        "directions": {args.direction: {"source_dir": args.source, "systems": systems}},
    }


# --------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------

def self_test() -> int:
    """The whole sweep on synthetic files with a stub judge. No credentials needed."""
    import tempfile
    failures: list[str] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        if not cond:
            failures.append(f"{label}{': ' + detail if detail else ''}")

    tmp = Path(tempfile.mkdtemp())
    src_dir, good_dir, bad_dir, ref_dir = (tmp / n for n in ("src", "good", "bad", "ref"))
    for d in (src_dir, good_dir, bad_dir, ref_dir):
        d.mkdir()
    for ep in (1, 2):
        cues = "\n\n".join(
            f"{i}\n00:0{i // 60}:{i % 60:02d},000 --> 00:0{i // 60}:{i % 60:02d},900\n"
            f"source line {i}" for i in range(1, 11))
        (src_dir / f"Show.S01E{ep:02d}.srt").write_text(cues + "\n", encoding="utf-8")
        (good_dir / f"S01E{ep:02d}.srt").write_text(
            cues.replace("source", "target") + "\n", encoding="utf-8")
        (bad_dir / f"sys2_1x{ep}.srt").write_text(
            cues.replace("source", "other") + "\n", encoding="utf-8")
        # An independently segmented reference: half as many cues, tokenised text.
        ref = "\n\n".join(
            f"{i}\n00:00:{2 * i - 1:02d},000 --> 00:00:{2 * i:02d},900\n我 不 知 道 {i} ， 是 的 。"
            for i in range(1, 6))
        (ref_dir / f"S01E{ep:02d}.srt").write_text(ref + "\n", encoding="utf-8")

    class Fixed(judge.StubBackend):
        """Different systems get different penalties, so best/second-best is meaningful."""

        def __init__(self, table):
            super().__init__()
            self.table = table

        def complete(self, system, user, temperature=0.0):
            self.calls += 1
            which = "other" if "other line" in user else ("我不知道" if "我不知道" in user
                                                          else "target")
            return json.dumps({t: self.table.get(which, {}).get(t, 0)
                               for t in rubric.ALL_TYPES})

    backend = Fixed({"target": {"mistranslation": 5},
                     "other": {"mistranslation": 10, "grammar": 5},
                     "我不知道": {"naturalness": 5}})
    job = {
        "judge": {"backend": "stub"}, "window": 5, "cache_dir": str(tmp / "cache"),
        "directions": {"en2zh": {"source_dir": str(src_dir), "systems": {
            "SMART": {"dir": str(good_dir), "mode": "block"},
            "Baseline": {"dir": str(bad_dir), "mode": "block"},
            "Online": {"dir": str(ref_dir), "mode": "passage"},
        }}},
    }
    out = tmp / "out"
    # Inject the scripted backend rather than letting run_job build a plain stub.
    real_make = judge.make_backend
    judge.make_backend = lambda *a, **k: backend
    try:
        results = run_job(job, out)
    finally:
        judge.make_backend = real_make

    check("wrong row count", len(results["rows"]) == 3, str(len(results["rows"])))
    by = {r["system"]: r for r in results["rows"]}
    check("episode pairing across differing filenames failed",
          all(r["episodes_scored"] == 2 for r in results["rows"]),
          json.dumps({k: v["episodes_scored"] for k, v in by.items()}))
    check("passage mode not recorded", by["Online"]["mode"] == "passage")
    # SMART mistranslation 5 over every window -> accuracy = 5/3
    check("SMART accuracy", abs(by["SMART"]["aggregate"]["dimensions"]["accuracy"] - 5 / 3)
          < 1e-9, str(by["SMART"]["aggregate"]["dimensions"]["accuracy"]))
    check("Baseline must score worse than SMART",
          by["Baseline"]["aggregate"]["overall"] > by["SMART"]["aggregate"]["overall"])
    check("reference normalisation did not run",
          all(e["alignment"]["normalised_cues"] > 0 for e in by["Online"]["per_episode"]),
          json.dumps([e["alignment"]["normalised_cues"] for e in by["Online"]["per_episode"]]))
    check("hypothesis was normalised, which it must not be",
          all(e["alignment"]["normalised_cues"] == 0 for e in by["SMART"]["per_episode"]))
    check("by_system missing", set(results["by_system"]) == {"SMART", "Baseline", "Online"})
    check("out_en grouping wrong",
          results["by_system"]["SMART"]["out_en"]["directions"] == 1
          and results["by_system"]["SMART"]["into_en"].get("directions", 0) == 0)
    check("csv not written", (out / "summary.csv").exists())
    rows = list(csv.reader((out / "summary.csv").open(encoding="utf-8")))
    check("csv header width", len(rows[0]) == 5 + 19 + 7 + 1, str(len(rows[0])))
    check("csv has the mean rows", any(c[1].startswith("MEAN[") for c in rows[1:]))

    # Exclusion: a judge that fails everything must be kept out of the aggregate.
    class Broken(judge.Backend):
        name, model = "broken", "broken"

        def complete(self, system, user, temperature=0.0):
            raise RuntimeError("no")

    row = evaluate_system("en2zh", "Broken", index_episodes(src_dir), good_dir, mode="block",
                          backend=Broken(), window=5, overlap=0.0, cache_dir=None,
                          raw_root=None, max_failure_rate=DEFAULT_MAX_FAILURE_RATE)
    check("a fully failed system was still scored", row["episodes_scored"] == 0)
    check("exclusions not recorded", row["episodes_excluded"] == 2)
    check("a fully failed system was aggregated", row["aggregate"]["episodes"] == 0)
    check("aggregate_directions counted an empty direction",
          aggregate_directions([row]).get("directions", 0) == 0)

    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"PASSED: 3 systems x 1 direction x 2 episodes swept; episode pairing across "
          f"differing filenames, per-system alignment mode, reference-only normalisation, "
          f"per-direction and per-group aggregation, CSV width {len(rows[0])}, and "
          f"failed-run exclusion all check out")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score systems against the SubMQM rubric.")
    ap.add_argument("--job", help="a JSON job describing directions and systems")
    ap.add_argument("--direction", default=None)
    ap.add_argument("--source", default=None, help="directory of source episodes")
    ap.add_argument("--system", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--mode", action="append", default=[], metavar="NAME=block|passage",
                    help="alignment mode per system; default block")
    ap.add_argument("--judge", choices=("anthropic", "bedrock", "vertex", "openai", "stub"),
                    default="anthropic")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--window", type=int, default=align.DEFAULT_WINDOW)
    ap.add_argument("--overlap", type=float, default=align.DEFAULT_OVERLAP)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N episodes of each direction")
    ap.add_argument("--max-failure-rate", type=float, default=DEFAULT_MAX_FAILURE_RATE)
    ap.add_argument("-o", "--out", default="results")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.job:
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    else:
        if not (args.direction and args.source and args.system):
            ap.error("either --job, or --direction with --source and at least one --system")
        job = job_from_args(args)

    results = run_job(job, Path(args.out))
    print()
    for system, groups in results.get("by_system", {}).items():
        for group, a in groups.items():
            if a.get("directions"):
                print(f"  {system:<24} {group:<8} {a['directions']:>2} dir  "
                      f"{a['episodes']:>3} ep  Overall {a['overall']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
