"""Run SMART over one episode, or over a whole series in order.

The thirty core modules have no `__main__`: each is a library holding one direction's agents,
tools, router, judge and refiner. This is the entry point that drives them.

    Why series mode is the default shape. SMART's persistent memory M is defined per series,
    not per file - terminology, character voices, corrections and background accumulate across
    episodes and are the reason a later episode is translated better than the first. Running
    episodes one at a time with a fresh memory each time measures a different system. So
    `--series` takes a directory, sorts it chronologically, and threads ONE `SeriesMemory`
    through every episode, saving after each so a crash mid-series resumes rather than
    restarts.

    What a run leaves behind. Everything the paper's tables are computed from, written as it
    goes rather than at the end:

        <out>/<episode>.srt              the translation
        <out>/<episode>.jsonl            per-segment reason-act trace, written by the core
        <out>/<episode>.usage.json       calls, tokens, tool calls, latency, cost
        <out>/series_memory.json         the accumulated memory M after the last episode
        <out>/router_<episode>.json      the policy and prompts this episode actually ran
        <out>/manifest.json              one row per episode: inputs, outputs, totals

    The router snapshot is per episode and not per run because `--config` can point at a
    test-time-adapted config; recording which prompts produced which output is what makes the
    self-evolution results auditable after the fact.

Usage:
    python3 run_smart.py en2zh input.srt -o out/
    python3 run_smart.py en2zh --series Episodes/ -o out/ --series-memory out/mem.json
    python3 run_smart.py en2zh input.srt -o out/ --config configs/adapted.json
    python3 run_smart.py --list
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import directions
import instrument

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_smart")

# Episode ordering. Subtitle files are named a dozen ways, so sort on (season, episode) where
# they can be recovered and fall back to the filename. Chronological order is not cosmetic:
# the memory built from episode 1 is an input to episode 2, so shuffling changes the result.
_SXXEYY = re.compile(r"[sS](\d{1,2})[ ._-]?[eE](\d{1,3})")
_NxM = re.compile(r"(?<!\d)(\d{1,2})x(\d{1,3})(?!\d)")


def episode_key(path: Path) -> tuple:
    for rx in (_SXXEYY, _NxM):
        m = rx.search(path.name)
        if m:
            return (0, int(m.group(1)), int(m.group(2)), path.name)
    m = re.search(r"(?<!\d)(\d{1,4})(?!\d)", path.stem)
    if m:
        return (1, 0, int(m.group(1)), path.name)
    return (2, 0, 0, path.name)


def episode_id(path: Path) -> str:
    """A short, stable id for one episode, used in filenames and in the memory."""
    m = _SXXEYY.search(path.name) or _NxM.search(path.name)
    if m:
        return f"S{int(m.group(1)):02d}E{int(m.group(2)):02d}"
    return path.stem[:40]


def find_episodes(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    files = sorted((p for p in target.rglob("*.srt") if p.is_file()), key=episode_key)
    if not files:
        raise SystemExit(f"no .srt files under {target}")
    return files


def run_one(mod, usage, src: Path, out_dir: Path, *, config: Path | None,
            memory, max_segments: int | None) -> dict:
    """One episode. Returns the manifest row."""
    eid = episode_id(src)
    out_srt = out_dir / f"{eid}.srt"
    started = time.time()

    pipeline = mod.Pipeline(config_path=str(config) if config else None,
                            series_memory=memory)
    calls_before = pipeline.claude.calls

    # `translate_file` derives the trace path from the output path by replacing `.srt` with
    # `.jsonl`, so the trace lands next to the subtitle without being asked for.
    results = pipeline.translate_file(str(src), str(out_srt), max_segments=max_segments,
                                      episode_id=eid)

    # What the episode actually ran with. Written after, not before: `--config` may have been
    # absent, in which case the defaults baked into this direction's module are the answer.
    pipeline.router.save(str(out_dir / f"router_{eid}.json"))

    refined_seg = sum(1 for r in results if r.refined_segment)
    refined_doc = sum(1 for r in results if r.refined_doc)
    scored = [r.score for r in results if r.score]
    row = {
        "episode": eid,
        "source": str(src),
        "srt": str(out_srt),
        "trace": str(out_srt.with_suffix(".jsonl")),
        "router": str(out_dir / f"router_{eid}.json"),
        "segments": len(results),
        "judge_mean": round(sum(scored) / len(scored), 3) if scored else None,
        "refined_segment": refined_seg,
        "refined_doc": refined_doc,
        "agents_used": sorted({r.agent for r in results}),
        "pipeline_calls": pipeline.claude.calls - calls_before,
        "seconds": round(time.time() - started, 1),
    }
    logger.info("%s: %d segments, judge mean %s, %d/%d refined (segment/doc), %.1f min",
                eid, row["segments"], row["judge_mean"], refined_seg, refined_doc,
                row["seconds"] / 60)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run SMART on a subtitle file or a series.")
    ap.add_argument("direction", nargs="?", help="e.g. en2zh, de2en, es2en_419")
    ap.add_argument("input", nargs="?", help="a .srt file, or a directory with --series")
    ap.add_argument("-o", "--out", default="output", help="output directory")
    ap.add_argument("--series", action="store_true",
                    help="treat the input as a directory of episodes, in chronological order")
    ap.add_argument("--series-memory", default=None,
                    help="path to the persistent memory M (default: <out>/series_memory.json)")
    ap.add_argument("--no-memory", action="store_true",
                    help="run each episode with no persistent memory (ablation)")
    ap.add_argument("--config", default=None,
                    help="a router config: adapted prompts and policy from self_evolve.py")
    ap.add_argument("--max-segments", type=int, default=None,
                    help="translate only the first N segments of each episode (smoke test)")
    ap.add_argument("--list", action="store_true", help="list the thirty directions and exit")
    args = ap.parse_args(argv)

    if args.list:
        return directions.main()
    if not args.direction or not args.input:
        ap.error("direction and input are required (or use --list)")

    d = directions.resolve(args.direction)
    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"no such input: {src}")
    if args.series and not src.is_dir():
        raise SystemExit(f"--series needs a directory, got {src}")
    episodes = find_episodes(src) if args.series else [src]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = Path(args.config) if args.config else None
    if config and not config.exists():
        raise SystemExit(f"no such config: {config}")

    logger.info("%s (%s): %d episode(s) -> %s", d.name, d.label, len(episodes), out_dir)
    mod = directions.load(d)
    usage = instrument.attach(mod, d.name)

    memory = None
    if not args.no_memory:
        mem_path = Path(args.series_memory) if args.series_memory \
            else out_dir / "series_memory.json"
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        memory = mod.SeriesMemory(str(mem_path))
        logger.info("persistent memory: %s", mem_path)

    rows, failed = [], []
    for i, ep in enumerate(episodes, 1):
        logger.info("[%d/%d] %s", i, len(episodes), ep.name)
        try:
            rows.append(run_one(mod, usage, ep, out_dir, config=config, memory=memory,
                                max_segments=args.max_segments))
        except Exception as exc:                       # one bad episode must not lose the rest
            logger.exception("%s FAILED: %s", ep.name, exc)
            failed.append({"source": str(ep), "error": f"{type(exc).__name__}: {exc}"})

    usage.finish()
    usage.dump(out_dir / f"usage_{d.name}.json")
    manifest = {
        "direction": d.name,
        "label": d.label,
        "group": d.group,
        "module": d.filename,
        "constraints": directions.constraints(d),
        "config": str(config) if config else None,
        "memory": None if memory is None else memory.path,
        "max_segments": args.max_segments,
        "episodes": rows,
        "failed": failed,
        "usage": usage.summary(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("done: %d episode(s), %d failed. %s", len(rows), len(failed), usage)
    logger.info("manifest: %s", out_dir / "manifest.json")
    return 1 if failed and not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
