"""Test-time self-evolution: adapt the prompts and the routing policy on a series prefix.

SMART's state is `S_t = (R, Pi_t, rho_t, M_t)`: a fixed role set `R`, the agent prompts `Pi`,
the routing policy `rho`, and the persistent series memory `M`. Self-evolution moves the three
that are not fixed. This script is that loop, and the split between what it freezes and what
it does not is the method:

    Chronological 3:7 split, per series. The first 30% of a series by episode order is the
    adaptation prefix; the remaining 70% is held-out inference. Not random: the claim is that
    a system improves over a series as it watches it, and a random split would let a later
    episode's terminology leak backwards into the prefix it was supposed to be learned from.

    Prompts and policy freeze after adaptation. `Pi` and `rho` are updated only on the prefix,
    then written once and reused verbatim for every held-out episode.

    Memory does not freeze. `M` keeps accumulating through inference, because it is task state
    rather than a learned parameter - a character's name established in episode 8 must be
    available in episode 9 whether or not the prompts are still moving. Freezing it would be
    measuring a different system.

Each epoch does five things: translate the prefix under the current config, aggregate the
judge's own scores per agent and per routing category, rewrite the prompts of agents that
underperform, revise the policy, and write the config the next epoch reads. The optimisers are
themselves model calls - the system rewrites its own prompts - which is why every rewrite is
logged with both versions rather than only the result.

    What this rewrites versus the old loop. The prompt and structure optimisers here are
    parameterised by direction: the source and target language names, and the required closing
    line, are read from the direction's own module rather than written into the optimiser. The
    same script therefore adapts all thirty directions, and an en->tr run cannot be silently
    told to produce English.

    The per-segment join. Attributing a result to a routing category cannot go through
    `router.log`: the core records there once per *single-segment* translation, so segments
    that were merged into a continuation group produce a result with no log entry, and joining
    on (agent, score) would mis-attribute the rest. `categorise()` instead recomputes the
    category with the router's own `classify` against the scene map the pipeline built, which
    covers grouped and ungrouped segments the same way.

Usage:
    python3 self_evolve.py en2zh --series Episodes/ -o out/en2zh
    python3 self_evolve.py en2zh --series Episodes/ -o out/en2zh --epochs 3 --adapt-ratio 0.3
    python3 self_evolve.py en2zh --series Episodes/ -o out/en2zh --adapt-only
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import directions
import instrument
import run_smart

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("self_evolve")

# An agent is rewritten when its mean judge score falls below this and it was used enough
# times for the mean to mean anything. Both are thresholds on the system's own judge, which is
# the only signal available at test time - there are no references on the adaptation prefix.
REWRITE_BELOW = 8.0
REWRITE_MIN_COUNT = 2


# --------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------

def categorise(mod, router, results: list, scenes: list) -> list[dict]:
    """Recover (category, features) for every result, via the router's own classifier.

    Reconstructs the scene map the pipeline built - scene bounds are in segment-index space -
    so `classify` sees the same `scene.tone` it saw during translation. `classify` also writes
    `ctx.features` as a side effect, which is where the features come from.
    """
    scene_map = {}
    for scene in scenes or []:
        for idx in range(scene.start_idx, scene.end_idx + 1):
            scene_map[idx] = scene
    rows = []
    for r in results:
        ctx = mod.Context(segment=r.segment, preceding_src=[], succeeding_src=[],
                          preceding_tgt=[], scene=scene_map.get(r.segment.index))
        category = router.classify(ctx)
        rows.append({
            "source": r.source, "translation": r.translation, "agent": r.agent,
            "score": r.score, "tools": list(r.tools), "category": category,
            "features": dict(ctx.features),
            "refined_segment": r.refined_segment, "refined_doc": r.refined_doc,
        })
    return rows


class Evaluator:
    """Turns one epoch's results into the numbers the optimisers are given."""

    def __init__(self):
        self.rows: list[dict] = []

    def extend(self, rows: list[dict]) -> None:
        self.rows += rows

    def analyze(self) -> dict:
        if not self.rows:
            return {"status": "no_data", "total": 0}

        by_agent = defaultdict(lambda: {"scores": [], "tools": defaultdict(int)})
        for r in self.rows:
            by_agent[r["agent"]]["scores"].append(r["score"])
            for t in r["tools"]:
                by_agent[r["agent"]]["tools"][t] += 1

        agent_stats = {}
        for name, data in by_agent.items():
            s = data["scores"]
            agent_stats[name] = {
                "avg": round(sum(s) / len(s), 2), "min": round(min(s), 2),
                "max": round(max(s), 2), "count": len(s), "tools": dict(data["tools"]),
            }

        by_cat = defaultdict(lambda: defaultdict(list))
        for r in self.rows:
            by_cat[r["category"]][r["agent"]].append(r["score"])
        cat_stats = {cat: {a: round(sum(s) / len(s), 2) for a, s in agents.items()}
                     for cat, agents in by_cat.items()}
        cat_counts = {cat: sum(len(s) for s in agents.values())
                      for cat, agents in by_cat.items()}

        worst = sorted(self.rows, key=lambda x: x["score"])[:15]
        scores = [r["score"] for r in self.rows]
        return {
            "total": len(self.rows),
            "overall_avg": round(sum(scores) / len(scores), 2),
            "agent_stats": agent_stats,
            "category_stats": cat_stats,
            "category_counts": cat_counts,
            "refined_segment": sum(1 for r in self.rows if r["refined_segment"]),
            "refined_doc": sum(1 for r in self.rows if r["refined_doc"]),
            "worst_examples": worst[:10],
        }


# --------------------------------------------------------------------------------
# the two optimisers
# --------------------------------------------------------------------------------

PROMPT_OPT_SYSTEM = """You are a prompt engineer optimizing one agent of a multi-agent \
subtitle translation system.

You are given the agent's current system prompt, how it scored, the tools it can call, and \
the lowest-scoring translations it produced. Rewrite the prompt so those failures are less \
likely.

The prompt instructs a model to translate a {SRC} subtitle into {TGT}. It is used with a \
tool-use API, so it must say when to call which tool.

Requirements:
- Keep a clear PRIORITY statement at the top.
- Keep a numbered WORKFLOW that says when to call each tool.
- Address the specific weaknesses visible in the low-scoring examples.
- Keep every display constraint and placeholder that the current prompt contains. \
Placeholders in braces, such as {duration} and {max_chars}, are substituted at run time and \
must survive verbatim.
- End with exactly this line: {CLOSER}

Output ONLY the new system prompt. No preamble, no explanation, no code fences."""

STRUCT_OPT_SYSTEM = """You optimize the routing policy of a multi-agent subtitle translation \
system ({SRC} to {TGT}).

The router sorts each subtitle into exactly one category and runs the listed translator agents \
on it; a judge then picks the best candidate. More agents means better coverage and higher \
cost.

Rules you must not break:
- The categories are exactly: {CATEGORIES}. Do not invent or drop a category.
- The agents are exactly: {AGENTS}. Do not invent an agent.
- Every category must list at least 2 agents.
- Tools must come from: {TOOLS}.

Use the performance data to drop agents that score consistently low in a category and to add \
ones that score well elsewhere and are plausibly relevant.

Output valid JSON and nothing else:
{
  "policy": {"<category>": ["<agent>", "..."], "...": []},
  "tool_updates": {"<agent>": ["<tool>", "..."]},
  "rationale": "one sentence"
}"""


def closing_line(prompt: str, tgt_name: str) -> str:
    """The line the rewritten prompt must still end with.

    Taken from the prompt being rewritten rather than constructed, because the core's own
    convention differs by role - an agent ends `Output ONLY the final X translation.`, the
    refiner ends `Output ONLY the refined X translation.` - and the optimiser must not
    quietly change which one this agent uses. The constructed form is only a fallback for a
    prompt that has no such line at all.
    """
    for line in reversed(prompt.strip().splitlines()):
        if line.strip().startswith("Output ONLY"):
            return line.strip()
    return f"Output ONLY the final {tgt_name} translation."


def optimize_prompt(claude, agent_name: str, current: str, stats: dict,
                    worst: list[dict], tools: list[str], src_name: str,
                    tgt_name: str) -> str:
    examples = [e for e in worst if e["agent"] == agent_name][:5] or worst[:3]
    ex = "\n".join(f'  src="{e["source"]}" -> got="{e["translation"]}" '
                   f'(score={e["score"]}, category={e["category"]})' for e in examples)
    msg = f"""Agent: {agent_name}
Mean judge score: {stats.get('avg', '?')}/10 over {stats.get('count', 0)} segments \
(min {stats.get('min', '?')}, max {stats.get('max', '?')})
Tools it actually called: {json.dumps(stats.get('tools', {}))}
Tools available to it: {tools}

Lowest-scoring translations it produced:
{ex}

Its current system prompt:
---
{current}
---

Rewrite it. Output ONLY the new prompt."""
    system = (PROMPT_OPT_SYSTEM
              .replace("{SRC}", src_name).replace("{TGT}", tgt_name)
              .replace("{CLOSER}", closing_line(current, tgt_name)))
    return claude.chat(system, msg, temperature=0.4).strip()


def optimize_structure(claude, policy: dict, agents: dict, cat_stats: dict,
                       agent_stats: dict, cat_counts: dict, tool_names: list[str],
                       src_name: str, tgt_name: str) -> dict:
    tool_info = {name: cfg.get("tools", []) for name, cfg in agents.items()}
    msg = f"""Current policy:
{json.dumps(policy, indent=2)}

Current tool assignments:
{json.dumps(tool_info, indent=2)}

Mean judge score by category and agent:
{json.dumps(cat_stats, indent=2)}

Segments seen per category:
{json.dumps(cat_counts, indent=2)}

Agent totals:
{json.dumps(agent_stats, indent=2)}

Output the optimized JSON."""
    system = (STRUCT_OPT_SYSTEM
              .replace("{SRC}", src_name).replace("{TGT}", tgt_name)
              .replace("{CATEGORIES}", ", ".join(sorted(policy)))
              .replace("{AGENTS}", ", ".join(sorted(agents)))
              .replace("{TOOLS}", ", ".join(tool_names)))
    resp = claude.chat(system, msg, temperature=0.3)
    m = re.search(r"\{.*\}", resp, re.DOTALL)
    if not m:
        logger.warning("structure optimiser returned no JSON object; keeping the policy")
        return {}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError as exc:
        logger.warning("structure optimiser returned invalid JSON (%s); keeping the policy",
                       exc)
        return {}


def validate_policy(proposed: dict, current: dict, agents: dict) -> tuple[dict, list[str]]:
    """Accept a proposed policy only where it is well formed; report every rejection.

    The optimiser is a model call, so it can hallucinate a category the router never emits or
    an agent that does not exist, and a policy containing either would make `get_agents` fall
    through to the default for those segments - a silent ablation. Rejections are per key, so
    one bad entry does not discard a whole good policy.
    """
    if not isinstance(proposed, dict) or not proposed:
        return dict(current), ["no policy proposed"]
    out, notes = dict(current), []
    for cat, names in proposed.items():
        if cat not in current:
            notes.append(f"dropped unknown category {cat!r}")
            continue
        if not isinstance(names, list):
            notes.append(f"{cat}: not a list")
            continue
        kept = [n for n in names if n in agents]
        if len(kept) != len(names):
            notes.append(f"{cat}: dropped unknown agent(s) "
                         f"{sorted(set(names) - set(kept))}")
        if len(kept) < 2:
            notes.append(f"{cat}: {len(kept)} agent(s) after validation, kept the old list")
            continue
        out[cat] = kept
    missing = [c for c in current if c not in proposed]
    if missing:
        notes.append(f"kept unchanged (not proposed): {', '.join(sorted(missing))}")
    return out, notes


def prompt_delta(before: str, after: str) -> dict:
    """A compact, auditable description of one prompt rewrite."""
    b, a = before.splitlines(), after.splitlines()
    added = sum(1 for ln in difflib.ndiff(b, a) if ln.startswith("+ "))
    removed = sum(1 for ln in difflib.ndiff(b, a) if ln.startswith("- "))
    # Placeholders are the one thing a rewrite must not lose. The core substitutes them with
    # `str.replace` in `Router.get_agents`, so losing one does not raise - it leaves a
    # `length_aware` agent silently unaware of the duration it is supposed to fit, and gaining
    # a new one leaves a literal `{...}` in the prompt the model is shown. Both are quiet, so
    # the set is compared exactly rather than checked for well-formedness.
    ph = lambda s: sorted(set(re.findall(r"\{[a-z_]+\}", s)))
    return {
        "chars_before": len(before), "chars_after": len(after),
        "lines_added": added, "lines_removed": removed,
        "placeholders_before": ph(before), "placeholders_after": ph(after),
        "placeholders_preserved": ph(before) == ph(after),
        "closer_preserved": after.strip().endswith(closing_line(before, "")),
    }


def accept_prompt(before: str, after: str) -> tuple[bool, str]:
    """Whether a rewrite is safe to install. A rejected rewrite leaves the prompt alone."""
    if not after or len(after) < 200:
        return False, "too short to be a system prompt"
    d = prompt_delta(before, after)
    if not d["placeholders_preserved"]:
        return False, (f"placeholders changed {d['placeholders_before']} -> "
                       f"{d['placeholders_after']}")
    if not d["closer_preserved"]:
        return False, f"does not end with {closing_line(before, '')!r}"
    return True, "accepted"


# --------------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------------

def split_series(episodes: list[Path], ratio: float) -> tuple[list[Path], list[Path]]:
    """Chronological prefix/suffix split. At least one episode adapts, at least one is held out."""
    n = len(episodes)
    if n < 2:
        return episodes, []
    k = max(1, min(n - 1, round(n * ratio)))
    return episodes[:k], episodes[k:]


def epoch_slice(adapt: list[Path], epoch: int, epochs: int,
                override: int | None) -> list[Path]:
    """Which adaptation episodes epoch `epoch` translates.

    Where the prefix has at least one episode per epoch, each epoch takes a different,
    contiguous chunk of it, so three epochs do not fit the optimisers to the same episode three
    times. Where it does not - a short series, or `--segments-per-episode` smoke tests - every
    epoch re-translates the whole prefix, which is the honest fallback: the prompts still move,
    they just move on the same text under the memory accumulated so far.
    """
    if not adapt:
        return []
    size = override or (len(adapt) // epochs if len(adapt) >= epochs else len(adapt))
    size = max(1, min(size, len(adapt)))
    if size >= len(adapt):
        return list(adapt)
    start = (epoch * size) % len(adapt)
    chunk = adapt[start:start + size]
    # Wrap rather than return a short tail, so a late epoch still sees `size` episodes.
    if len(chunk) < size:
        chunk += adapt[:size - len(chunk)]
    return chunk


def evolve(direction: str, series: Path, out_dir: Path, *, epochs: int, adapt_ratio: float,
           segments_per_episode: int | None, adapt_only: bool,
           episodes_per_epoch: int | None) -> dict:
    d = directions.resolve(direction)
    src_name = directions.LOCALES[d.src].qualified
    tgt_name = directions.LOCALES[d.tgt].qualified
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dir = out_dir / "configs"
    config_dir.mkdir(exist_ok=True)

    episodes = run_smart.find_episodes(series)
    adapt, held_out = split_series(episodes, adapt_ratio)
    logger.info("%s (%s -> %s): %d episodes, %d adapt / %d held out",
                d.name, src_name, tgt_name, len(episodes), len(adapt), len(held_out))
    if not held_out and not adapt_only:
        logger.warning("only %d episode(s): nothing is held out, so the inference phase "
                       "would re-score the adaptation prefix. Use --adapt-only.", len(episodes))

    mod = directions.load(d)
    usage = instrument.attach(mod, d.name)
    claude = mod.Claude()
    tool_names = [s["name"] for s in mod.TOOL_SCHEMAS]

    # One memory for the whole series, adaptation and inference alike. This is the `M` that is
    # deliberately not frozen.
    memory = mod.SeriesMemory(str(out_dir / "series_memory.json"))

    log = {
        "direction": d.name, "label": d.label, "module": d.filename,
        "source_language": src_name, "target_language": tgt_name,
        "model_id": getattr(mod, "MODEL_ID", ""),
        "series": str(series),
        "split": {"ratio": adapt_ratio,
                  "adapt": [p.name for p in adapt],
                  "held_out": [p.name for p in held_out]},
        "settings": {"epochs": epochs, "segments_per_episode": segments_per_episode,
                     "episodes_per_epoch": episodes_per_epoch,
                     "rewrite_below": REWRITE_BELOW,
                     "rewrite_min_count": REWRITE_MIN_COUNT},
        "epochs": [],
        "inference": None,
    }

    prev_config: Path | None = None
    for epoch in range(epochs):
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("EPOCH %d/%d  config=%s", epoch, epochs - 1,
                    prev_config.name if prev_config else "module defaults")
        epoch_dir = out_dir / f"epoch{epoch}"
        epoch_dir.mkdir(exist_ok=True)
        config_in = prev_config          # captured before it is overwritten below

        pipeline = mod.Pipeline(config_path=str(prev_config) if prev_config else None,
                               series_memory=memory)
        policy_before = json.loads(json.dumps(pipeline.router.policy))
        prompts_before = {k: v.get("prompt", "") for k, v in pipeline.router.agents.items()}

        slice_ = epoch_slice(adapt, epoch, epochs, episodes_per_epoch)
        logger.info("  prefix slice: %s", ", ".join(p.name for p in slice_))

        evaluator, episode_rows = Evaluator(), []
        for ep in slice_:
            eid = run_smart.episode_id(ep)
            logger.info("  adapt: %s", ep.name)
            try:
                results = pipeline.translate_file(
                    str(ep), str(epoch_dir / f"{eid}.srt"),
                    max_segments=segments_per_episode, episode_id=eid)
            except Exception as exc:
                logger.exception("  %s FAILED: %s", ep.name, exc)
                episode_rows.append({"episode": eid, "error": f"{type(exc).__name__}: {exc}"})
                continue
            rows = categorise(mod, pipeline.router, results, pipeline.scenes)
            evaluator.extend(rows)
            episode_rows.append({"episode": eid, "segments": len(rows),
                                 "judge_mean": round(sum(r["score"] for r in rows)
                                                     / len(rows), 3) if rows else None})

        analysis = evaluator.analyze()
        logger.info("  epoch %d: %d segments, judge mean %s", epoch, analysis.get("total", 0),
                    analysis.get("overall_avg"))
        (epoch_dir / "analysis.json").write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

        # ---- rewrite the prompts of the agents the judge is unhappy with ----
        new_agents = {k: dict(v) for k, v in pipeline.router.agents.items()}
        rewrites = []
        for name, stats in sorted(analysis.get("agent_stats", {}).items()):
            if stats["avg"] >= REWRITE_BELOW or stats["count"] < REWRITE_MIN_COUNT:
                continue
            current = new_agents.get(name, {}).get("prompt", "")
            if not current:
                continue
            logger.info("  optimising %r (mean %.2f over %d)", name, stats["avg"],
                        stats["count"])
            proposed = optimize_prompt(
                claude, name, current, stats, analysis.get("worst_examples", []),
                new_agents[name].get("tools", tool_names), src_name, tgt_name)
            ok, why = accept_prompt(current, proposed)
            record = {"agent": name, "trigger": {"avg": stats["avg"], "count": stats["count"]},
                      "accepted": ok, "reason": why, **prompt_delta(current, proposed),
                      "before": current, "after": proposed}
            if ok:
                new_agents[name]["prompt"] = proposed
                logger.info("    accepted (%d -> %d chars)", len(current), len(proposed))
            else:
                logger.warning("    rejected: %s", why)
            rewrites.append(record)

        # ---- revise the routing policy ----
        struct = optimize_structure(
            claude, pipeline.router.policy, pipeline.router.agents,
            analysis.get("category_stats", {}), analysis.get("agent_stats", {}),
            analysis.get("category_counts", {}), tool_names, src_name, tgt_name)
        new_policy, notes = validate_policy(struct.get("policy", {}),
                                            pipeline.router.policy, new_agents)
        for name, tools in (struct.get("tool_updates") or {}).items():
            if name not in new_agents or not isinstance(tools, list):
                notes.append(f"tool_updates: ignored {name!r}")
                continue
            kept = [t for t in tools if t in tool_names]
            if kept != tools:
                notes.append(f"tool_updates[{name}]: dropped unknown "
                             f"{sorted(set(tools) - set(kept))}")
            if kept:
                new_agents[name]["tools"] = kept
        for n in notes:
            logger.info("  policy: %s", n)

        # ---- write the config the next epoch reads ----
        config = {"epoch": epoch, "direction": d.name,
                  "policy": new_policy, "agents": new_agents}
        config_path = config_dir / f"epoch{epoch}.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        prev_config = config_path

        log["epochs"].append({
            "epoch": epoch,
            "config_in": str(config_in) if config_in else None,
            "config_out": str(config_path),
            "episodes": episode_rows,
            "segments": analysis.get("total", 0),
            "judge_mean": analysis.get("overall_avg"),
            # The per-agent judge means, which is what "the system improved" is measured on.
            "agent_judge_mean": {k: v["avg"]
                                 for k, v in analysis.get("agent_stats", {}).items()},
            "agent_counts": {k: v["count"]
                             for k, v in analysis.get("agent_stats", {}).items()},
            "category_judge_mean": analysis.get("category_stats", {}),
            "refined_segment": analysis.get("refined_segment"),
            "refined_doc": analysis.get("refined_doc"),
            "prompts_optimized": rewrites,
            "prompts_accepted": [r["agent"] for r in rewrites if r["accepted"]],
            "policy_before": policy_before,
            "policy_after": new_policy,
            "policy_changed": new_policy != policy_before,
            "policy_notes": notes,
            "structure_rationale": struct.get("rationale", ""),
            "tools_after": {k: v.get("tools", []) for k, v in new_agents.items()},
            "prompt_chars_before": {k: len(v) for k, v in prompts_before.items()},
            "prompt_chars_after": {k: len(v.get("prompt", ""))
                                   for k, v in new_agents.items()},
            "seconds": round(time.time() - t0, 1),
            "usage_so_far": usage.summary(),
        })
        _write_log(out_dir, log)          # after every epoch, so a crash keeps what ran

    # ---- held-out inference with the adapted config frozen ----
    if held_out and not adapt_only:
        logger.info("=" * 60)
        logger.info("INFERENCE: %d held-out episode(s), config frozen at %s",
                    len(held_out), prev_config.name if prev_config else "defaults")
        infer_dir = out_dir / "inference"
        infer_dir.mkdir(exist_ok=True)
        rows, failed = [], []
        for ep in held_out:
            logger.info("  infer: %s", ep.name)
            try:
                rows.append(run_smart.run_one(mod, usage, ep, infer_dir, config=prev_config,
                                              memory=memory,
                                              max_segments=segments_per_episode))
            except Exception as exc:
                logger.exception("  %s FAILED: %s", ep.name, exc)
                failed.append({"source": str(ep), "error": f"{type(exc).__name__}: {exc}"})
        means = [r["judge_mean"] for r in rows if r["judge_mean"] is not None]
        log["inference"] = {
            "config": str(prev_config) if prev_config else None,
            "episodes": rows, "failed": failed,
            "judge_mean": round(sum(means) / len(means), 3) if means else None,
        }

    usage.finish()
    log["usage"] = usage.summary()
    log["memory"] = {
        "path": memory.path,
        "terminology": len(memory.data.get("terminology", {})),
        "characters": len(memory.data.get("characters", [])),
        "domain_knowledge": len(memory.data.get("domain_knowledge", [])),
        "episodes_processed": memory.data.get("episodes_processed", []),
    }
    _write_log(out_dir, log)
    logger.info("evolution log: %s", out_dir / "evolution_log.json")
    logger.info("final config:  %s", prev_config)
    logger.info("%s", usage)
    return log


def _write_log(out_dir: Path, log: dict) -> None:
    (out_dir / "evolution_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Test-time self-evolution of prompts and routing policy.")
    ap.add_argument("direction", help="e.g. en2zh, de2en, es2en_419")
    ap.add_argument("--series", required=True,
                    help="directory of one series' episodes, or a single .srt")
    ap.add_argument("-o", "--out", default="output/evolve", help="output directory")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--adapt-ratio", type=float, default=0.3,
                    help="fraction of the series used for adaptation (default 0.3)")
    ap.add_argument("--episodes-per-epoch", type=int, default=None,
                    help="override the per-epoch slice of the adaptation prefix")
    ap.add_argument("--segments-per-episode", type=int, default=None,
                    help="translate only the first N segments of each episode")
    ap.add_argument("--adapt-only", action="store_true",
                    help="stop after adaptation; do not run held-out inference")
    args = ap.parse_args(argv)

    if not 0 < args.adapt_ratio < 1:
        ap.error("--adapt-ratio must be strictly between 0 and 1")
    evolve(args.direction, Path(args.series), Path(args.out), epochs=args.epochs,
           adapt_ratio=args.adapt_ratio, segments_per_episode=args.segments_per_episode,
           adapt_only=args.adapt_only, episodes_per_epoch=args.episodes_per_epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
