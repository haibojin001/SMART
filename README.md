# SMART — Self-evolving Multi-Agent subtitle tRanslaTion

Code for the paper. This repository holds the system, the self-evolution loop, the instrumentation
and the SubMQM evaluator. **It does not hold the benchmark.** The evaluation data is licensed
subtitle material from commercial series and cannot be redistributed, so what is published here is
everything needed to re-run the method on your own data, and nothing that would reproduce a number
in the paper without it.

```
harness/            entry points — nothing in mas_core/ has a __main__
  run_smart.py        translate one episode, or a series in order
  self_evolve.py      test-time self-evolution over a series prefix
  instrument.py       cost and latency accounting, added from outside the cores
  directions.py       the 30 directions, and how to load a core for one
  submqm/             the evaluator: rubric, alignment, judge, sweep, LaTeX tables
  tools/
    check_release.py  the anonymity and compile gate for this tree
  examples/           one synthetic 12-cue SRT, for checking an install
mas_core/           30 self-contained cores, mas_core_<src>2<tgt>_v4.py
web_search/         the research tool's three interchangeable backends
```

## Install

```
pip install -r requirements.txt          # everything, including the browser backends
pip install -r requirements-min.txt      # enough to translate, evolve and score
```

Python 3.9 or newer.

Models are reached through the Anthropic Messages API. One key, read from the environment by the
SDK itself — nothing in this repository reads a credential from a file, and no credential is ever
passed as a function argument, so none can be captured by a traceback that renders its arguments:

```
export ANTHROPIC_API_KEY=...       # the only credential the method needs
export ANTHROPIC_MODEL=...         # optional; default: claude-sonnet-4-6
```

If your site can only reach Claude through a cloud provider, set `ANTHROPIC_PROVIDER` and let that
provider's own SDK resolve its own credentials in its own documented way:

```
export ANTHROPIC_PROVIDER=bedrock  # or vertex; default: anthropic
export ANTHROPIC_MODEL=...         # required with a provider: see below
```

This changes the transport and nothing else *in the method*: the request and response bodies are
the Messages API in all three cases, the same client class serves all three, and no result in the
paper depends on which one was used.

It does change how a model is named, which is the one thing you have to set by hand. Each provider
publishes Claude under its own identifier scheme, and the default here is the direct API's, so a
provider selected with the model left alone fails on the first call with an invalid-identifier
error that says nothing about the real cause. Set `ANTHROPIC_MODEL` to whatever id your provider
lists for the model you want; `sonnet-4-6` is what the paper reports.

Smoke-test the install on the bundled example, which is synthetic and not from the benchmark:

```
python3 harness/directions.py            # list the 30 directions; needs no credentials
python3 harness/directions.py --check    # every core present and parsing; still no credentials
python3 harness/run_smart.py en2it harness/examples/example_en.srt -o /tmp/smoke
```

`--check` imports nothing — it reads each core with `ast`, so it works before the SDK is
installed and tells a missing file apart from a broken one. It is a parse check, not an import
check: a core can parse and still fail to import, which is why `requirements.txt` states the
Python floor rather than leaving it to be discovered.

## Why thirty files instead of one

`mas_core/` contains thirty near-duplicate modules rather than one core parameterised by locale.
That is deliberate. Each file carries its own agent prompts, judge criteria, cue lists, quotation
and punctuation rules and display limits, written in the target language's own terms — Korean line
breaking, French spacing before `!`, Chinese cue punctuation — and a shared core would have to
express all of it as configuration, which is how locale-specific behaviour quietly regresses. The
cost is that a change to shared machinery must be applied thirty times; `harness/directions.py`
exists so that everything outside the cores can still treat them uniformly.

The directions are English into and out of each of fifteen locales: `zh-CN`, `ko-KR`, and thirteen
European ones — `de-DE`, `fr-FR`, `it-IT`, `es-ES`, `es-419`, `pt-PT`, `pt-BR`, `nl-NL`, `sv-SE`,
`da-DK`, `no-NO`, `ro-RO`, `tr-TR`. Both Spanish and both Portuguese variants are separate
directions because they differ in register and terminology, not only in spelling; for that reason
the ambiguous short forms `en2es` and `en2pt` are rejected rather than guessed, and you must write
`en2es_ES` or `en2es_419`.

## Running the method

Series mode is the default shape rather than a convenience. SMART's state is
`S_t = (R, Π_t, ρ_t, M_t)`: a fixed role set `R`, the agent prompts `Π`, the routing policy `ρ` and
the persistent series memory `M`. Self-evolution moves the three that are not fixed, and `M`
accumulates across episodes in broadcast order, so translating episode 5 with episode 4's memory
discarded does not measure the same system.

`direction` and `input` are positionals and `-o` is an output *directory*, not a file.

```
# one episode
python3 harness/run_smart.py en2ko episode.srt -o out/

# a series, in order, carrying memory forward into out/series_memory.json
python3 harness/run_smart.py en2ko path/to/series/ --series -o out/

# the no-memory ablation: same episodes, M discarded between them
python3 harness/run_smart.py en2ko path/to/series/ --series --no-memory -o out_nomem/

# test-time self-evolution: adapt on the first 30% chronologically, hold out the rest
python3 harness/self_evolve.py en2ko --series path/to/series/ --adapt-ratio 0.3 -o state/

# then run the held-out 70% under the adapted prompts and policy
python3 harness/run_smart.py en2ko heldout/ --series --config state/configs/epoch2.json -o out/
```

`--config` is that handoff and nothing else: it takes one of the `configs/epochN.json` files
`self_evolve.py` writes, carrying the adapted prompts `Π` and routing policy `ρ`. It is not the
direction and not a general settings file.

The 3:7 chronological split is per series: the first 30% of episodes is what the prompts and the
routing policy may adapt on, and the remaining 70% is held out. The split is by broadcast order,
not at random, because a random split lets terminology established in a later episode leak into an
earlier one.

## Scoring

`harness/submqm/` is the evaluator, and it is deliberately **not** SMART's internal judge. The
cores contain a 1–10 judge–refiner; that is a component of the system under measurement. SubMQM is
the instrument: seven dimensions, nineteen error types, penalties in {0, 5, 10}, lower is better.
Different prompts, different scale, and by default a different model.

```
python3 -m submqm.evaluate --job jobs/main.json -o results/main.json
python3 -m submqm.tables results/main.json -o tables/
```

Every module runs standalone with `--self-test`, and `evaluate.py --self-test` drives the whole
scoring chain against a stub judge, so the alignment and the arithmetic can be checked without
spending a model call:

```
for m in rubric align judge evaluate tables; do python3 -m submqm.$m --self-test; done
```

Two things about the scoring are easy to get wrong and are therefore enforced rather than
documented. Hypotheses that preserve the source segmentation are aligned in `block` mode, 1:1;
independently segmented references are aligned in `passage` mode by timestamp overlap. And
references are normalised while hypotheses are not — normalising a hypothesis would silently
repair the exact spacing and punctuation errors the rubric is there to count.
