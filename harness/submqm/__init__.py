"""SubMQM: the subtitle-translation evaluator used to score every system in the paper.

Five modules, in the order a run uses them:

    rubric.py     the 7 dimensions, 19 error types, {0,5,10} penalties and the Overall weights
    align.py      pair a hypothesis against a reference and cut the pair into judging windows
    judge.py      ask one model to score one window, and turn its JSON into penalties
    evaluate.py   sweep systems x directions x episodes and aggregate
    tables.py     render the aggregated results as the paper's LaTeX tables

Each module runs standalone with `--self-test`, and `evaluate.py`'s self-test exercises the
whole chain on a stub judge, so the arithmetic can be checked without spending a model call:

    python3 -m submqm.rubric --self-test
    python3 -m submqm.evaluate --self-test

The evaluator is deliberately separate from SMART's own internal 1-10 judge-refiner. That judge
is a component of the system under measurement; this one is the measuring instrument. They use
different prompts, different scales and - by default - different models, and nothing in this
package imports the cores except `judge.budget_for`, which reads a direction's `MAX_LINE` and
`MAX_CPS` so the technical checks score against the limits that direction was actually run
under rather than a constant repeated here.
"""

from __future__ import annotations

__all__ = ["rubric", "align", "judge", "evaluate", "tables"]
