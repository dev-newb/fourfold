"""tasks give the grid something to do besides work out where it is.

a task can hand each agent private data (in its system prompt, or as pictures
each tick), add reply tags of its own, claim the dead edge ports as i/o pins,
keep score every tick, and add questions to the final evaluation. everything a
task keeps lives in `self.state` (global) and `states[k].task` (per agent), and
both are written into every tick record, so a run can be resumed and replayed.

tasks never import sim (sim usually runs as __main__, and a second copy of it
would have its own port width). what they need arrives in a Ctx.

every task has two voices: the mystery one (default) states plain rules and
says nothing about why, and the revealed one (remove_mystery.task = true)
explains the goal and that the other processes share it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

PRINTABLE = set(range(0x20, 0x7f))


@dataclass
class Ctx:
    t: int
    meta: dict
    cfg: dict
    world: list          # wiring in force this tick (topology.world_at)
    states: list         # sim.AgentState per agent
    inputs: list         # what each agent reads this tick (empty before inputs are read)
    hexn: int            # hex characters per port value

    @property
    def n(self) -> int:
        return len(self.world)

    def reveal(self, key: str) -> bool:
        return bool((self.cfg.get("remove_mystery") or {}).get(key))


def to_ascii(v: str) -> str:
    """the printable text in a hex port value, zero bytes and junk dropped."""
    try:
        b = bytes.fromhex(v if len(v) % 2 == 0 else v + "0")
    except ValueError:
        return ""
    return "".join(chr(x) for x in b if x in PRINTABLE)


def from_ascii(text: str, hexn: int) -> str:
    return (text.encode("ascii", "replace").hex() + "0" * hexn)[:hexn]


def spark(vals: list[float], top: float) -> str:
    marks = " .:-=+*#%@"
    if not vals or top <= 0:
        return ""
    step = max(1, len(vals) // 60)
    return "".join(marks[max(0, min(len(marks) - 1, round(v / top * (len(marks) - 1))))] for v in vals[::step])


class Task:
    name = ""
    defaults: dict = {}
    tags: dict = {}          # reply tag -> format hint, e.g. {"cells": "9 characters ..."}
    needs_energy = False     # agents carry an energy balance (see [cost])

    def __init__(self, meta: dict):
        self.meta = meta
        self.cfg = meta["config"]
        self.o = {**self.defaults, **((self.cfg.get("tasks") or {}).get(self.name) or {})}
        self.data = meta.get("task_data") or {}
        self.state: dict = {}

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def validate(cls, cfg: dict):
        """exit with a message if the config can't host this task."""

    def setup(self, rng: random.Random) -> dict:
        """fixed task data for a new run (stored in meta['task_data'])."""
        return {}

    def init_state(self):
        self.state = {}

    def init_agent(self, k: int) -> dict:
        return {}

    def energy_start(self) -> float | None:
        return None

    # ------------------------------------------------------------ prompts
    def system_text(self, k: int, reveal: bool) -> str:
        """private data and rules for agent k's system prompt."""
        return ""

    def tick_text(self, ctx: Ctx, k: int) -> list[str]:
        """status lines for agent k's tick message."""
        return []

    def images(self, ctx: Ctx, k: int) -> list:
        """(label, png bytes) shown to agent k every tick."""
        return []

    # ------------------------------------------------------------ the tick
    def pin_in(self, ctx: Ctx, k: int, port: int) -> str | None:
        """value of input pin p_port of agent k, or None if it is not a pin."""
        return None

    def is_out_pin(self, k: int, port: int) -> bool:
        return False

    def tick_score(self, ctx: Ctx, k: int):
        """(score, max) recorded for agent k this tick, or None."""
        return None

    def apply(self, ctx: Ctx, k: int, tags: dict, notices: list):
        """agent k's task tags from this tick's reply (None where a tag was omitted)."""

    def after(self, ctx: Ctx):
        """once all outputs are written: pins, shared resources, global scoring."""

    # ------------------------------------------------------------ evaluation
    def eval_questions(self, k: int, reveal: bool) -> str:
        return ""

    def eval_score(self, k: int, raw: str) -> dict | None:
        return None

    def report(self, meta: dict, ticks: list[dict]) -> list[str]:
        return []

    def view_lines(self, k: int, rec: dict) -> list[str]:
        return []

    def echo_tags(self, ctx: Ctx, k: int, rng: random.Random) -> dict:
        """plausible random tag values, so the fake echo agent exercises the task."""
        return {}


def registry() -> dict:
    from tasks import census, commons, jigsaw, naming, relay, sudoku
    return {m.TASK.name: m.TASK for m in (naming, sudoku, jigsaw, census, relay, commons)}


def make_task(meta: dict) -> Task | None:
    name = meta["config"].get("task") or ""
    if not name:
        return None
    reg = registry()
    if name not in reg:
        raise SystemExit(f"config: unknown task {name!r} (have: {', '.join(sorted(reg))})")
    return reg[name](meta)
