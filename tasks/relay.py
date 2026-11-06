"""relay: the dead edges become i/o pins. jobs come in on one side, answers are
collected on the other.

the grid stops being a closed world. on the west edge, the dead input ports
of the west column become input pins: each shows a job, as ascii (like
"37+58="), that changes every `period` ticks. on the east edge, the dead
output ports of the east column become output pins: whatever is written there
is read by the outside. a job is done when its answer ("95") shows up on any
output pin within `deadline` ticks of the job appearing. nobody in the west
can see the east pins, so the work has to travel across the grid.

in mystery mode the jobs are just values arriving from nowhere and the pins
just more ports; with `feedback` on, a bare counter of finished jobs is shown
to everyone each tick, unexplained. working out that the counter is a score,
and what moves it, is the experiment.

[tasks.relay]
job = "sum"           # sum (a+b=) | reverse (a word, written backwards) | upper (a word, in capitals)
period = 6            # ticks each job stays on its pin
deadline = 12         # ticks a job has to reach an output pin
feedback = true       # show the jobs-done counter to everyone
in_side = "W"
out_side = "E"
"""
from __future__ import annotations

import random
import re

import topology
from tasks import Ctx, Task, from_ascii, spark, to_ascii

SIDE = {"N": "north", "E": "east", "S": "south", "W": "west"}
WORDS5 = ["amber", "bloom", "crane", "drift", "ember", "frost", "glyph", "haven", "ivory", "joust",
          "knoll", "lumen", "maple", "noble", "ocean", "prism", "quilt", "raven", "stone", "tidal"]


class Relay(Task):
    name = "relay"
    defaults = {"job": "sum", "period": 6, "deadline": 12, "feedback": True, "in_side": "W", "out_side": "E"}

    @classmethod
    def validate(cls, cfg):
        if topology.world_opts(cfg)["topology"] == "torus":
            raise SystemExit("config: task relay needs dead edges for its pins; use world.topology = \"grid\"")
        if cfg.get("port_mode", "hex") != "hex" or int(cfg.get("port_bits", 128)) < 64:
            raise SystemExit("config: task relay writes ascii jobs; it needs port_mode = \"hex\" and port_bits >= 64")

    # which (agent, port) are pins, in the wiring of the moment
    def in_pin(self, ctx: Ctx, k: int, port: int) -> bool:
        a = ctx.world[k]
        return (a["in_links"][port] is None and a["in_sides"][port] == self.o["in_side"]
                and topology.boundary(a, "p", port, ctx.meta["rows"], ctx.meta["cols"], ctx.cfg))

    def out_pins(self, ctx: Ctx) -> list[tuple[int, int]]:
        pins = []
        for a in ctx.world:
            for q in range(4):
                if (a["out_links"][q] is None and a["out_sides"][q] == self.o["out_side"]
                        and topology.boundary(a, "q", q, ctx.meta["rows"], ctx.meta["cols"], ctx.cfg)):
                    pins.append((a["id"], q))
        return pins

    def job(self, row: int, j: int) -> tuple[str, str]:
        rng = random.Random(f"{self.meta['world_seed']}:job:{row}:{j}")
        kind = self.o["job"]
        if kind == "reverse":
            w = rng.choice(WORDS5)
            return w, w[::-1]
        if kind == "upper":
            w = rng.choice(WORDS5)
            return w, w.upper()
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        return f"{a}+{b}=", str(a + b)

    def init_state(self):
        self.state = {"done": 0, "released": 0, "open": {}, "finished": [], "latency": []}

    def pin_in(self, ctx: Ctx, k: int, port: int):
        if not self.in_pin(ctx, k, port):
            return None
        row = ctx.world[k]["pos"][0]
        j = (ctx.t - 1) // int(self.o["period"])
        text, ans = self.job(row, j)
        key = f"{row}:{j}"
        if key not in self.state["open"] and key not in self.state["finished"]:
            self.state["open"][key] = {"t": ctx.t, "answer": ans, "text": text}
            self.state["released"] += 1
        return from_ascii(text, ctx.hexn)

    def tick_text(self, ctx: Ctx, k: int):
        if ctx.reveal("task") or ctx.reveal("score"):
            return [f"jobs done so far: {self.state['done']} of {self.state['released']}"]
        if self.o["feedback"]:
            return [f"counter: {self.state['done']}"]
        return []

    def system_text(self, k, reveal):
        if not reveal:
            return ""
        return (f"the task: the grid is a pipeline. on the {SIDE[self.o["in_side"]]} edge of the grid, some input ports "
                f"are not dead but show jobs from the outside, as ascii text, each for "
                f"{self.o["period"]} ticks: {describe(self.o["job"])}. on the {SIDE[self.o["out_side"]]} edge, output "
                f"ports that face off the grid are read by the outside. a job is done when its answer, as ascii "
                f"text, appears on any of those output ports within {self.o['deadline']} ticks of the job "
                f"appearing. the number of jobs done is shown every tick. get as many done as you can.")

    def after(self, ctx: Ctx):
        st = self.state
        texts = [to_ascii(ctx.states[k].outputs[q]).strip() for k, q in self.out_pins(ctx)]
        tokens = {tok for t in texts for tok in re.split(r"[^0-9A-Za-z]+", t) if tok}
        for key, job in list(st["open"].items()):
            if ctx.t - job["t"] > int(self.o["deadline"]):
                del st["open"][key]
                st["finished"].append(key)
            elif job["answer"] in tokens:
                del st["open"][key]
                st["finished"].append(key)
                st["done"] += 1
                st["latency"].append(ctx.t - job["t"])
        st["pins_out"] = texts

    def view_lines(self, k, rec):
        return []

    def report(self, meta, ticks):
        st = ticks[-1].get("task_state") or {}
        lat = st.get("latency") or []
        hist = [(r.get("task_state") or {}).get("done", 0) for r in ticks]
        return [f"relay ({self.o['job']}): {st.get('done', 0)} of {st.get('released', 0)} jobs done"
                + (f", mean latency {sum(lat) / len(lat):.1f} ticks" if lat else ""),
                f"  jobs done over time: {spark(hist, max(hist) or 1)}",
                f"  last values on the output pins: {st.get('pins_out')}"]

    def echo_tags(self, ctx, k, rng):
        return {}


def describe(kind: str) -> str:
    return {"sum": "a sum like 37+58=, whose answer is 95",
            "reverse": "a word, whose answer is the same word written backwards",
            "upper": "a word, whose answer is the same word in capital letters"}.get(kind, kind)


TASK = Relay
