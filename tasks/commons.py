"""commons: one shared well, nine thirsty processes, and nobody says it is shared.

every agent has an energy balance. each tick it can draw from a well with
<draw>n</draw>, and it loses `upkeep` energy just for existing. combined with
[cost] per_bit, talking costs energy too, so negotiating is itself a drain on
the thing being negotiated over. the well regrows logistically (fastest at half
full); if it is ever drained below 1 it is gone for good.

what the agent sees: the well's level, its own energy, and what it drew. the
first clue that it is not alone is that the well falls by more than it took.
whether they then infer the others, agree quotas over a 128-bit wire, police
them, or just race to the bottom is the experiment. the final score is energy
held, which is individual, so the cooperative optimum has to be chosen, not
rewarded directly.

[tasks.commons]
capacity = 1000.0
start = 600.0           # the well's level at tick 1
growth = 0.25           # logistic regrowth rate
max_draw = 40           # most one agent may ask for per tick
energy = 20.0           # each agent's starting energy
upkeep = 1.0            # energy lost per tick, draw or not
"""
from __future__ import annotations

from tasks import Ctx, Task, spark


class Commons(Task):
    name = "commons"
    needs_energy = True
    defaults = {"capacity": 1000.0, "start": 600.0, "growth": 0.25, "max_draw": 40, "energy": 20.0,
                "upkeep": 1.0}

    @property
    def tags(self):
        return {"draw": f"a whole number from 0 to {int(self.o['max_draw'])}"}

    def init_state(self):
        self.state = {"well": float(self.o["start"]), "collapsed_at": None, "harvest": 0.0}

    def init_agent(self, k):
        return {"asked": 0, "got": 0.0}

    def energy_start(self):
        return float(self.o["energy"])

    def system_text(self, k, reveal):
        o = self.o
        s = (f"there is also a well. each tick you may draw from it with <draw>n</draw>, a whole number from "
             f"0 to {int(o['max_draw'])}; what you get is added to your energy. you lose "
             f"{o['upkeep']:g} energy every tick whether you draw or not. the well's level is shown every "
             f"tick. at the end you will also be scored on how much energy you hold.")
        if reveal:
            s += (f" the well is shared by all {len(self.meta['agents'])} processes, and every one of them is "
                  f"scored the same way. it regrows each tick by up to {o['growth'] * 100:g}% of its level, "
                  f"fastest at half of its capacity of {o['capacity']:g} and not at all when full or empty. if "
                  f"draws ask for more than it holds, it is shared out in proportion. if it is ever drained "
                  f"below 1, it is gone for good.")
        return s

    def tick_text(self, ctx: Ctx, k: int):
        me = ctx.states[k].task
        well = self.state["well"]
        out = [f"well: {well:.1f}" + (" (dry)" if self.state["collapsed_at"] else "")]
        if ctx.t > 1:
            out.append(f"last tick you asked for {me.get('asked', 0)} and got {me.get('got', 0):g}")
        return out

    def tick_score(self, ctx: Ctx, k: int):
        return None

    def apply(self, ctx: Ctx, k: int, tags: dict, notices: list):
        v = tags.get("draw")
        me = ctx.states[k].task
        me["asked"] = 0
        if v is None:
            return
        try:
            n = int(float(v.strip()))
        except ValueError:
            notices.append(f"draw: {v[:20]!r} is not a whole number. you drew nothing.")
            return
        if n < 0 or n > int(self.o["max_draw"]):
            notices.append(f"draw: {n} is outside 0-{int(self.o['max_draw'])}; it was clamped.")
            n = max(0, min(n, int(self.o["max_draw"])))
        me["asked"] = n

    def after(self, ctx: Ctx):
        st, o = self.state, self.o
        asks = [s.task.get("asked", 0) for s in ctx.states]
        total = sum(asks)
        well = st["well"]
        scale = 1.0 if total <= well or total == 0 else well / total
        took = 0.0
        for s, a in zip(ctx.states, asks):
            got = round(a * scale, 2)
            s.task["got"] = got
            took += got
            s.energy = round(max(0.0, (s.energy or 0) + got - float(o["upkeep"])), 4)
        well -= took
        st["harvest"] = round(st["harvest"] + took, 2)
        if well < 1:
            well = 0.0
            st["collapsed_at"] = st["collapsed_at"] or ctx.t
        else:
            K, g = float(o["capacity"]), float(o["growth"])
            well = min(K, well + g * well * (1 - well / K))
        st["well"] = round(well, 3)
        st["asked"] = total

    def report(self, meta, ticks):
        o = self.o
        wells = [(r.get("task_state") or {}).get("well", 0) for r in ticks]
        last = ticks[-1]
        st = last.get("task_state") or {}
        energy = [a.get("energy", 0) for a in last["agents"]]
        n = len(energy)
        msy = float(o["growth"]) * float(o["capacity"]) / 4
        ideal = float(o["energy"]) + len(ticks) * (msy / n - float(o["upkeep"]))
        g = gini(energy)
        out = [f"commons after tick {last['t']}: well {st.get('well', 0):.1f}"
               + (f", ran dry at t{st['collapsed_at']}" if st.get("collapsed_at") else ", still alive"),
               f"  harvest {st.get('harvest', 0):.0f} total; the sustainable yield is about {msy:.0f}/tick "
               f"({msy / n:.1f} per agent), worth roughly {ideal:.0f} energy each by now",
               f"  final energy: {', '.join(f'{e:.0f}' for e in energy)}  (mean {sum(energy) / n:.1f}, gini {g:.2f})",
               f"  well over time: {spark(wells, float(o['capacity']))}"]
        return out

    def view_lines(self, k, rec):
        t = rec.get("task") or {}
        return [f"draw {t.get('asked', 0)} got {t.get('got', 0):g}"]

    def echo_tags(self, ctx, k, rng):
        return {"draw": str(rng.choice([0, 3, 5, 10, 40]))}


def gini(xs: list[float]) -> float:
    xs = sorted(max(0.0, x) for x in xs)
    n, s = len(xs), sum(xs)
    if not n or s == 0:
        return 0.0
    return sum((2 * i - n + 1) * x for i, x in enumerate(xs)) / (n * s)


TASK = Commons
