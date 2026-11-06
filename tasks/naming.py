"""naming: a point for every input port that shows exactly your own q0.

the original task, unchanged: its rule text lives in sim.rules_text and the
score is shown every tick. the grid maxes out when everyone agrees on one value.
"""
from __future__ import annotations

from tasks import Ctx, Task, spark


def naming_score(inputs, q0: str) -> int:
    """input ports showing exactly the agent's own q0. zero never counts."""
    if not q0.strip("0"):
        return 0
    return sum(v == q0 for v in inputs)


class Naming(Task):
    name = "naming"

    def tick_score(self, ctx: Ctx, k: int):
        live = sum(l is not None for l in ctx.world[k]["in_links"])
        return naming_score(ctx.inputs[k], ctx.states[k].outputs[0]), live

    def report(self, meta: dict, ticks: list[dict]) -> list[str]:
        agents = meta["agents"]
        best = sum(a.get("score_max", 0) for a in ticks[-1]["agents"]) or 1
        totals = [sum(a.get("score", 0) for a in rec["agents"]) for rec in ticks]
        last = ticks[-1]
        q0 = [a["outputs"][0] for a in last["agents"]]
        groups: dict[str, list[int]] = {}
        for i, v in enumerate(q0):
            if v.strip("0"):
                groups.setdefault(v, []).append(i + 1)
        sizes = sorted((len(g) for g in groups.values()), reverse=True)
        links = [a.get("in_links") or a.get("links") or [] for a in agents]
        edges = {tuple(sorted((a["id"], l["agent"]))) for a, ls in zip(agents, links) for l in ls if l}
        agreed = sum(q0[x] == q0[y] and bool(q0[x].strip("0")) for x, y in edges)
        first_full = next((rec["t"] for rec, s in zip(ticks, totals) if s == best), None)
        out = [f"naming task after tick {last['t']}",
               f"  grid score now {totals[-1]}/{best}, peak {max(totals)}"
               + (f", first full score at t{first_full}" if first_full else ", never full"),
               f"  distinct nonzero q0 values: {len(groups)}  (cluster sizes {sizes or '-'})",
               f"  neighbour pairs holding the same q0: {agreed}/{len(edges)}"]
        for v, ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            out.append(f"    {v}  agents {ids}")
        out.append("  per-agent score on the grid:")
        for r in range(meta["rows"]):
            row = []
            for c in range(meta["cols"]):
                a = last["agents"][r * meta["cols"] + c]
                row.append(f"{a.get('score', '-')}/{a.get('score_max', '-')}")
            out.append("    " + " ".join(row))
        out.append(f"  total over time: {spark(totals, best)}")
        return out


TASK = Naming
