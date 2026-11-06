"""wiring: what feeds each input port and where each output port goes.

every agent has four input ports p0..p3 and four output ports q0..q3. each
port faces one physical side of the agent's cell (north, east, south, west),
shuffled per agent. two neighbouring cells share a wire: what one writes on
the side facing the other arrives on the other's input facing back.

  links = "paired"   p_i and q_i face the same side (the original). writing on
                     q_i talks to whoever feeds p_i, so echoes come straight back.
  links = "split"    p_i and q_i face different sides. q_i still reaches a
                     neighbour, just never the one that feeds p_i, so an echo
                     comes back on a different port from the one it left on.
  links = "oneway"   paired sides, but every wire carries data in one direction
                     only, chosen at random. half the ports go quiet, and the only
                     way to hear yourself is around a loop of agents.

  topology = "grid"  outer sides are dead (or I/O pins, if the task wants them).
  topology = "torus" the grid wraps around. nothing is dead and no cell is special.

the true wiring lives in the meta record (epoch 0). rewiring mid-run is
deterministic from the world seed, so world_at() can rebuild the wiring of any
tick for the sim, the evaluator and the viewer alike.
"""
from __future__ import annotations

import copy
import random

SIDES = ("N", "E", "S", "W")
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
DELTA = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


def world_opts(cfg: dict) -> dict:
    w = cfg.get("world") or {}
    return {
        "topology": w.get("topology", "grid"),
        "links": w.get("links", "paired"),
        "rewire_at": sorted(int(x) for x in (w.get("rewire_at") or [])),
        "rewire_mode": w.get("rewire_mode", "ports"),
    }


def _derange(rng: random.Random, ref: list[str]) -> list[str]:
    """a shuffle of SIDES with no position equal to ref's."""
    while True:
        out = list(SIDES)
        rng.shuffle(out)
        if all(a != b for a, b in zip(out, ref)):
            return out


def neighbour(pos, side, rows, cols, torus):
    r, c = pos
    dr, dc = DELTA[side]
    rr, cc = r + dr, c + dc
    if torus:
        return rr % rows, cc % cols
    return (rr, cc) if 0 <= rr < rows and 0 <= cc < cols else None


def link(agents: list[dict], rows: int, cols: int, opts: dict, rng: random.Random | None):
    """fill in_links / out_links from each agent's pos and in/out sides.
    oneway direction per wire is drawn from rng (epoch 0: the world rng)."""
    torus = opts["topology"] == "torus"
    at = {tuple(a["pos"]): a for a in agents}
    for a in agents:
        a["in_links"] = [None] * 4
        a["out_links"] = [None] * 4
    wires = []  # (a, side, b): a's output on `side` lands on b's input facing OPP[side]
    for a in agents:
        for s in SIDES:
            nb = neighbour(a["pos"], s, rows, cols, torus)
            if nb is not None:
                wires.append((a, s, at[nb]))
    alive = None
    if opts["links"] == "oneway":
        # one physical wire = the pair of directions (a,s)->b and (b,OPP s)->a. keep one.
        alive, seen = set(), set()
        for a, s, b in wires:
            key = frozenset({(a["id"], s), (b["id"], OPP[s])})
            if key in seen:
                continue
            seen.add(key)
            keep = (a["id"], s) if (rng or random).random() < 0.5 else (b["id"], OPP[s])
            alive.add(keep)
    for a, s, b in wires:
        if alive is not None and (a["id"], s) not in alive:
            continue
        q = a["out_sides"].index(s)
        p = b["in_sides"].index(OPP[s])
        b["in_links"][p] = {"agent": a["id"], "port": q}
        a["out_links"][q] = {"agent": b["id"], "port": p}
    for a in agents:   # the original field, kept so old tools and runs still read
        a["links"] = list(a["in_links"]) if opts["links"] == "paired" else None


def build_world(cfg: dict, rng: random.Random) -> list[dict]:
    rows, cols = cfg["rows"], cfg["cols"]
    opts = world_opts(cfg)
    agents = []
    for r in range(rows):
        for c in range(cols):
            sides = list(SIDES)
            rng.shuffle(sides)            # same draws, in the same order, as the original sim
            agents.append({
                "id": len(agents),
                "pos": [r, c],
                "seed": f"{rng.getrandbits(32):08x}",
                "port_sides": sides,      # paired: port i faces this side both ways
                "in_sides": list(sides),
                "out_sides": list(sides),
            })
    if opts["links"] == "split":
        for a in agents:
            a["out_sides"] = _derange(rng, a["in_sides"])
    link(agents, rows, cols, opts, rng)
    return agents


def normalize(agents: list[dict]) -> list[dict]:
    """runs made before in/out links existed only have port_sides and links."""
    for a in agents:
        if "in_links" not in a:
            a["in_sides"] = a["out_sides"] = a["port_sides"]
            a["in_links"] = list(a["links"])
            a["out_links"] = list(a["links"])
    return agents


def _rewire(agents: list[dict], rows: int, cols: int, opts: dict, rng: random.Random) -> list[dict]:
    agents = copy.deepcopy(agents)
    mode = opts["rewire_mode"]
    if mode == "ports":
        for a in agents:
            s = list(SIDES)
            rng.shuffle(s)
            a["in_sides"] = a["port_sides"] = s
            a["out_sides"] = _derange(rng, s) if opts["links"] == "split" else s
    elif mode == "places":
        cells = [list(a["pos"]) for a in agents]
        rng.shuffle(cells)
        for a, p in zip(agents, cells):
            a["pos"] = p
    elif mode == "swap":
        x, y = rng.sample(range(len(agents)), 2)
        agents[x]["pos"], agents[y]["pos"] = agents[y]["pos"], agents[x]["pos"]
    else:
        raise SystemExit(f"config: unknown world.rewire_mode {mode!r}")
    link(agents, rows, cols, opts, rng)
    return agents


_cache: dict = {}


def epoch(meta: dict, t: int) -> int:
    return sum(1 for x in world_opts(meta["config"])["rewire_at"] if x <= t)


def world_at(meta: dict, t: int) -> list[dict]:
    """the wiring in force during tick t. rewiring at tick x applies from x on."""
    opts = world_opts(meta["config"])
    e = epoch(meta, t)
    key = (meta.get("run_id"), meta.get("world_seed"), e)
    if key not in _cache:
        agents = normalize(copy.deepcopy(meta["agents"]))
        for x in opts["rewire_at"][:e]:
            agents = _rewire(agents, meta["rows"], meta["cols"], opts,
                             random.Random(f"{meta['world_seed']}:rewire:{x}"))
        _cache[key] = agents
    return _cache[key]


def side_of(a: dict, kind: str, port: int) -> str:
    return (a["in_sides"] if kind == "p" else a["out_sides"])[port]


def port_on(a: dict, kind: str, side: str) -> int:
    return (a["in_sides"] if kind == "p" else a["out_sides"]).index(side)


def boundary(a: dict, kind: str, port: int, rows: int, cols: int, cfg: dict) -> bool:
    """true if this port faces off the edge of a (non-torus) grid."""
    if world_opts(cfg)["topology"] == "torus":
        return False
    return neighbour(a["pos"], side_of(a, kind, port), rows, cols, False) is None
