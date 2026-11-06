"""invariants of the wiring for every topology x links x rewire mode, over many seeds.

  every in-link has the matching out-link on the other end, and vice versa
  a wire joins physical neighbours (with wraparound on a torus), facing each other
  paired: p_i and q_i face the same side; split: never the same side
  oneway: each physical wire carries exactly one direction
  grid: a port facing off the edge is dead; torus: nothing is dead (paired/split)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import topology  # noqa: E402


def check(agents, rows, cols, topo, links):
    torus = topo == "torus"
    at = {tuple(a["pos"]): a for a in agents}
    assert len(at) == len(agents), "two agents on one cell"
    for a in agents:
        for i in range(4):
            l = a["in_links"][i]
            if l:
                b = agents[l["agent"]]
                assert b["out_links"][l["port"]] == {"agent": a["id"], "port": i}
                s_in = a["in_sides"][i]
                s_out = b["out_sides"][l["port"]]
                assert s_out == topology.OPP[s_in], "wire ends don't face each other"
                assert topology.neighbour(a["pos"], s_in, rows, cols, torus) == tuple(b["pos"])
            elif links != "oneway":
                assert not torus, "dead input on a torus"
                assert topology.neighbour(a["pos"], a["in_sides"][i], rows, cols, False) is None
            l = a["out_links"][i]
            if l:
                assert agents[l["agent"]]["in_links"][l["port"]] == {"agent": a["id"], "port": i}
        if links == "paired":
            assert a["in_sides"] == a["out_sides"]
        if links == "split":
            assert all(x != y for x, y in zip(a["in_sides"], a["out_sides"]))
    if links == "oneway":
        for a in agents:
            for s in topology.SIDES:
                nb = topology.neighbour(a["pos"], s, rows, cols, torus)
                if nb is None:
                    continue
                b = at[nb]
                fwd = a["out_links"][a["out_sides"].index(s)] is not None
                back = b["out_links"][b["out_sides"].index(topology.OPP[s])] is not None
                assert fwd != back, "a oneway wire carries both directions, or neither"


def main():
    n = 0
    for topo in ("grid", "torus"):
        for links in ("paired", "split", "oneway"):
            for mode in ("ports", "places", "swap"):
                for rows, cols in ((3, 3), (2, 4), (4, 4), (1, 3), (2, 2)):
                    for seed in range(15):
                        cfg = {"rows": rows, "cols": cols,
                               "world": {"topology": topo, "links": links, "rewire_at": [3, 7], "rewire_mode": mode}}
                        meta = {"run_id": f"w{n}", "world_seed": seed, "rows": rows, "cols": cols, "config": cfg,
                                "agents": topology.build_world(cfg, random.Random(seed))}
                        if rows * cols < 2 and mode == "swap":
                            continue
                        for t in (1, 3, 8):
                            check(topology.world_at(meta, t), rows, cols, topo, links)
                            n += 1
    print(f"wiring invariants hold in {n} worlds")


if __name__ == "__main__":
    main()
