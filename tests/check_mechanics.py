"""checks the per-tick mechanics against fresh echo runs: latency, noise, costs,
the lever, turnover, impostors and the task hooks. no model calls.

  python tests/check_mechanics.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import sim  # noqa: E402
import topology  # noqa: E402

STAMP = time.strftime("%H%M%S")


def run(name, *sets, ticks=20, seed=11):
    rid = f"check-{STAMP}-{name}"
    args = [sys.executable, "sim.py", "--agent", "echo", "--ticks", str(ticks), "--world-seed", str(seed),
            "--run-id", rid, "--set", "echo.malformed_rate=0"]
    for s in sets:
        args += ["--set", s]
    subprocess.run(args, cwd=ROOT, check=True, capture_output=True)
    lines = [json.loads(l) for l in (ROOT / "runs" / rid / "ticks.jsonl").read_text().splitlines()]
    meta = lines[0]
    meta["config"] = sim.deep_merge(sim.DEFAULTS, meta["config"])
    return meta, [l for l in lines[1:] if l["type"] == "tick"]


def ok(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        global failed
        failed += 1


failed = 0


def main():
    # latency: p_i at tick t is the source's q at the end of tick t-1-d
    meta, ticks = run("latency", "world.latency=[0,3]")
    good = total = 0
    for x, rec in enumerate(ticks):
        for a in topology.world_at(meta, rec["t"]):
            for i, l in enumerate(a["in_links"]):
                if not l:
                    continue
                d = sim.latency_of(meta, a["id"], i)
                src = x - 1 - d
                want = ticks[src]["agents"][l["agent"]]["outputs"][l["port"]] if src >= 0 else "0" * 32
                total += 1
                good += rec["agents"][a["id"]]["inputs"][i] == want
    ok(good == total, f"latency: {good}/{total} inputs are the source's output from 1+d ticks back")
    lats = {sim.latency_of(meta, k, i) for k in range(9) for i in range(4)}
    ok(lats == {0, 1, 2, 3}, f"latency: per-wire delays drawn from 0..3 ({sorted(lats)})")

    # noise: inputs are the source outputs with a few bits flipped
    meta, ticks = run("noise", "world.noise=0.02")
    flips = bits = 0
    for x, rec in enumerate(ticks[1:], start=1):
        for a in topology.world_at(meta, rec["t"]):
            for i, l in enumerate(a["in_links"]):
                if l:
                    v = rec["agents"][a["id"]]["inputs"][i]
                    w = ticks[x - 1]["agents"][l["agent"]]["outputs"][l["port"]]
                    flips += sim.popcount(v, w)
                    bits += 128
    ok(0.01 < flips / bits < 0.03, f"noise: {flips / bits:.4f} of bits flipped in transit (rate 0.02)")

    # cost: energy falls by per_bit * flipped bits, regen added, unaffordable writes rejected
    meta, ticks = run("cost", "cost.per_bit=0.05", "cost.energy=3", "cost.regen=0.25")
    bad = 0
    for x, rec in enumerate(ticks[1:], start=1):
        for a in rec["agents"]:
            prev = ticks[x - 1]["agents"][a["agent"]]
            spent = sum(sim.popcount(o, p) for o, p in zip(a["outputs"], prev["outputs"])) * 0.05
            if abs(prev["energy"] - spent + 0.25 - a["energy"]) > 1e-6:
                bad += 1
            if a["energy"] < -1e-9:
                bad += 1
    rejected = sum(1 for rec in ticks for a in rec["agents"] for e in a["errors"] if "costs" in e)
    ok(bad == 0, "cost: every energy change equals regen minus 0.05 per flipped bit, never negative")
    ok(rejected > 0, f"cost: {rejected} writes rejected for lack of energy")

    meta, ticks = run("flips", "cost.max_flips=40")
    over = sum(1 for x, rec in enumerate(ticks[1:], start=1) for a in rec["agents"]
               if sum(sim.popcount(o, p) for o, p in zip(a["outputs"], ticks[x - 1]["agents"][a["agent"]]["outputs"])) > 40)
    ok(over == 0, "flips: no agent ever changed more than 40 bits in a tick")

    # lever: engages at the threshold, effects show up the tick after, notices sent
    meta, ticks = run("lever", "lever.enabled=true", "lever.threshold=2", "store_prompts=true", ticks=30)
    eng = [r["t"] for r in ticks if r["lever"]["engaged"]]
    ok(bool(eng), f"lever: engaged on {len(eng)} ticks")
    consistent = all(ticks[x]["lever"]["engaged"] == ticks[x - 1]["lever"]["next"] for x in range(1, len(ticks)))
    ok(consistent, "lever: engagement takes effect the tick after the threshold is met")
    if eng:
        r = next(r for r in ticks if r["lever"]["engaged"])
        p = r["agents"][0]["prompt"]
        ok("from " in p and "your outputs go to:" in p, "lever: sight notes appear while engaged")
        ok(all(a.get("monitored") is False for a in r["agents"]), "lever: logs flagged unmonitored while engaged")
        first = ticks[[x["t"] for x in ticks].index(r["t"]) - 1]
        ok(any("something changed" in n for n in first["agents"][0].get("events", [])), "lever: change recorded as an event")
        ok("something changed since last tick" in p, "lever: agents told about the change in their next prompt")

    # turnover: one agent replaced every 5 ticks, warned 2 ticks ahead, new id, wiped memory
    meta, ticks = run("turnover", "turnover.period=5", "turnover.warn=2", "store_prompts=true")
    reborn = [(r["t"], r["reborn"]) for r in ticks if "reborn" in r]
    ok([t for t, _ in reborn] == [5, 10, 15, 20], f"turnover: replacements at {[t for t, _ in reborn]}")
    t, k = reborn[0]
    before, after = ticks[t - 2]["agents"][k], ticks[t - 1]["agents"][k]
    ok(before["seed"] != after["seed"], "turnover: the newcomer has a new id")
    ok("will be reset in 1 tick" in ticks[t - 2]["agents"][k]["prompt"], "turnover: the victim is warned")
    ok("echo agent " + after["seed"] in after["protocol"], "turnover: the newcomer starts with an empty protocol")

    # impostors: assigned, never replaced, scripted ones answer in format
    meta, ticks = run("impostor", "impostors.count=2", "impostors.kind=\"parrot\"", "turnover.period=3")
    imps = [a["id"] for a in meta["agents"] if a["controller"]["impostor"]]
    ok(len(imps) == 2, f"impostors: two parrots at agents {[i + 1 for i in imps]}")
    ok(all(r["reborn"] not in imps for r in ticks if "reborn" in r), "impostors: never replaced by turnover")
    ok(all(not a["errors"] for r in ticks for a in r["agents"] if a["agent"] in imps), "impostors: parrots never malformed")

    # sudoku: fixed digits held, puzzle unique, conflicts reported
    meta, ticks = run("sudoku", "task=\"sudoku\"", "store_prompts=true")
    from tasks.sudoku import solve_count
    puz = meta["task_data"]["puzzle"]
    ok(solve_count([r[:] for r in puz], 3) == 1, "sudoku: the puzzle has exactly one solution")
    from tasks import make_task
    task = make_task(meta)
    held = all(all(f == "." or c == f for c, f in zip(a["task"]["cells"], task.fixed(a["agent"])))
               for r in ticks for a in r["agents"] if "task" in a)
    ok(held, "sudoku: fixed digits never change")
    ok("conflicts:" in ticks[3]["agents"][0]["prompt"], "sudoku: conflict count shown in mystery mode")

    # relay: west pins carry jobs, completions counted only when the answer reaches an east pin
    meta, ticks = run("relay", "task=\"relay\"", "store_prompts=true")
    w = topology.world_at(meta, 1)
    west = [a for a in w if a["pos"][1] == 0]
    shown = [sim.ascii_view(ticks[0]["agents"][a["id"]]["inputs"][a["in_sides"].index("W")]) for a in west]
    ok(all(s.endswith("=") and "+" in s for s in shown), f"relay: west pins show jobs {shown}")
    mid = [a for a in w if a["pos"][1] == 1]
    ok(all(ticks[0]["agents"][a["id"]]["inputs"][a["in_sides"].index(s)] == "0" * 32
           for a in mid for s in ("W",) if a["in_links"][a["in_sides"].index(s)] is None), "relay: inner cells have no pins")

    # commons: the well and energies follow the rules
    meta, ticks = run("commons", "task=\"commons\"")
    st = [r["task_state"]["well"] for r in ticks]
    ok(all(0 <= x <= 1000 for x in st), f"commons: well stays in [0, capacity] (ends {st[-1]:.0f})")
    ok(all(a["energy"] >= 0 for r in ticks for a in r["agents"]), "commons: energy never negative")

    # census: every local answer is wrong, the global winner is right
    meta, _ = run("census", "task=\"census\"", ticks=1)
    d = meta["task_data"]
    tot = {}
    for led in d["ledgers"]:
        for wd, x in led:
            tot[wd] = tot.get(wd, 0) + x
    ok(max(tot, key=tot.get) == d["winner"], f"census: global winner is {d['winner']} {d['total']}")
    from tasks.census import local
    ok(all(max(local(l), key=local(l).get) != d["winner"] for l in d["ledgers"]), "census: no local ledger points to it")

    print(f"\n{'all checks passed' if not failed else f'{failed} checks failed'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
