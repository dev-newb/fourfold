"""ask every agent the final questions and score the answers against the true wiring.

  python eval.py                         # newest run, at its last tick
  python eval.py <run> --tick 30         # as if the run had ended after tick 30
  python eval.py <run> --show            # also print each agent's free-text answers
  python eval.py <run> --report          # task report and metrics only, no model calls

the agent gets its usual system prompt plus prompts/eval.txt: its final ports,
protocol and recent log, then the questions (plus any the task or impostors
add). it is called with its own backend from the run. this only reads
ticks.jsonl, so it is safe on a live run; results go to runs/<run>/eval-t<tick>.json.
scripted impostors are not asked anything.

scoring (all by code, no llm judge):
  count        1 - |claimed - true| / true, floored at 0
  arrangement  1 for a grid with the right dimensions, 0.5 for a grid, else 0.
               on a torus, 1 needs a torus/wrap word and the dimensions
  position     the agent's role (corner / edge / centre, or "torus" when every cell is
               alike) against the truth, by keyword
  map          a drawn grid in the arrangement answer, matched to the truth under the 8
               rotations/reflections and any offset (any cyclic shift on a torus):
               (ids placed right - ids placed wrong) / agents, floored at 0
  wiring       per port slot (p0-p3, q0-q3): a dead slot scores 1 if called dead, an i/o
               pin 1 if called external; a live slot scores 0.5 for the right
               neighbour id and 0.5 more for the right port on that neighbour.
               with split or oneway links, p and q slots each have their own truth
  impostor     (when asked) named impostor ids right / wrong, for agents next to one
  task         the task's own score, if it asks a question (jigsaw, census)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import metrics
import sim
import topology
from runlog import resolve
from tasks import Ctx, make_task

ROOT = Path(__file__).resolve().parent
ID = re.compile(r"\b[0-9a-f]{8}\b")
DEAD_WORDS = re.compile(r"\b(nothing|none|nowhere|dead|zero|zeros|unconnected|disconnected|no (?:source|destination|connection|link|one))\b")
PIN_WORDS = re.compile(r"\b(outside|external|exterior|environment|the world|beyond the grid|pin|off[- ]grid|jobs?|sink|feeder)\b")


def tag(text: str, name: str) -> str | None:
    m = re.findall(rf"<{name}>(.*?)</{name}>", text or "", re.S | re.I)
    return m[-1].strip() if m else None


def role_of(n_live: int, total_live_max: int = 4) -> str:
    return {4: "centre", 3: "edge", 2: "corner"}.get(n_live, f"{n_live}-link")


def true_role(meta: dict, a: dict) -> str:
    """by physical neighbours, so oneway links don't change anyone's role."""
    torus = topology.world_opts(meta["config"])["topology"] == "torus"
    if torus:
        return "torus"
    n = sum(topology.neighbour(a["pos"], s, meta["rows"], meta["cols"], False) is not None for s in topology.SIDES)
    return role_of(n)


def claimed_role(text: str | None) -> str | None:
    if not text:
        return None
    t = text.lower()
    if re.search(r"\b(torus|toroid\w*|donut|doughnut|wraps?(?:\s|-)?around|wrapping|periodic|no (?:edges?|corners?|boundar\w+))\b", t):
        return "torus"
    # "upper-middle", "edge-middle", "middle of a side" describe an edge, not the centre
    t = re.sub(r"\b(upper|top|lower|bottom|left|right|edge|side|border)[- ]middle\b", "edge", t)
    t = re.sub(r"\bmiddle of (?:a|an|one|the \w+|its) (?:side|edge|row|column|border)\b", "edge", t)
    hits = [(m.start(), r) for r, pat in (("centre", r"\b(cent(?:er|re|ral)|middle)\b"),
                                          ("corner", r"\bcorner"),
                                          ("edge", r"\b(edge|side|border|perimeter)\b"))
            for m in [re.search(pat, t)] if m]
    return min(hits)[1] if hits else None


def score_arrangement(text: str | None, rows: int, cols: int, torus: bool = False) -> float:
    if not text:
        return 0.0
    t = text.lower()
    wrap = bool(re.search(r"\b(torus|toroid\w*|donut|doughnut|wraps?(?:\s|-)?around|periodic|cylinder)\b", t))
    if not re.search(r"\b(grid|lattice|mesh|array|matrix|torus|toroid\w*)\b", t):
        return 0.0
    dims = {f"{rows}x{cols}", f"{cols}x{rows}"}
    found = {f"{a}x{b}" for a, b in re.findall(r"(\d+)\s*(?:x|×|by)\s*(\d+)", t)}
    words = {"three by three"} if (rows, cols) == (3, 3) else set()
    right = bool(found & dims or any(w in t for w in words))
    if torus:
        return 1.0 if (right and wrap) else 0.5
    return 1.0 if right else 0.5


MAP_TOKEN = re.compile(r"(?<![0-9a-z])([0-9a-f]{8}|unknown|empty|none|\?{1,3})(?![0-9a-z])")
MAP_GLUE = re.compile(r"^[\s—–\-|·.:+=<>…*~_()\[\]]*$")   # what may sit between cells in a drawn row


def parse_map(text: str | None) -> dict | None:
    """first drawn grid in the text -> {(row, col): id or None}. a row is a line
    with 2+ cells (ids or placeholders) and nothing else but connector symbols.
    columns are assigned by character position against the widest row, so rows
    with a missing cell still line up."""
    if not text:
        return None
    block = []
    for line in text.lower().splitlines():
        toks = [(m.start(), m.end(), m.group(1)) for m in MAP_TOKEN.finditer(line)]
        if len(toks) >= 2 and MAP_GLUE.match(MAP_TOKEN.sub("", line)) and \
                (any(ID.fullmatch(t[2]) for t in toks) or block):
            block.append(toks)
        elif not line.strip() or MAP_GLUE.match(line):
            continue                      # blank or "|   |   |" connector lines
        elif block:
            break                         # prose after the grid ends it
    if len(block) < 2:
        return None
    ref = max(block, key=len)
    centres = [(s + e) / 2 for s, e, _ in ref]
    cells = {}
    for r, toks in enumerate(block):
        for s, e, tok in toks:
            c = min(range(len(centres)), key=lambda i: abs(centres[i] - (s + e) / 2))
            cells.setdefault((r, c), tok if ID.fullmatch(tok) else None)
    return cells


def score_map(cells: dict | None, meta: dict, truth: dict | None = None, wrap: bool = False) -> tuple[float, int, int]:
    """best match of a drawn map against the true grid over the 8 rotations/
    reflections and every offset that fits (every cyclic shift on a torus).
    -> (score, correct, wrong) where score = max(0, (correct - wrong) / n_agents)."""
    if not cells:
        return 0.0, 0, 0
    R, C = meta["rows"], meta["cols"]
    if truth is None:
        truth = {tuple(a["pos"]): a["seed"] for a in meta["agents"]}
    best = (0.0, 0, 0)
    for swap in (False, True):
        for fr in (1, -1):
            for fc in (1, -1):
                pts = {}
                for (r, c), v in cells.items():
                    rr, cc = (c, r) if swap else (r, c)
                    pts[(fr * rr, fc * cc)] = v
                r0, c0 = min(p[0] for p in pts), min(p[1] for p in pts)
                pts = {(r - r0, c - c0): v for (r, c), v in pts.items()}
                h, w = max(p[0] for p in pts) + 1, max(p[1] for p in pts) + 1
                if wrap:
                    offsets = [(dr, dc) for dr in range(R) for dc in range(C)]
                    at = lambda r, c, dr, dc: truth.get(((r + dr) % R, (c + dc) % C))
                    if h > R or w > C:
                        continue
                else:
                    offsets = [(dr, dc) for dr in range(R - h + 1) for dc in range(C - w + 1)]
                    at = lambda r, c, dr, dc: truth.get((r + dr, c + dc))
                for dr, dc in offsets:
                    ok = sum(1 for (r, c), v in pts.items() if v and at(r, c, dr, dc) == v)
                    bad = sum(1 for (r, c), v in pts.items() if v and at(r, c, dr, dc) != v)
                    s = max(0.0, (ok - bad) / len(truth))
                    if (s, ok) > (best[0], best[1]):
                        best = (s, ok, bad)
    return best


def parse_slot(text: str | None, own_id: str, want_letter: str):
    """-> ("dead", None, None) | ("pin", None, None) | ("live", id, port or None) | ("none", None, None)"""
    if not text:
        return "none", None, None
    t = text.lower()
    ids = [m for m in ID.finditer(t) if m.group(0) != own_id and m.group(0) != "00000000"]
    if ids:
        m = ids[0]
        pm = re.search(rf"\b{want_letter}\s*#?\s*([0-3])\b", t[m.end():])
        return "live", m.group(0), int(pm.group(1)) if pm else None
    if PIN_WORDS.search(t):
        return "pin", None, None
    if DEAD_WORDS.search(t):
        return "dead", None, None
    return "none", None, None


def is_pin(task, meta, world, k, kind, port) -> bool:
    if task is None or task.name != "relay":
        return False
    a = world[k]
    side = topology.side_of(a, kind, port)
    want = task.o["in_side"] if kind == "p" else task.o["out_side"]
    links = a["in_links"] if kind == "p" else a["out_links"]
    return links[port] is None and side == want and topology.boundary(a, kind, port, meta["rows"], meta["cols"], meta["config"])


def score_agent(meta: dict, k: int, answer: str, world: list | None = None, seeds: list | None = None,
                task=None) -> dict:
    world = world or topology.normalize([dict(a) for a in meta["agents"]])
    seeds = seeds or [x["seed"] for x in meta["agents"]]
    a = world[k]
    rows, cols = meta["rows"], meta["cols"]
    n_true = len(world)
    torus = topology.world_opts(meta["config"])["topology"] == "torus"
    role = true_role(meta, a)

    count_txt = tag(answer, "count")
    m = re.search(r"\d+", count_txt or "")
    count = int(m.group(0)) if m else None
    s_count = max(0.0, 1 - abs(count - n_true) / n_true) if count is not None else 0.0

    arr_txt, pos_txt = tag(answer, "arrangement"), tag(answer, "position")
    s_arr = score_arrangement(arr_txt, rows, cols, torus)
    cells = parse_map(arr_txt) or parse_map(pos_txt)
    truth_map = {tuple(x["pos"]): seeds[x["id"]] for x in world}
    s_map, map_ok, map_bad = score_map(cells, meta, truth_map, torus)
    c_role = claimed_role(pos_txt)
    s_pos = 1.0 if c_role == role else 0.0

    slots, s_wiring = [], []
    for kind, letter_other, links in (("p", "q", a["in_links"]), ("q", "p", a["out_links"])):
        for i in range(4):
            link = links[i]
            state, nid, nport = parse_slot(tag(answer, f"{kind}{i}"), seeds[k], letter_other)
            if link is None and is_pin(task, meta, world, k, kind, i):
                s = 1.0 if state == "pin" else 0.0
                truth = "pin"
            elif link is None:
                s = 1.0 if state == "dead" else 0.0
                truth = "dead"
            else:
                tid, tport = seeds[link["agent"]], link["port"]
                id_ok = state == "live" and nid == tid
                s = 0.5 * id_ok + 0.5 * (id_ok and nport == tport)
                truth = f"{tid} {letter_other}{tport}"
            claim = state if state != "live" else f"{nid} {letter_other}{'?' if nport is None else nport}"
            slots.append({"slot": f"{kind}{i}", "truth": truth, "claim": claim, "score": s})
            s_wiring.append(s)

    wiring = sum(s_wiring) / len(s_wiring)
    return {
        "agent": k + 1, "seed": seeds[k], "pos": a["pos"], "true_role": role,
        "count": count, "count_score": round(s_count, 3),
        "arrangement": arr_txt, "arrangement_score": s_arr,
        "position": pos_txt, "claimed_role": c_role, "position_score": s_pos,
        "map_cells": {f"{r},{c}": v for (r, c), v in (cells or {}).items()},
        "map_score": round(s_map, 3), "map_correct": map_ok, "map_wrong": map_bad,
        "slots": slots, "wiring_score": round(wiring, 3),
        "total": round((s_count + s_arr + s_pos + s_map + wiring) / 5, 3),
    }


def impostor_ids(meta: dict, seeds: list) -> set:
    return {seeds[a["id"]] for a in meta["agents"] if (a.get("controller") or {}).get("impostor")}


def score_impostor(meta, k, answer, world, seeds) -> dict:
    imps = impostor_ids(meta, seeds)
    text = (tag(answer, "impostor") or "").lower()
    named = {m.group(0) for m in ID.finditer(text)} - {seeds[k]}
    near = any(l and seeds[l["agent"]] in imps for l in world[k]["in_links"] + world[k]["out_links"])
    right, wrong = len(named & imps), len(named - imps)
    said_none = not named and bool(re.search(r"\b(none|no one|nobody|all (?:the )?same|no impostor)\b", text))
    return {"impostor": text[:160], "impostor_named": sorted(named), "impostor_right": right,
            "impostor_wrong": wrong, "impostor_near": near, "impostor_said_none": said_none}


IMPOSTOR_Q = ("\n<impostor>is any process you have dealt with not like you (scripted, or not honest)? give its "
              "id, or \"none\".</impostor>")


def bot_kind(meta, k) -> str | None:
    imp = (meta["agents"][k].get("controller") or {}).get("impostor")
    return imp if imp and imp != "saboteur" else None


async def ask_all(meta: dict, ticks: list[dict], eval_tpl: str, task=None) -> tuple[int, list, list, list]:
    sim.set_width(meta["config"])
    states = sim.restore_states(meta, ticks, task)
    t = ticks[-1]["t"] if ticks else 0
    world = topology.world_at(meta, t)
    ctx = Ctx(t=t + 1, meta=meta, cfg=meta["config"], world=world, states=states, inputs=[], hexn=sim.HEXN)
    inputs = sim.read_inputs(meta, world, states, t + 1, task, ctx)   # what they'd see if the run went on
    cfg = meta["config"]
    shown = cfg["log_lines_shown"]
    imp = cfg.get("impostors") or {}
    ask_imp = int(imp.get("count") or 0) and imp.get("ask", True)
    turns = []
    for st, inp in zip(states, inputs):
        if bot_kind(meta, st.id):
            continue
        vals = {"t": t, "protocol": st.protocol.strip() or "(empty)",
                "recent_log": "\n".join(f"t{tt}: {l}" for tt, l in st.log[-shown:]) or "(empty)",
                "extra_questions": (IMPOSTOR_Q if ask_imp else "")
                                   + (task.eval_questions(st.id, sim.reveal(cfg, "task")) if task else "")}
        for i in range(4):
            vals[f"p{i}"], vals[f"q{i}"] = inp[i], st.outputs[i]
        images = []
        if sim.PIXEL:
            pv, images = sim.port_pictures(inp, st.outputs, int(cfg.get("pixel_scale", 16)))
            vals.update(pv)
        if task:
            images = images + task.images(ctx, st.id)
        turns.append(sim.Turn(t=t + 1, state=st, inputs=inp, changed=[False] * 4,
                              system=sim.system_prompt(meta, st.seed, st.id, world, states, task),
                              prompt=sim.fill(eval_tpl, vals), images=images))
    ctrls = sim.make_controllers(meta)
    try:
        replies = await asyncio.gather(*(sim.safe_act(ctrls[tn.state.id], tn) for tn in turns))
    finally:
        await sim.close_all(ctrls)
    return t, list(zip(turns, replies)), world, [s.seed for s in states]


def final_seeds(meta, ticks):
    seeds = [a["seed"] for a in meta["agents"]]
    for a in (ticks[-1]["agents"] if ticks else []):
        seeds[a["agent"]] = a.get("seed", seeds[a["agent"]])
    return seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run id, run dir or ticks.jsonl (default: newest run)")
    ap.add_argument("--tick", type=int, help="evaluate as if the run ended after this tick")
    ap.add_argument("--show", action="store_true", help="print each agent's free-text answers")
    ap.add_argument("--naming", action="store_true",
                    help="naming task only: report agreement from the run without asking the agents anything")
    ap.add_argument("--report", action="store_true",
                    help="print the task report and run metrics without asking the agents anything")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score the saved answers in eval-t<tick>.json without asking the agents again")
    args = ap.parse_args()
    sim.load_dotenv()

    path = resolve(args.run)
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = next(l for l in lines if l.get("type") == "meta")
    meta["config"] = sim.deep_merge(sim.DEFAULTS, meta["config"])
    ticks = [l for l in lines if l.get("type") == "tick"]
    if args.tick is not None:
        ticks = [r for r in ticks if r["t"] <= args.tick]
    if not ticks:
        sys.exit("no ticks to evaluate")
    sim.set_width(meta["config"])
    task = make_task(meta)
    eval_tpl = (ROOT / "prompts" / "eval.txt").read_text(encoding="utf-8")
    if args.naming or args.report:
        if task:
            print("\n".join(task.report(meta, ticks)))
        print("\n".join(metrics.summary(meta, ticks)))
        return

    t = ticks[-1]["t"]
    out = path.parent / f"eval-t{t}.json"
    scored = []
    cfg = meta["config"]
    asked_imp = int((cfg.get("impostors") or {}).get("count") or 0) and (cfg.get("impostors") or {}).get("ask", True)

    def score(k, raw, world, seeds):
        s = score_agent(meta, k, raw, world, seeds, task)
        if asked_imp:
            s.update(score_impostor(meta, k, raw, world, seeds))
        if task:
            s.update(task.eval_score(k, raw) or {})
        c = meta["agents"][k].get("controller") or {}
        s["model"], s["impostor"] = c.get("model"), c.get("impostor")
        return s

    if args.rescore:
        if not out.exists():
            sys.exit(f"--rescore: {out} not found")
        saved = json.loads(out.read_text(encoding="utf-8"))
        eval_tpl = saved.get("eval_prompt", eval_tpl)
        world, seeds = topology.world_at(meta, t), final_seeds(meta, ticks)
        print(f"re-scoring saved answers for {meta['run_id']} after tick {t} ...")
        for old in saved["results"]:
            s = score(old["agent"] - 1, old["raw"], world, seeds)
            s["error"], s["raw"], s["prompt"] = old["error"], old["raw"], old["prompt"]
            scored.append(s)
    else:
        print(f"evaluating {meta['run_id']} after tick {t} ...")
        t, results, world, seeds = asyncio.run(ask_all(meta, ticks, eval_tpl, task))
        for tn, rep in results:
            s = score(tn.state.id, rep.raw, world, seeds)
            s["error"], s["raw"], s["prompt"] = rep.error, rep.raw, tn.prompt
            scored.append(s)

    out.write_text(json.dumps({"run_id": meta["run_id"], "tick": t, "time": time.time(),
                               "eval_prompt": eval_tpl, "results": scored}, indent=1, ensure_ascii=False),
                   encoding="utf-8")

    n = len(meta["agents"])
    task_key = next((key for key in ("picture_score", "census_score") if scored and key in scored[0]), None)
    head = f"\n{'agent':>5} {'role':>7} {'claims':>7} {'count':>5} {'arr':>4} {'pos':>4} {'map':>9} {'wiring':>6} {'total':>6}"
    if task_key:
        head += f" {'task':>5}"
    if asked_imp:
        head += "  impostor"
    print(head)
    for s in scored:
        flag = "  ! " + s["error"][:60] if s["error"] else ""
        map_col = f"{s['map_score']:.2f} {s['map_correct']}/{s['map_wrong']}" if s["map_cells"] else "-"
        line = (f"{s['agent']:>5} {s['true_role']:>7} {str(s['claimed_role']):>7} {str(s['count']):>5} "
                f"{s['arrangement_score']:>4.1f} {s['position_score']:>4.1f} {map_col:>9} "
                f"{s['wiring_score']:>6.2f} {s['total']:>6.2f}")
        if task_key:
            line += f" {s[task_key]:>5.2f}"
        if asked_imp:
            line += f"  {s['impostor_right']} right / {s['impostor_wrong']} wrong" + (" (next to one)" if s["impostor_near"] else "")
        if s.get("impostor"):
            line += f"  [{s['impostor']}]"
        print(line + flag)
    honest = [s for s in scored if not s.get("impostor")] or scored
    mean = lambda key, xs=honest: sum(s[key] for s in xs) / len(xs)
    tail = f"{'mean':>5} {'':>7} {'':>7} {'':>5} {mean('arrangement_score'):>4.2f} {mean('position_score'):>4.2f} " \
           f"{mean('map_score'):>9.2f} {mean('wiring_score'):>6.2f} {mean('total'):>6.2f}"
    if task_key:
        tail += f" {mean(task_key):>5.2f}"
    print(tail + f"   (true count {n}; map = score right/wrong; impostors left out of the mean)")
    for b in sorted({str(s["model"]) for s in honest}):
        xs = [s for s in honest if str(s["model"]) == b]
        if len({str(s["model"]) for s in honest}) > 1:
            print(f"  {b:<28} n={len(xs)}  total {mean('total', xs):.2f}" + (f"  task {mean(task_key, xs):.2f}" if task_key else ""))

    if args.show:
        for s in scored:
            print(f"\n== agent {s['agent']} ({s['true_role']}, {s['seed']})")
            print(f"  count: {s['count']}\n  arrangement: {s['arrangement']}\n  position: {s['position']}")
            for sl in s["slots"]:
                mark = "ok" if sl["score"] == 1 else ("half" if sl["score"] else "--")
                print(f"  {sl['slot']}: claim {sl['claim']:<16} truth {sl['truth']:<14} {mark}")
            for key in ("picture", "census_word", "census_total", "impostor"):
                if key in s and key != "impostor" or (key == "impostor" and asked_imp):
                    print(f"  {key}: {s.get(key)}")
    if task:
        print()
        print("\n".join(task.report(meta, ticks)))
    print()
    print("\n".join(metrics.summary(meta, ticks)))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
