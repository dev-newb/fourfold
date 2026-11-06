"""run a grid of configs x seeds, evaluate each run, and tabulate (and plot) the results.

the point is to turn one-off anecdotes into curves, e.g. where between 8 and
128 bits per port first contact stops happening:

  python sweep.py --config config.toml --vary port_bits=8,16,24,32,48,64,128 --seeds 3 --ticks 60
  python sweep.py --vary world.topology=grid,torus --vary world.links=paired,split --seeds 2
  python sweep.py --agent echo --vary port_bits=8,32,128 --seeds 2 --ticks 30 --no-eval   # free dry run

every combination of --vary values is run once per seed (seeds 1..n, so the
same seeds give the same worlds across settings). runs are named
sweep-<name>-<setting>-s<seed> and skipped if already finished, so a sweep can
be stopped and picked up again. results go to sweeps/<name>.csv, plus
sweeps/<name>.png when matplotlib is installed (x axis = the first --vary).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import metrics

ROOT = Path(__file__).resolve().parent


def run_id(name, combo, seed):
    tag = "-".join(f"{k.split('.')[-1]}{v}" for k, v in combo)
    return re.sub(r"[^0-9A-Za-z_.-]", "", f"sweep-{name}-{tag}-s{seed}")


def finished(rid: str, ticks: int) -> bool:
    p = ROOT / "runs" / rid / "ticks.jsonl"
    if not p.exists():
        return False
    last = p.read_text(encoding="utf-8").strip().splitlines()[-1]
    return json.loads(last).get("t", 0) >= ticks


def one(args, combo, seed):
    rid = run_id(args.name, combo, seed)
    log = []
    if not finished(rid, args.ticks):
        exists = (ROOT / "runs" / rid / "ticks.jsonl").exists()
        cmd = [sys.executable, "sim.py", "--ticks", str(args.ticks)]
        if exists:
            cmd += ["--resume", rid]
        else:
            cmd += ["--run-id", rid, "--world-seed", str(seed)]
            if args.config:
                cmd += ["--config", args.config]
            if args.agent:
                cmd += ["--agent", args.agent]
            for k, v in combo:
                cmd += ["--set", f"{k}={v}"]
            for s in args.set:
                cmd += ["--set", s]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        log.append(r.stdout[-400:] + r.stderr[-800:])
        if r.returncode:
            return {"run": rid, "error": "sim failed: " + r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "sim failed"}
    ev = sorted((ROOT / "runs" / rid).glob(f"eval-t{args.ticks}.json"))
    if args.eval and not ev:
        r = subprocess.run([sys.executable, "eval.py", rid], cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            return {"run": rid, "error": "eval failed: " + (r.stderr.strip().splitlines() or ["?"])[-1]}
        ev = sorted((ROOT / "runs" / rid).glob(f"eval-t{args.ticks}.json"))
    lines = [json.loads(l) for l in (ROOT / "runs" / rid / "ticks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = lines[0]
    ticks = [l for l in lines if l.get("type") == "tick"]
    row = {"run": rid, "seed": seed, **{k: v for k, v in combo}, **metrics.row(meta, ticks)}
    if ev:
        res = [r for r in json.loads(ev[-1].read_text(encoding="utf-8"))["results"] if not r.get("impostor")]
        if res:
            for key in ("total", "map_score", "wiring_score", "count_score", "position_score"):
                row[key] = round(sum(r[key] for r in res) / len(res), 3)
            for key in ("picture_score", "census_score"):
                if key in res[0]:
                    row[key] = round(sum(r[key] for r in res) / len(res), 3)
    ts = ticks[-1].get("task_state") or {}
    for key in ("correct", "solved_at", "done", "well", "collapsed_at", "harvest"):
        if key in ts:
            row[f"task_{key}"] = ts[key]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="base config (default: config.toml)")
    ap.add_argument("--vary", action="append", default=[], metavar="KEY=V1,V2,...",
                    help="a config key and the values to try (repeatable; combinations are crossed)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="fixed override for every run")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--agent", choices=["llm", "anthropic", "echo"])
    ap.add_argument("--name", help="sweep name (default: from the --vary keys)")
    ap.add_argument("--parallel", type=int, default=1, help="runs at once (each run already makes its calls in parallel)")
    ap.add_argument("--no-eval", dest="eval", action="store_false", help="skip eval.py (metrics only, no extra calls)")
    args = ap.parse_args()
    if not args.vary:
        sys.exit("give at least one --vary KEY=V1,V2")
    axes = []
    for v in args.vary:
        k, vals = v.split("=", 1)
        axes.append([(k.strip(), x.strip()) for x in vals.split(",")])
    args.name = args.name or "_".join(a[0][0].split(".")[-1] for a in axes)
    jobs = [(combo, s) for combo in itertools.product(*axes) for s in range(1, args.seeds + 1)]
    print(f"sweep {args.name}: {len(jobs)} runs of {args.ticks} ticks")
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        for row in ex.map(lambda j: one(args, *j), jobs):
            rows.append(row)
            print("  " + ("! " + row["error"] if "error" in row else
                          f"{row['run']}: contact {row.get('first_contact')}  text {row.get('first_text')}"
                          + (f"  total {row['total']:.2f}" if "total" in row else "")))
    out = ROOT / "sweeps"
    out.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / f"{args.name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out / (args.name + '.csv')}")
    table(rows, axes)
    plot(rows, axes, out / f"{args.name}.png", args.ticks)


def group(rows, axes):
    keys = [a[0][0] for a in axes]
    groups: dict = {}
    for r in rows:
        if "error" not in r:
            groups.setdefault(tuple(str(r.get(k)) for k in keys), []).append(r)
    return keys, groups


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def table(rows, axes):
    keys, groups = group(rows, axes)
    print("\n" + "  ".join(f"{k.split('.')[-1]:>10}" for k in keys)
          + f"  {'runs':>4}  {'contact%':>8}  {'contact t':>9}  {'text t':>7}  {'eval':>5}  {'map':>5}")
    for combo, rs in groups.items():
        hit = [r for r in rs if r.get("first_contact")]
        ft = mean([r.get("first_contact") for r in rs])
        tx = mean([r.get("first_text") for r in rs])
        ev = mean([r.get("total") for r in rs])
        mp = mean([r.get("map_score") for r in rs])
        f = lambda x, d=1: "-" if x is None else f"{x:.{d}f}"
        print("  ".join(f"{c:>10}" for c in combo)
              + f"  {len(rs):>4}  {100 * len(hit) / len(rs):>7.0f}%  {f(ft):>9}  {f(tx):>7}  {f(ev, 2):>5}  {f(mp, 2):>5}")


def plot(rows, axes, path, ticks):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: no plot)")
        return
    xkey = axes[0][0][0]
    xs = [v for _, v in axes[0]]
    others = list(itertools.product(*axes[1:])) or [()]
    num = all(re.fullmatch(r"-?\d+(\.\d+)?", x) for x in xs)
    pos = [float(x) for x in xs] if num else list(range(len(xs)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for combo in others:
        label = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in combo) or None
        ct, ev = [], []
        for x in xs:
            rs = [r for r in rows if "error" not in r and str(r.get(xkey)) == x
                  and all(str(r.get(k)) == v for k, v in combo)]
            # runs with no contact count as "after the end" so the curve shows the cliff
            ct.append(mean([r.get("first_contact") or ticks + 1 for r in rs]))
            ev.append(mean([r.get("total") for r in rs]))
        a1.plot(pos, ct, marker="o", label=label)
        if any(e is not None for e in ev):
            a2.plot(pos, [e if e is not None else float("nan") for e in ev], marker="o", label=label)
    for a, title in ((a1, "first confirmed contact (tick; above the line = never)"), (a2, "mean eval score")):
        a.set_xlabel(xkey)
        a.set_title(title, fontsize=10)
        if num:
            a.set_xscale("log", base=2)
        a.set_xticks(pos, xs)
        a.grid(alpha=0.3)
    a1.axhline(ticks, color="grey", lw=0.8, ls="--")
    if others != [()]:
        a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
