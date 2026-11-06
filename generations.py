"""culture: a chain of runs where each generation starts with the protocols the
previous one ended with. kirby-style iterated learning, on a wire.

  python generations.py --config presets/culture.toml --gens 5 --ticks 40
  python generations.py --from <run> --gens 3          # continue a lineage from an existing run

generation 1 is a normal run. each later one is a fresh world (new wiring, new
ids, new seed) whose agents find, at tick 1, a protocol written by someone in
the previous generation, picked by [culture].pick (best eval score by default,
so each generation is evaluated before the next one starts). with no label set
they are not told it was inherited: it is simply theirs, ids that no longer
exist and all. watch whether the protocol shrinks and stabilises over
generations, and whether first contact comes sooner.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import metrics

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config for every generation (default: config.toml)")
    ap.add_argument("--gens", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--from", dest="start", help="an existing run to inherit from for the first generation")
    ap.add_argument("--agent", choices=["llm", "anthropic", "echo"])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--name", default=time.strftime("lineage-%Y%m%d-%H%M%S"))
    ap.add_argument("--no-eval", dest="eval", action="store_false",
                    help="don't evaluate (then 'best' falls back to the longest protocol)")
    args = ap.parse_args()

    prev = args.start
    rows = []
    for g in range(1, args.gens + 1):
        rid = f"{args.name}-g{g}"
        cmd = [sys.executable, "sim.py", "--ticks", str(args.ticks), "--run-id", rid]
        if args.config:
            cmd += ["--config", args.config]
        if args.agent:
            cmd += ["--agent", args.agent]
        for s in args.set:
            cmd += ["--set", s]
        if prev:
            cmd += ["--set", f'culture.inherit_from="{prev}"']
        print(f"generation {g}: {rid}" + (f" (inherits from {prev})" if prev else ""), flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode:
            sys.exit(f"generation {g} failed")
        if args.eval:
            subprocess.run([sys.executable, "eval.py", rid], cwd=ROOT, check=False)
        path = ROOT / "runs" / rid / "ticks.jsonl"
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        meta, ticks = lines[0], [l for l in lines if l.get("type") == "tick"]
        row = metrics.row(meta, ticks)
        protos = [a["protocol"] for a in ticks[-1]["agents"]]
        row["protocol_words"] = round(sum(len(p.split()) for p in protos) / len(protos), 1)
        ev = sorted((ROOT / "runs" / rid).glob("eval-t*.json"))
        if ev:
            res = json.loads(ev[-1].read_text(encoding="utf-8"))["results"]
            row["total"] = round(sum(r["total"] for r in res) / len(res), 3)
        rows.append((g, rid, row))
        prev = rid

    print(f"\n{'gen':>3}  {'run':<36} {'contact':>7} {'text':>5} {'words':>6} {'eval':>5}")
    for g, rid, r in rows:
        f = lambda x: "-" if x is None else str(x)
        print(f"{g:>3}  {rid:<36} {f(r['first_contact']):>7} {f(r['first_text']):>5} "
              f"{r['protocol_words']:>6} {f(r.get('total')):>5}")


if __name__ == "__main__":
    main()
