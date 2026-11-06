"""run every option and preset for a few ticks with the free echo agents, then
evaluate, report, resume and render each run. no model calls, no key needed.

  python tests/smoke.py              # everything
  python tests/smoke.py sudoku lever # only cases whose name contains one of these
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

CASES = {
    "default": [],
    "torus": ["world.topology=\"torus\""],
    "split": ["world.links=\"split\""],
    "oneway": ["world.links=\"oneway\""],
    "torus-split": ["world.topology=\"torus\"", "world.links=\"split\""],
    "noise": ["world.noise=0.01"],
    "latency": ["world.latency=[0,3]"],
    "rewire-places": ["world.rewire_at=[5,10]", "world.rewire_mode=\"places\""],
    "rewire-swap-split": ["world.rewire_at=[6]", "world.rewire_mode=\"swap\"", "world.links=\"split\""],
    "rewire-ports": ["world.rewire_at=[4]", "world.rewire_mode=\"ports\""],
    "impostor-parrot": ["impostors.count=2", "impostors.kind=\"parrot\""],
    "impostor-mirror": ["impostors.count=1", "impostors.kind=\"mirror\""],
    "impostor-silent": ["impostors.count=1", "impostors.kind=\"silent\"", "impostors.cells=[5]"],
    "impostor-echo": ["impostors.count=1", "impostors.kind=\"echo\""],
    "impostor-saboteur": ["impostors.count=1", "impostors.kind=\"saboteur\""],
    "cost": ["cost.per_bit=0.05", "cost.energy=20", "cost.regen=0.5"],
    "flips": ["cost.max_flips=24"],
    "lever": ["lever.enabled=true", "lever.threshold=2", "cost.max_flips=30",
              "lever.effects=[\"sight\",\"memory\",\"flips\"]"],
    "lever-revealed": ["lever.enabled=true", "remove_mystery.task=true"],
    "turnover": ["turnover.period=5", "turnover.warn=2"],
    "reveal-all": ["remove_mystery.peers=true", "remove_mystery.grid=true", "remove_mystery.position=true",
                   "remove_mystery.sides=true", "remove_mystery.wiring=true", "remove_mystery.ticks=true",
                   "remove_mystery.text=true", "remove_mystery.impostors=true", "impostors.count=1"],
    "drift": ["mystery.evaluation=\"none\"", "mystery.show_changed=false"],
    "narrow": ["port_bits=8"],
    "pixels": ["port_mode=\"pixels\"", "pixel_size=5"],
    "mix": ["mix.backends=[\"a\",\"b\"]", "backends.a.kind=\"echo\"", "backends.b.kind=\"echo\"",
            "backends.b.shift=2", "mix.assign=\"blocks\""],
    "task-naming": ["task=\"naming\""],
    "task-sudoku": ["task=\"sudoku\""],
    "task-sudoku-2x2": ["task=\"sudoku\"", "rows=2", "cols=2", "remove_mystery.task=true", "remove_mystery.score=true"],
    "task-jigsaw": ["task=\"jigsaw\""],
    "task-census": ["task=\"census\"", "remove_mystery.task=true"],
    "task-relay": ["task=\"relay\""],
    "task-relay-reverse": ["task=\"relay\"", "tasks.relay.job=\"reverse\"", "remove_mystery.task=true",
                           "world.links=\"split\""],
    "task-commons": ["task=\"commons\"", "cost.per_bit=0.02"],
    "task-commons-revealed": ["task=\"commons\"", "remove_mystery.task=true", "remove_mystery.peers=true"],
}


def sh(args, **kw):
    r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True, **kw)
    return r.returncode, (r.stdout + r.stderr)


def main():
    want = sys.argv[1:]
    stamp = time.strftime("%H%M%S")
    presets = sorted((ROOT / "presets").glob("*.toml"))
    cases = [(name, None, sets) for name, sets in CASES.items()]
    cases += [(f"preset-{p.stem}", p, []) for p in presets]
    if want:
        cases = [c for c in cases if any(w in c[0] for w in want)]
    env = dict(os.environ, SDL_VIDEODRIVER="dummy")
    failed = []
    for name, preset, sets in cases:
        rid = f"smoke-{stamp}-{name}"
        args = ["sim.py", "--agent", "echo", "--ticks", "12", "--world-seed", "7", "--run-id", rid]
        if preset:
            args += ["--config", str(preset)]
        for s in sets:
            args += ["--set", s]
        steps = [("run", args),
                 ("resume", ["sim.py", "--resume", rid, "--ticks", "16"]),
                 ("eval", ["eval.py", rid]),
                 ("report", ["eval.py", rid, "--report"]),
                 ("view", ["view.py", rid, "--snapshot", str(ROOT / "runs" / rid / "snap.png"), "--debug",
                           "--tick", "14"])]
        ok = True
        for label, a in steps:
            rc, out = sh(a, env=env)
            if rc:
                ok = False
                failed.append(name)
                print(f"FAIL {name} at {label}:\n" + "\n".join(out.splitlines()[-12:]))
                break
        if ok:
            print(f"ok   {name}")
    print(f"\n{len(cases) - len(failed)}/{len(cases)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
