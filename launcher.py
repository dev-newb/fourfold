"""point-and-click launcher: pick an experiment, up to four models, who sits where, go.

  python launcher.py

left: the experiments (every preset). middle: what the selected one does.
right: four model slots (click a name to cycle forward, right-click back, click
the effort to change it) and the grid, coloured by slot. click a cell to hand it
to the next slot, or use the layout buttons. launch writes launches/<run>.toml
(the preset plus your models) and starts sim.py with the live viewer. the runs
list at the bottom lets you watch, stop, evaluate or report on any of them.

only models whose provider key is in .env are offered (see models.toml).
"free test" runs the same experiment with the fake echo agents: no calls, no cost.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import textwrap
import threading
import time
import tomllib
from pathlib import Path

import pygame

import sim
from render import DIM, ERR, FAINT, FG, HI_ON, MODEL_COLORS, MONO, PROTO_FG

ROOT = Path(__file__).resolve().parent
LAUNCHES = ROOT / "launches"
W, H = 1360, 880
BG = (0, 0, 0)
PANEL = (12, 12, 12)
EFFORTS = ["low", "medium", "high"]

GROUPS = [
    ("the original", ["original"]),
    ("the world", ["torus", "oneway", "static", "earthquake", "impostor", "saboteur", "thrift", "culture", "drift"]),
    ("real tasks", ["sudoku", "sudoku-open", "jigsaw", "census", "relay"]),
    ("agency", ["commons", "lever", "turnover"]),
]
ORIGINAL_TEXT = ("the original experiment. nine models in a 3x3 grid, four 128-bit ports each, and nothing "
                 "else: nobody tells them there is a grid, or neighbours, or that anyone else exists. every "
                 "tick they read their inputs, write their outputs and keep a small notebook. at the end "
                 "they are asked to describe their situation, and scored on it.")


# ------------------------------------------------------------------ data

def load_env_keys() -> set:
    sim.load_dotenv()
    return {k for k, v in os.environ.items() if v and k.endswith("_API_KEY")}


def load_catalog(keys: set) -> list[dict]:
    with open(ROOT / "models.toml", "rb") as f:
        cat = tomllib.load(f)
    out = []
    for m in cat.get("models", []):
        prov = dict(cat["providers"][m["provider"]])
        if prov.get("api_key_env") not in keys:
            continue
        b = {"kind": "llm", "inherit": False, **prov, "model": m["id"], "temperature": "",
             "max_concurrency": 9}
        for k in ("max_tokens_param", "max_tokens", "rpm", "retries", "timeout", "system_role"):
            if k in m:
                b[k] = m[k]
        out.append({"id": m["id"], "provider": m["provider"], "note": m.get("note", ""),
                    "effort": m.get("effort", True), "backend": b})
    return out


def preset_info(name: str) -> dict:
    if name == "original":
        cfg = sim.load_config(str(ROOT / "config.toml"))
        return {"name": name, "path": ROOT / "config.toml", "text": ORIGINAL_TEXT, "cfg": cfg}
    path = ROOT / "presets" / f"{name}.toml"
    lines = []
    for l in path.read_text(encoding="utf-8").splitlines():
        if not l.startswith("#"):
            break
        lines.append(l[1:].strip())
    text = " ".join(lines).replace("  ", " ")
    import io
    import contextlib
    with contextlib.redirect_stderr(io.StringIO()):
        cfg = sim.load_config(str(path))
    return {"name": name, "path": path, "text": text, "cfg": cfg}


def ensure_config():
    if not (ROOT / "config.toml").exists():
        txt = (ROOT / "config.example.toml").read_text(encoding="utf-8")
        (ROOT / "config.toml").write_text(txt.replace('model = "gpt-6.1-sol"', 'model = "gpt-5.4-mini"'),
                                          encoding="utf-8")


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    return json.dumps(str(v))


def last_tick(rid: str) -> int:
    p = ROOT / "runs" / rid / "ticks.jsonl"
    if not p.exists():
        return 0
    with open(p, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 400_000))
        tail = f.read().splitlines()
    for line in reversed(tail):
        m = re.match(rb'\{"type": "tick", "t": (\d+)', line)
        if m:
            return int(m.group(1))
    return 0


# ------------------------------------------------------------------ app

class Launcher:
    def __init__(self):
        ensure_config()
        self.keys = load_env_keys()
        self.catalog = load_catalog(self.keys)
        self.presets = {n: preset_info(n) for _, names in GROUPS for n in names
                        if n == "original" or (ROOT / "presets" / f"{n}.toml").exists()}
        self.sel = "original"
        first = next((i for i, m in enumerate(self.catalog) if m["id"] == "gpt-5.4-mini"), 0)
        self.slots = [{"model": first if self.catalog else None, "effort": "medium"}] + \
                     [{"model": None, "effort": "medium"} for _ in range(3)]
        self.ticks = self.presets[self.sel]["cfg"]["ticks"]
        self.cells = [0] * 9
        self.watch = True
        self.free = not self.catalog
        self.procs: dict[str, subprocess.Popen] = {}
        self.panel: dict | None = None        # {"title", "lines", "scroll"}
        self.status = ""
        self.status_t = 0.0
        self.hits: list = []                  # (rect, left_fn, right_fn)
        self.fonts = {}
        self.mouse = (-1, -1)
        LAUNCHES.mkdir(exist_ok=True)

    def font(self, px, bold=False):
        if (px, bold) not in self.fonts:
            self.fonts[(px, bold)] = pygame.font.SysFont(MONO, px, bold=bold)
        return self.fonts[(px, bold)]

    def say(self, msg):
        self.status, self.status_t = msg, time.time()

    # -------------------------------------------------------------- model slots
    def active(self) -> list[int]:
        return [i for i, s in enumerate(self.slots) if s["model"] is not None]

    def cycle_model(self, i, step):
        n = len(self.catalog)
        if not n:
            self.say("no models: put OPENAI_API_KEY or GEMINI_API_KEY in .env")
            return
        opts = list(range(n)) + ([None] if i > 0 else [])
        cur = self.slots[i]["model"]
        j = (opts.index(cur) + step) % len(opts) if cur in opts else 0
        self.slots[i]["model"] = opts[j]
        if opts[j] is None:
            self.cells = [0 if c == i else c for c in self.cells]

    def cycle_effort(self, i):
        e = self.slots[i]["effort"]
        self.slots[i]["effort"] = EFFORTS[(EFFORTS.index(e) + 1) % len(EFFORTS)]

    def layout(self, kind):
        act = self.active()
        n = self.rows() * self.cols()
        if kind == "all":
            self.cells = [act[0]] * n
        elif kind == "mix":
            self.cells = [act[k % len(act)] for k in range(n)]
        elif kind == "rows":
            self.cells = [act[(k // self.cols()) % len(act)] for k in range(n)]
        elif kind == "cols":
            self.cells = [act[(k % self.cols()) % len(act)] for k in range(n)]
        elif kind == "random":
            self.cells = [random.choice(act) for _ in range(n)]

    def rows(self):
        return int(self.presets[self.sel]["cfg"]["rows"])

    def cols(self):
        return int(self.presets[self.sel]["cfg"]["cols"])

    def click_cell(self, k, step):
        act = self.active()
        cur = self.cells[k] if self.cells[k] in act else act[0]
        self.cells[k] = act[(act.index(cur) + step) % len(act)]

    def select(self, name):
        self.sel = name
        self.ticks = self.presets[name]["cfg"]["ticks"]
        n = self.rows() * self.cols()
        self.cells = (self.cells + [self.cells[-1] if self.cells else 0] * n)[:n]

    # -------------------------------------------------------------- launching
    def write_launch(self, rid: str) -> Path:
        p = self.presets[self.sel]
        lines = [f"# launched {time.strftime('%Y-%m-%d %H:%M')} from launcher.py",
                 f"base = {toml_value(str(p['path']))}", f"ticks = {self.ticks}"]
        if self.sel == "culture":
            prev = self.inherit_source()
            if prev:
                lines += ["", "[culture]", f"inherit_from = {toml_value(prev)}"]
        if not self.free:
            used = sorted({c for c in self.cells if self.slots[c]["model"] is not None})
            names = [f"slot{i + 1}" for i in used]
            lines += ["", "[mix]", f"backends = {toml_value(names)}", 'assign = "cells"',
                      f"cells = {toml_value([f'slot{c + 1}' for c in self.cells])}"]
            for i in used:
                m = self.catalog[self.slots[i]["model"]]
                lines += ["", f"[backends.slot{i + 1}]"]
                for k, v in m["backend"].items():
                    lines.append(f"{k} = {toml_value(v)}")
                if m["effort"]:
                    lines += ["", f"[backends.slot{i + 1}.extra_body]",
                              f"reasoning_effort = {toml_value(self.slots[i]['effort'])}"]
        path = LAUNCHES / f"{rid}.toml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def inherit_source(self) -> str | None:
        for rid in self.run_ids():
            if not rid.startswith("culture") and last_tick(rid) > 0:
                return rid
        return None

    def launch(self):
        if not self.free and not self.active():
            self.say("pick a model in slot 1 first")
            return
        rid = f"{self.sel}-{time.strftime('%m%d-%H%M%S')}" + ("-free" if self.free else "")
        cfg_path = self.write_launch(rid)
        cmd = [sys.executable, str(ROOT / "sim.py"), "--config", str(cfg_path), "--run-id", rid]
        if self.free:
            cmd += ["--agent", "echo"]
        if self.watch:
            cmd.append("--view")
        log = open(LAUNCHES / f"{rid}.log", "w", encoding="utf-8")
        self.procs[rid] = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
        log.close()
        who = "echo agents (free)" if self.free else ", ".join(
            sorted({self.catalog[self.slots[c]["model"]]["id"] for c in self.cells}))
        self.say(f"launched {rid} with {who}" + ("; the viewer will open in a moment" if self.watch else ""))

    def run_ids(self) -> list[str]:
        files = sorted(LAUNCHES.glob("*.toml"), key=lambda f: f.stat().st_mtime, reverse=True)
        return [f.stem for f in files]

    def run_total(self, rid) -> int:
        try:
            with open(LAUNCHES / f"{rid}.toml", "rb") as f:
                return int(tomllib.load(f).get("ticks", 0))
        except Exception:
            return 0

    def watch_run(self, rid):
        subprocess.Popen([sys.executable, str(ROOT / "view.py"), rid, "--live"], cwd=ROOT,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)

    def stop_run(self, rid):
        p = self.procs.get(rid)
        if p and p.poll() is None:
            p.terminate()
            self.say(f"stopped {rid}. it can be resumed: python sim.py --resume {rid}")

    def background(self, title, args):
        self.panel = {"title": title, "lines": ["working on it ..."], "scroll": 0}

        def work():
            r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
            text = (r.stdout + ("\n" + r.stderr if r.returncode else "")).strip() or "(no output)"
            if self.panel and self.panel["title"] == title:
                self.panel["lines"] = text.splitlines()
        threading.Thread(target=work, daemon=True).start()

    def log_tail(self, rid):
        p = LAUNCHES / f"{rid}.log"
        text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else "(no log)"
        self.panel = {"title": f"log of {rid}", "lines": text.splitlines()[-400:], "scroll": 10 ** 6}

    # -------------------------------------------------------------- drawing helpers
    def text(self, s, txt, pos, col=FG, px=15, bold=False):
        img = self.font(px, bold).render(txt, True, col)
        s.blit(img, pos)
        return img.get_rect(topleft=pos)

    def button(self, s, label, rect, fn, right=None, on=False, col=None, px=14):
        r = pygame.Rect(rect)
        hover = r.collidepoint(self.mouse)
        pygame.draw.rect(s, (40, 40, 40) if hover else PANEL, r, border_radius=4)
        pygame.draw.rect(s, col or (HI_ON if on else FAINT), r, 1, border_radius=4)
        img = self.font(px).render(label, True, col or (HI_ON if on else FG))
        s.blit(img, img.get_rect(center=r.center))
        self.hits.append((r, fn, right))
        return r

    def wrap(self, s, txt, x, y, width_px, col=PROTO_FG, px=15, max_lines=99):
        cw = self.font(px).size("M")[0]
        lh = self.font(px).get_linesize()
        n = 0
        for para in txt.split("\n"):
            for line in textwrap.wrap(para, max(10, width_px // cw)) or [""]:
                if n >= max_lines:
                    return y
                self.text(s, line, (x, y), col, px)
                y += lh
                n += 1
        return y

    # -------------------------------------------------------------- the frame
    def draw(self, s: pygame.Surface):
        self.hits = []
        s.fill(BG)
        self.text(s, "fourfold", (28, 22), FG, 26, True)
        self.text(s, "agentic cellular automata  ·  launcher", (180, 31), DIM, 14)
        keys = []
        for env, name in (("OPENAI_API_KEY", "openai"), ("GEMINI_API_KEY", "google"), ("ANTHROPIC_API_KEY", "anthropic")):
            keys.append((name, env in self.keys))
        x = W - 28
        for name, ok in reversed(keys):
            img = self.font(13).render(f"{name} {'key ok' if ok else 'no key'}", True, (120, 225, 130) if ok else FAINT)
            x -= img.get_width()
            s.blit(img, (x, 31))
            x -= 22
        pygame.draw.line(s, FAINT, (28, 64), (W - 28, 64))

        # --- experiments
        y = 82
        for title, names in GROUPS:
            self.text(s, title.upper(), (28, y), DIM, 12, True)
            y += 20
            for n in names:
                if n not in self.presets:
                    continue
                r = pygame.Rect(28, y, 250, 24)
                sel = n == self.sel
                if sel:
                    pygame.draw.rect(s, (30, 22, 8), r, border_radius=3)
                    pygame.draw.rect(s, HI_ON, pygame.Rect(28, y, 3, 24))
                self.text(s, n, (40, y + 3), HI_ON if sel else (FG if r.collidepoint(self.mouse) else PROTO_FG), 15)
                self.hits.append((r, lambda n=n: self.select(n), None))
                y += 25
            y += 8

        # --- description
        p = self.presets[self.sel]
        cfg = p["cfg"]
        x0, wd = 310, 430
        self.text(s, p["name"], (x0, 82), HI_ON, 22, True)
        facts = [f"{cfg['rows']}x{cfg['cols']} grid", f"{cfg.get('port_bits', 128)}-bit ports"]
        if cfg.get("task"):
            facts.append(f"task: {cfg['task']}")
        w = cfg.get("world") or {}
        if w.get("topology") != "grid":
            facts.append(w.get("topology"))
        if w.get("links") != "paired":
            facts.append(f"{w.get('links')} links")
        self.text(s, "  ·  ".join(facts), (x0, 114), DIM, 13)
        y = self.wrap(s, p["text"], x0, 142, wd, PROTO_FG, 15, 26)
        if self.sel == "culture":
            src = self.inherit_source()
            y = self.wrap(s, f"inherits from: {src or 'nothing yet (launch another experiment first)'}",
                          x0, y + 10, wd, HI_ON, 14)
        if self.sel == "jigsaw":
            self.wrap(s, "needs a vision model: every model in the list can see pictures.", x0, y + 10, wd, DIM, 13)
        self.button(s, "watch the original showcase run", (x0, 640, 300, 30),
                    lambda: self.watch_run(str(ROOT / "examples" / "sol-50")))

        # --- model slots
        x1 = 780
        self.text(s, "MODELS", (x1, 82), DIM, 12, True)
        y = 104
        for i, slot in enumerate(self.slots):
            col = MODEL_COLORS[i]
            pygame.draw.rect(s, col, pygame.Rect(x1, y + 4, 14, 22), border_radius=3)
            self.text(s, str(i + 1), (x1 + 3, y + 6), BG, 13, True)
            m = self.catalog[slot["model"]] if slot["model"] is not None else None
            label = m["id"] if m else "— off —"
            self.button(s, label, (x1 + 24, y, 250, 30), lambda i=i: self.cycle_model(i, 1),
                        lambda i=i: self.cycle_model(i, -1), col=col if m else FAINT)
            if m and m["effort"]:
                self.button(s, slot["effort"], (x1 + 282, y, 86, 30), lambda i=i: self.cycle_effort(i), col=DIM)
            if m:
                self.text(s, m["provider"], (x1 + 376, y), DIM, 12)
                if m["note"]:
                    self.text(s, m["note"][:26], (x1 + 376, y + 15), FAINT, 12)
            y += 40
        self.text(s, "click a model to change it (right-click: back), click effort to change it", (x1, y), FAINT, 12)

        # --- grid
        y += 30
        self.text(s, "WHO SITS WHERE", (x1, y), DIM, 12, True)
        y += 22
        R, C = self.rows(), self.cols()
        n = R * C
        if len(self.cells) != n:
            self.cells = (self.cells + [0] * n)[:n]
        act = self.active()
        cs = min(78, 260 // max(R, C))
        for k in range(n):
            r_, c_ = divmod(k, C)
            rect = pygame.Rect(x1 + c_ * (cs + 6), y + r_ * (cs + 6), cs, cs)
            slot = self.cells[k] if self.cells[k] in act else (act[0] if act else 0)
            col = MODEL_COLORS[slot] if not self.free else DIM
            pygame.draw.rect(s, PANEL, rect, border_radius=4)
            pygame.draw.rect(s, col, rect, 2, border_radius=4)
            if self.free:
                self.text(s, "echo", (rect.x + 8, rect.y + 8), DIM, 13)
            else:
                m = self.catalog[self.slots[slot]["model"]]["id"] if act else "?"
                self.text(s, str(slot + 1), (rect.x + 8, rect.y + 6), col, 20, True)
                short = m.replace("gemini-", "gem-").replace("-preview", "")
                parts = textwrap.wrap(short, max(6, (cs - 10) // 7))[:3]
                for j, part in enumerate(parts):
                    self.text(s, part, (rect.x + 6, rect.bottom - 6 - (len(parts) - j) * 14), PROTO_FG, 11)
            if act and not self.free:
                self.hits.append((rect, lambda k=k: self.click_cell(k, 1), lambda k=k: self.click_cell(k, -1)))
        gx = x1 + C * (cs + 6) + 14
        for j, (lab, kind) in enumerate((("all slot 1", "all"), ("mix", "mix"), ("by rows", "rows"),
                                         ("by columns", "cols"), ("random", "random"))):
            self.button(s, lab, (gx, y + j * 36, 130, 28), lambda kind=kind: act and self.layout(kind))
        y += R * (cs + 6) + 6
        self.text(s, "click a cell to give it to the next model", (x1, y), FAINT, 12)

        # --- launch bar
        y = 640
        x = x1
        self.text(s, "ticks", (x, y + 6), DIM, 14)
        x += 56
        for lab, d in (("-10", -10), ("-1", -1)):
            self.button(s, lab, (x, y, 38, 30), lambda d=d: self.set_ticks(d))
            x += 42
        self.text(s, f"{self.ticks:>4}", (x + 2, y + 5), FG, 18, True)
        x += 56
        for lab, d in (("+1", 1), ("+10", 10)):
            self.button(s, lab, (x, y, 38, 30), lambda d=d: self.set_ticks(d))
            x += 42
        calls = self.ticks * n
        self.text(s, f"{calls} calls" if not self.free else "free", (x + 8, y + 7), DIM, 13)
        y += 42
        self.button(s, ("[x] " if self.watch else "[ ] ") + "watch live", (x1, y, 150, 30), self.toggle_watch,
                    on=self.watch)
        self.button(s, ("[x] " if self.free else "[ ] ") + "free test (echo agents)", (x1 + 160, y, 230, 30),
                    self.toggle_free, on=self.free)
        self.button(s, "LAUNCH", (x1 + 400, y - 42, 150, 72), self.launch, col=HI_ON, px=20)

        # --- runs
        y0 = 740
        pygame.draw.line(s, FAINT, (28, y0 - 10), (W - 28, y0 - 10))
        self.text(s, "RUNS", (28, y0), DIM, 12, True)
        y = y0 + 20
        for rid in self.run_ids()[:4]:
            t, total = last_tick(rid), self.run_total(rid)
            p_ = self.procs.get(rid)
            running = p_ is not None and p_.poll() is None
            failed = p_ is not None and p_.poll() not in (None, 0)
            state = ("running" if running else ("failed" if failed else ("done" if t >= total > 0 else "stopped")))
            col = HI_ON if running else (ERR if failed else PROTO_FG)
            self.text(s, f"{rid:<34} {state:<8} t{t}/{total}", (28, y + 5), col, 14)
            bx = 560
            acts = [("watch", lambda rid=rid: self.watch_run(rid)), ("log", lambda rid=rid: self.log_tail(rid))]
            if running:
                acts.append(("stop", lambda rid=rid: self.stop_run(rid)))
            elif t > 0:
                acts += [("report", lambda rid=rid: self.background(f"report on {rid}", ["eval.py", rid, "--report"])),
                         ("evaluate", lambda rid=rid: self.background(f"evaluation of {rid}", ["eval.py", rid]))]
            for lab, fn in acts:
                self.button(s, lab, (bx, y, 84, 24), fn, px=13)
                bx += 90
            y += 28
        if not self.run_ids():
            self.text(s, "nothing launched yet", (28, y + 5), FAINT, 14)

        if self.status and time.time() - self.status_t < 12:
            self.text(s, self.status, (x1, 716), HI_ON, 13)

        if self.panel:
            self.draw_panel(s)

    def draw_panel(self, s):
        r = pygame.Rect(90, 70, W - 180, H - 140)
        pygame.draw.rect(s, (8, 8, 8), r)
        pygame.draw.rect(s, HI_ON, r, 1)
        self.text(s, self.panel["title"], (r.x + 16, r.y + 12), HI_ON, 16, True)
        self.button(s, "close", (r.right - 90, r.y + 8, 76, 26), self.close_panel)
        f = self.font(13)
        lh = f.get_linesize()
        rows = (r.h - 60) // lh
        lines = self.panel["lines"]
        top = max(0, min(self.panel["scroll"], len(lines) - rows))
        self.panel["scroll"] = top
        for i, line in enumerate(lines[top:top + rows]):
            s.blit(f.render(line[:170], True, PROTO_FG), (r.x + 16, r.y + 46 + i * lh))
        # only "close" works while the panel is up; the catch-all goes first so it is tried last
        self.hits = [(pygame.Rect(0, 0, W, H), lambda: None, None)] + \
                    [h for h in self.hits if h[1] == self.close_panel]

    def close_panel(self):
        self.panel = None

    def set_ticks(self, d):
        self.ticks = max(1, min(1000, self.ticks + d))

    def toggle_watch(self):
        self.watch = not self.watch

    def toggle_free(self):
        if not self.catalog:
            self.say("no model keys in .env, so only free test runs are possible")
            return
        self.free = not self.free

    # -------------------------------------------------------------- loop
    def click(self, pos, button):
        for rect, left, right in reversed(self.hits):
            if rect.collidepoint(pos):
                fn = right if button == 3 else left
                if fn:
                    fn()
                return


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", metavar="PNG", help="render one frame to a file and exit (for checking)")
    ap.add_argument("--select", help="preset to select")
    ap.add_argument("--slots", help="comma-separated model ids for slots 1-4")
    ap.add_argument("--layout", choices=["all", "mix", "rows", "cols", "random"])
    args = ap.parse_args()
    if args.snapshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    if args.snapshot:
        app = Launcher()
        if args.select:
            app.select(args.select)
        for i, mid in enumerate((args.slots or "").split(",")[:4] if args.slots else []):
            ids = [m["id"] for m in app.catalog]
            app.slots[i]["model"] = ids.index(mid) if mid in ids else None
        if args.layout:
            app.layout(args.layout)
        canvas = pygame.Surface((W, H))
        app.draw(canvas)
        pygame.image.save(canvas, args.snapshot)
        print(f"wrote {args.snapshot}")
        return
    pygame.display.set_caption("fourfold launcher")
    dw, dh = pygame.display.get_desktop_sizes()[0]
    scale = min(1.0, (dw * 0.95) / W, (dh * 0.9) / H)
    win = pygame.display.set_mode((int(W * scale), int(H * scale)), pygame.RESIZABLE)
    app = Launcher()
    canvas = pygame.Surface((W, H))
    clock = pygame.time.Clock()
    while True:
        ww, wh = win.get_size()
        k = min(ww / W, wh / H)
        ox, oy = (ww - W * k) / 2, (wh - H * k) / 2
        mx, my = pygame.mouse.get_pos()
        app.mouse = ((mx - ox) / k, (my - oy) / k)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                if app.panel:
                    app.panel = None
                else:
                    pygame.quit()
                    return
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3):
                app.click(((ev.pos[0] - ox) / k, (ev.pos[1] - oy) / k), ev.button)
            if ev.type == pygame.MOUSEWHEEL and app.panel:
                app.panel["scroll"] = max(0, app.panel["scroll"] - ev.y * 3)
        app.draw(canvas)
        win.fill(BG)
        win.blit(pygame.transform.smoothscale(canvas, (int(W * k), int(H * k))) if k != 1 else canvas, (ox, oy))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
