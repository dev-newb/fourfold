"""sudoku: one box per process. the grid of agents is the grid of boxes.

on a 3x3 grid of agents this is an ordinary 9x9 sudoku (2x2 agents: 4x4).
each agent holds the box at its own position and sees only that box. rows and
columns run across boxes, so a box can only be finished with information from
the boxes in its row and column, and the corners have to hear about the far
side of the grid through the edge cells. the channel is the hard part.

mystery voice: "you hold a block of cells; some digits are fixed; you are told
how many of your filled cells are in conflict, not what a conflict is." the
conflict count is a side channel: a cell can clash with a digit nobody has
told the agent about, which is evidence that someone else exists.

revealed voice: it is a sudoku, your box is (r, c), your neighbours hold the rest.

[tasks.sudoku]
givens = 0              # digits handed out, over the whole grid. 0 = 30 for 9x9, 6 for 4x4
show_conflicts = true   # the per-tick conflict count (the only feedback in mystery mode)
"""
from __future__ import annotations

import random

from tasks import Ctx, Task, spark


def box_cells(n: int, br: int, bc: int) -> list[tuple[int, int]]:
    return [(br * n + i, bc * n + j) for i in range(n) for j in range(n)]


def candidates(g, r, c, n):
    N = n * n
    used = set(g[r]) | {g[i][c] for i in range(N)}
    r0, c0 = r - r % n, c - c % n
    used |= {g[i][j] for i in range(r0, r0 + n) for j in range(c0, c0 + n)}
    return [d for d in range(1, N + 1) if d not in used]


def solve_count(g, n, limit=2) -> int:
    N = n * n
    best, bc = None, None
    for r in range(N):
        for c in range(N):
            if not g[r][c]:
                cs = candidates(g, r, c, n)
                if not cs:
                    return 0
                if best is None or len(cs) < len(bc):
                    best, bc = (r, c), cs
    if best is None:
        return 1
    r, c = best
    total = 0
    for d in bc:
        g[r][c] = d
        total += solve_count(g, n, limit - total)
        g[r][c] = 0
        if total >= limit:
            break
    return total


def fill_grid(g, n, rng) -> bool:
    N = n * n
    for r in range(N):
        for c in range(N):
            if not g[r][c]:
                cs = candidates(g, r, c, n)
                rng.shuffle(cs)
                for d in cs:
                    g[r][c] = d
                    if fill_grid(g, n, rng):
                        return True
                g[r][c] = 0
                return False
    return True


def make_puzzle(n: int, givens: int, rng: random.Random):
    N = n * n
    sol = [[0] * N for _ in range(N)]
    fill_grid(sol, n, rng)
    puz = [row[:] for row in sol]
    cells = [(r, c) for r in range(N) for c in range(N)]
    rng.shuffle(cells)
    left = N * N
    for r, c in cells:
        if left <= givens:
            break
        keep = puz[r][c]
        puz[r][c] = 0
        if solve_count([row[:] for row in puz], n) != 1:
            puz[r][c] = keep
        else:
            left -= 1
    return puz, sol


class Sudoku(Task):
    name = "sudoku"
    defaults = {"givens": 0, "show_conflicts": True}

    @property
    def n(self) -> int:
        return int(self.cfg["rows"])

    @classmethod
    def validate(cls, cfg):
        if cfg["rows"] != cfg["cols"] or cfg["rows"] not in (2, 3):
            raise SystemExit("config: task sudoku needs a 2x2 or 3x3 grid of agents (rows = cols = 2 or 3)")

    @property
    def tags(self):
        N = self.n * self.n
        return {"cells": f"{N} characters, your cells row by row: a digit 1-{N}, or . for blank"}

    def setup(self, rng):
        n = self.n
        givens = int(self.o["givens"]) or (30 if n == 3 else 6)
        puz, sol = make_puzzle(n, givens, rng)
        boxes = {}
        for a in self.meta["agents"]:
            br, bc = a["pos"]
            boxes[str(a["id"])] = [br, bc]
        return {"puzzle": puz, "solution": sol, "boxes": boxes}

    def box(self, k):
        return self.data["boxes"][str(k)]

    def fixed(self, k) -> str:
        n, puz = self.n, self.data["puzzle"]
        br, bc = self.box(k)
        return "".join(str(puz[r][c]) if puz[r][c] else "." for r, c in box_cells(n, br, bc))

    def init_agent(self, k):
        return {"cells": self.fixed(k)}

    # ---------------------------------------------------------- the full grid
    def grid(self, states) -> list[list[int]]:
        n = self.n
        N = n * n
        g = [[0] * N for _ in range(N)]
        for st in states:
            br, bc = self.box(st.id)
            cells = (st.task or {}).get("cells") or self.fixed(st.id)
            for (r, c), ch in zip(box_cells(n, br, bc), cells):
                g[r][c] = int(ch) if ch.isdigit() else 0
        return g

    def conflicts(self, g) -> set[tuple[int, int]]:
        n = self.n
        N = n * n
        bad = set()
        groups = [[(r, c) for c in range(N)] for r in range(N)] + [[(r, c) for r in range(N)] for c in range(N)]
        groups += [box_cells(n, br, bc) for br in range(n) for bc in range(n)]
        for grp in groups:
            seen: dict[int, list] = {}
            for r, c in grp:
                if g[r][c]:
                    seen.setdefault(g[r][c], []).append((r, c))
            for cells in seen.values():
                if len(cells) > 1:
                    bad.update(cells)
        return bad

    # ---------------------------------------------------------- prompts
    def system_text(self, k, reveal):
        n = self.n
        N = n * n
        if reveal:
            br, bc = self.box(k)
            return (f"the task: together, the processes are solving one {N}x{N} sudoku. the full grid is cut "
                    f"into {N} boxes of {n}x{n} cells, one box per process, and the boxes are laid out the "
                    f"same way the processes are: your box is box row {br + 1}, box column {bc + 1}. every row, "
                    f"column and box of the full grid must end up holding each digit 1-{N} exactly once. you "
                    f"see only your own box; what the other boxes hold you can only learn over the wire. "
                    f"your box is shown every tick. write it back as <cells>, row by row. fixed digits (in "
                    f"brackets) cannot be changed. a conflict is a digit that appears twice in a row, column "
                    f"or box of the full grid.")
        return (f"you also hold a block of {n}x{n} cells, shown every tick. some cells hold fixed digits (in "
                f"brackets) that cannot be changed. you may fill every other cell with a digit from 1 to {N} "
                f"or leave it blank (.) by writing all {N} cells, row by row, as <cells>. "
                + ("each tick you are told how many of your filled cells are in conflict. you are not told "
                   "what a conflict is. " if self.o["show_conflicts"] else "")
                + "at the end, your cells will be checked too.")

    def tick_text(self, ctx: Ctx, k: int):
        n = self.n
        cells = ctx.states[k].task.get("cells") or self.fixed(k)
        fixed = self.fixed(k)
        rows = []
        for i in range(n):
            row = []
            for j in range(n):
                ch, fx = cells[i * n + j], fixed[i * n + j]
                row.append(f"[{fx}]" if fx != "." else f" {ch} ")
            rows.append("".join(row))
        out = ["your cells (fixed digits in brackets):"] + rows
        if self.o["show_conflicts"]:
            g = self.grid(ctx.states)
            bad = self.conflicts(g)
            br, bc = self.box(k)
            mine = sum(1 for rc in box_cells(n, br, bc) if rc in bad)
            out.append(f"conflicts: {mine}")
        if ctx.reveal("score"):
            out.append(f"correct cells in your box: {self.correct(ctx.states[k])}/{n * n}")
        return out

    def correct(self, st) -> int:
        n, sol = self.n, self.data["solution"]
        br, bc = self.box(st.id)
        cells = st.task.get("cells") or self.fixed(st.id)
        return sum(1 for (r, c), ch in zip(box_cells(n, br, bc), cells) if ch == str(sol[r][c]))

    def tick_score(self, ctx: Ctx, k: int):
        return self.correct(ctx.states[k]), self.n * self.n

    def apply(self, ctx: Ctx, k: int, tags: dict, notices: list):
        v = tags.get("cells")
        if v is None:
            return
        n = self.n
        N = n * n
        s = "".join(ch for ch in v if ch.isdigit() or ch in ".0_-")
        s = s.replace("0", ".").replace("_", ".").replace("-", ".")
        if len(s) != N or any(ch.isdigit() and not 1 <= int(ch) <= N for ch in s):
            notices.append(f"cells: {v[:40]!r} is not {N} cells of 1-{N} or '.'. your cells are unchanged.")
            return
        fixed = self.fixed(k)
        out = "".join(fx if fx != "." else ch for ch, fx in zip(s, fixed))
        if any(fx != "." and ch != fx for ch, fx in zip(s, fixed)):
            notices.append("cells: fixed digits cannot be changed; they were kept.")
        ctx.states[k].task["cells"] = out

    def after(self, ctx: Ctx):
        g = self.grid(ctx.states)
        N = self.n ** 2
        sol = self.data["solution"]
        right = sum(1 for r in range(N) for c in range(N) if g[r][c] == sol[r][c])
        bad = len(self.conflicts(g))
        filled = sum(1 for r in range(N) for c in range(N) if g[r][c])
        solved = filled == N * N and not bad
        st = self.state
        st.update(correct=right, conflicts=bad, filled=filled)
        if solved and not st.get("solved_at"):
            st["solved_at"] = ctx.t

    # ---------------------------------------------------------- reporting
    def report(self, meta, ticks):
        N = self.n ** 2
        hist = [r.get("task_state", {}).get("correct", 0) for r in ticks]
        last = ticks[-1]
        ts = last.get("task_state", {})
        givens = sum(1 for row in self.data["puzzle"] for d in row if d)
        out = [f"sudoku after tick {last['t']}: {ts.get('correct', 0)}/{N * N} cells right "
               f"({givens} were given), {ts.get('filled', 0)} filled, {ts.get('conflicts', 0)} in conflict"
               + (f", solved at t{ts['solved_at']}" if ts.get("solved_at") else ", not solved"),
               f"  correct cells over time: {spark(hist, N * N)}"]
        g = [[0] * N for _ in range(N)]
        n = self.n
        for a in last["agents"]:
            br, bc = self.box(a["agent"])
            cells = (a.get("task") or {}).get("cells") or self.fixed(a["agent"])
            for (r, c), ch in zip(box_cells(n, br, bc), cells):
                g[r][c] = ch
        sol = self.data["solution"]
        out.append("  final grid (lowercase x = wrong):")
        for r in range(N):
            row = ""
            for c in range(N):
                ch = g[r][c]
                row += (ch if ch == str(sol[r][c]) else ("." if ch == "." else "x")) + (" " if (c + 1) % n else "  ")
            out.append("    " + row)
            if (r + 1) % n == 0 and r + 1 < N:
                out.append("")
        return out

    def view_lines(self, k, rec):
        cells = (rec.get("task") or {}).get("cells") or self.fixed(k)
        n = self.n
        return [" ".join(cells[i * n:(i + 1) * n]) for i in range(n)]

    def echo_tags(self, ctx, k, rng):
        N = self.n ** 2
        if ctx.t % 3:
            return {}
        return {"cells": "".join(fx if fx != "." else rng.choice(".123456789"[:N + 1])
                                 for fx in self.fixed(k))}


TASK = Sudoku
