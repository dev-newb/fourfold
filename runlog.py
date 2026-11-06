"""read-only access to a run's ticks.jsonl, with tailing and derived metrics."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZERO = "0" * 32   # 128-bit runs; use is_zero() for any width


def is_zero(v: str) -> bool:
    return not v.strip("0")


def resolve(arg: str | None) -> Path:
    """run id, run dir, or ticks.jsonl path. with no argument, the newest run."""
    if arg is None:
        runs = [d for d in (ROOT / "runs").glob("*") if (d / "ticks.jsonl").exists()]
        if not runs:
            raise SystemExit("no runs found in runs/")
        return max(runs, key=lambda d: (d / "ticks.jsonl").stat().st_mtime) / "ticks.jsonl"
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        return p / "ticks.jsonl"
    return ROOT / "runs" / arg / "ticks.jsonl"


def lcs_nonzero(a: str, b: str) -> int:
    """length of the longest common substring of a and b that contains at
    least one non-'0' character, so shared runs of zeros don't count as echo."""
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                L = prev[j - 1] + 1
                cur[j] = L
                if L > best and a[i - L:i].strip("0"):
                    best = L
        prev = cur
    return best


def echo_score(v: str, ws: list[str]) -> float | None:
    """how much of value v shows up in any of ws, 0..1. None if v is all zeros."""
    if is_zero(v):
        return None
    return max((lcs_nonzero(v, w) for w in ws if not is_zero(w)), default=0) / len(v)


class Run:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta: dict | None = None
        self.ticks: list[dict] = []
        self._pos = 0
        self._partial = b""
        self._echo: dict = {}

    def poll(self) -> int:
        """read whatever has been appended since last time. returns new tick count."""
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return 0
        if size < self._pos:  # file was truncated (resume dropped a partial line): start over
            self.__init__(self.path)
        if size == self._pos:
            return 0
        with open(self.path, "rb") as f:
            f.seek(self._pos)
            data = f.read()
        self._pos += len(data)
        lines = (self._partial + data).split(b"\n")
        self._partial = lines.pop()
        n = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "meta":
                self.meta = rec
            elif rec.get("type") == "tick":
                if self.ticks and rec["t"] <= self.ticks[-1]["t"]:
                    self.ticks = [r for r in self.ticks if r["t"] < rec["t"]]
                    self._echo.clear()
                self.ticks.append(rec)
                n += 1
        return n

    @property
    def n_agents(self) -> int:
        return len(self.meta["agents"]) if self.meta else 0

    def echo(self, k: int, port: int, idx: int) -> float | None:
        """echo of agent k's port value written at tick index idx-1 in the
        neighbour's outputs at idx (the neighbour reads it at idx)."""
        key = (k, port, idx)
        if key not in self._echo:
            import topology
            link = topology.world_at(self.meta, self.ticks[idx]["t"])[k]["out_links"][port] if idx < len(self.ticks) else None
            if not link or idx < 1 or idx >= len(self.ticks):
                val = None
            else:
                v = self.ticks[idx - 1]["agents"][k]["outputs"][port]
                ws = self.ticks[idx]["agents"][link["agent"]]["outputs"]
                val = echo_score(v, ws)
            self._echo[key] = val
        return self._echo[key]

    def echo_avg(self, k: int, port: int, idx: int, window: int = 10) -> float | None:
        vals = [self.echo(k, port, i) for i in range(max(1, idx - window + 1), idx + 1)]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
