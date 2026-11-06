"""tile renderers: agent state -> image of one tile.

the viewer only talks to the TileRenderer interface, so a different renderer
(e.g. a 128x128 led panel) can be dropped in by implementing render().

each live side has two strips of 128 segments: the outer one is the agent's
output on that side, the inner (dimmer) one is the input arriving there.

bit order on edges: bit i is the i-th bit of the hex string read left to
right (msb of the first hex char is bit 0). north and south edges run left to
right, east and west run top to bottom, so two facing edges across a gap line
up segment for segment.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

import pygame

SIDES = ("N", "E", "S", "W")

TILE_BG = (6, 6, 6)
FG = (232, 232, 232)
DIM = (125, 125, 125)
FAINT = (55, 55, 55)
PROTO_FG = (190, 190, 190)
BIT_ON = (242, 242, 242)  # output strip
BIT_IN = (140, 140, 140)  # input strip, dimmer so the two read apart
BIT_OFF = (24, 24, 24)
DEAD = (70, 70, 70)
HI_ON = (255, 176, 48)    # debug: bit turned on this tick
HI_OFF = (120, 42, 30)    # debug: bit turned off this tick
ERR = (215, 95, 95)
MODEL_COLORS = [(90, 200, 255), (255, 150, 60), (225, 100, 225), (120, 225, 130)]  # slot 1-4 in the launcher

MONO = "consolas,dejavusansmono,menlo,cascadiamono,couriernew,monospace"


@dataclass
class TileView:
    label: str
    seed: str
    tick: int
    sides: dict                       # side -> output hex on that side, None if dead
    side_ports: dict                  # side -> port index facing it
    log: list                         # (tick, kind, text), kind "log" | "error", oldest first
    protocol: str = ""
    prev_sides: dict | None = None    # previous tick's sides, for debug highlighting
    inputs: dict = field(default_factory=dict)  # side -> input hex arriving on that side
    prev_inputs: dict | None = None
    echo: dict = field(default_factory=dict)  # side -> (now, avg10), either may be None
    show_protocol: bool = False
    debug: bool = False
    pixel: int = 0                    # pixels mode: image side length. 0 = bit strips
    score: int | None = None          # naming task: this tick's score
    score_max: int | None = None      # naming task: live ports, the most this agent can score
    extra: list = field(default_factory=list)   # task / energy / lever lines under the header
    accent: tuple | None = None       # mixed-model runs: the model's colour


class TileRenderer:
    def render(self, tile: TileView, size: tuple[int, int]) -> pygame.Surface:
        raise NotImplementedError


def bits(hexstr: str) -> list[int]:
    v = int(hexstr, 16)
    n = len(hexstr) * 4
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def wrap_block(text: str, cols: int) -> list[str]:
    out = []
    for para in text.expandtabs(2).splitlines() or [""]:
        out.extend(textwrap.wrap(para, cols, break_long_words=True, drop_whitespace=True) or [""])
    return out


def pad_accent(r) -> int:
    return 7


class ScreenRenderer(TileRenderer):
    """the default: renders at whatever pixel size it's asked for."""

    def __init__(self):
        self._fonts: dict = {}

    def font(self, px: int, bold: bool = False) -> pygame.font.Font:
        key = (px, bold)
        if key not in self._fonts:
            self._fonts[key] = pygame.font.SysFont(MONO, px, bold=bold)
        return self._fonts[key]

    # -------------------------------------------------------------- layout

    def render(self, tile: TileView, size: tuple[int, int]) -> pygame.Surface:
        w, h = max(48, int(size[0])), max(48, int(size[1]))
        s = pygame.Surface((w, h))
        s.fill(TILE_BG)
        u = min(w, h)
        fs = max(9, min(22, u // 24))
        strip = max(2, round(u * 0.014))
        sgap = max(1, strip // 3)
        e = 2 * strip + sgap
        if tile.pixel:  # one picture deep: output and input sit side by side along the edge
            e = max(tile.pixel, round(u * 0.13))
        font, small, bold = self.font(fs), self.font(max(8, fs - 2)), self.font(fs, True)

        for side in SIDES:
            self._edge(s, side, tile, self._edge_rect(side, w, h, e), strip, sgap)

        pad = max(4, fs // 2)
        inset = e + pad
        if tile.debug:
            inset = self._debug(s, tile, w, h, e, small) + pad
        inner = pygame.Rect(inset, inset, w - 2 * inset, h - 2 * inset)
        if inner.w > 30 and inner.h > 3 * font.get_linesize():
            self._interior(s, inner, tile, font, bold)
        return s

    @staticmethod
    def _edge_rect(side, w, h, e) -> pygame.Rect:
        return {
            "N": pygame.Rect(e, 0, w - 2 * e, e),
            "S": pygame.Rect(e, h - e, w - 2 * e, e),
            "W": pygame.Rect(0, e, e, h - 2 * e),
            "E": pygame.Rect(w - e, e, e, h - 2 * e),
        }[side]

    # -------------------------------------------------------------- edges

    def _edge(self, s, side, tile: TileView, rect: pygame.Rect, strip: int, sgap: int):
        """outer strip: what this agent writes out on that side. inner strip:
        what arrives on it."""
        horiz = side in "NS"
        val = tile.sides.get(side)
        inp = tile.inputs.get(side)
        if val is None and inp is not None:   # a one-way wire, or an input pin: inbound only
            val = "0" * len(inp)
        if val is None:
            dash = max(3, min(rect.w, rect.h) * 2)
            if horiz:
                y = rect.centery
                for x in range(rect.x, rect.right, dash * 2):
                    pygame.draw.line(s, DEAD, (x, y), (min(x + dash, rect.right), y))
            else:
                x = rect.centerx
                for y in range(rect.y, rect.bottom, dash * 2):
                    pygame.draw.line(s, DEAD, (x, y), (x, min(y + dash, rect.bottom)))
            return
        if tile.pixel:
            self._pictures(s, side, tile, rect, val)
            return
        if side == "N":
            outer = pygame.Rect(rect.x, rect.y, rect.w, strip)
            inner = pygame.Rect(rect.x, rect.y + strip + sgap, rect.w, strip)
        elif side == "S":
            outer = pygame.Rect(rect.x, rect.bottom - strip, rect.w, strip)
            inner = pygame.Rect(rect.x, rect.y, rect.w, strip)
        elif side == "W":
            outer = pygame.Rect(rect.x, rect.y, strip, rect.h)
            inner = pygame.Rect(rect.x + strip + sgap, rect.y, strip, rect.h)
        else:
            outer = pygame.Rect(rect.right - strip, rect.y, strip, rect.h)
            inner = pygame.Rect(rect.x, rect.y, strip, rect.h)
        prev_out = (tile.prev_sides or {}).get(side) if tile.debug else None
        prev_in = (tile.prev_inputs or {}).get(side) if tile.debug else None
        self._bits(s, outer, horiz, val, prev_out, BIT_ON)
        self._bits(s, inner, horiz, tile.inputs.get(side) or "0" * len(val), prev_in, BIT_IN)

    @staticmethod
    def _picture(v: str, k: int, px: int) -> pygame.Surface:
        img = pygame.image.frombuffer(bytes.fromhex(v), (k, k), "RGB")
        return pygame.transform.scale(img, (px, px))

    def _pictures(self, s, side, tile: TileView, rect: pygame.Rect, val: str):
        """output picture and input picture next to each other along the edge. n and e put
        the output first, s and w the input first, so across a gap each output faces the
        neighbour's matching input."""
        px = min(rect.w, rect.h) - 2
        k = tile.pixel
        out_img = self._picture(val, k, px)
        inp = tile.inputs.get(side) or "0" * len(val)
        in_img = self._picture(inp, k, px)
        horiz = side in "NS"
        length = rect.w if horiz else rect.h
        gap = max(4, px // 6)
        start = (length - 2 * px - gap) // 2
        order = [(out_img, val, (tile.prev_sides or {}).get(side), FG),
                 (in_img, inp, (tile.prev_inputs or {}).get(side), DIM)]
        if side in "SW":
            order.reverse()
        for n, (img, v, prev, edge_col) in enumerate(order):
            off = start + n * (px + gap)
            pos = (rect.x + off, rect.y + 1) if horiz else (rect.x + 1, rect.y + off)
            s.blit(img, pos)
            col = HI_ON if (tile.debug and prev and prev != v) else edge_col
            pygame.draw.rect(s, col, pygame.Rect(pos[0] - 1, pos[1] - 1, px + 2, px + 2), 1)

    @staticmethod
    def _bits(s, rect: pygame.Rect, horiz: bool, val: str, prev_val: str | None, on_col):
        cur = bits(val)
        prev = bits(prev_val) if prev_val else None
        n = len(cur)
        length = rect.w if horiz else rect.h
        gapped = length / n >= 4
        for i in range(n):
            a, b = int(i * length / n), int((i + 1) * length / n)
            if gapped:
                b -= 1
            b = max(b, a + 1)
            col = on_col if cur[i] else BIT_OFF
            if prev is not None and prev[i] != cur[i]:
                col = HI_ON if cur[i] else HI_OFF
            r = (pygame.Rect(rect.x + a, rect.y, b - a, rect.h) if horiz
                 else pygame.Rect(rect.x, rect.y + a, rect.w, b - a))
            s.fill(col, r)

    # -------------------------------------------------------------- debug

    def _debug(self, s, tile: TileView, w, h, e, font) -> int:
        """per-edge port, echo score and hex. returns the inset it used."""
        lh = font.get_linesize()
        cw = font.size("0")[0]
        gap = 2
        for side in SIDES:
            port = tile.side_ports.get(side)
            val = tile.sides.get(side)
            if val is None and tile.inputs.get(side) is not None:
                head = f"port {port}  in only"
                val = "0" * len(tile.inputs[side])
            if val is None:
                head = f"port {port}  dead"
            else:
                now, avg = tile.echo.get(side, (None, None))
                fmt = lambda x: "  -" if x is None else f"{x:.2f}"
                head = f"port {port}  echo {fmt(now)}  avg {fmt(avg)}"
            nl = 1 if (val is None or tile.pixel) else 3
            block = pygame.Surface((max(font.size(head)[0], 35 * cw) + 2, nl * lh), pygame.SRCALPHA)
            block.blit(font.render(head, True, DIM), (0, 0))
            if val is not None and not tile.pixel:
                rows = [(f"q{port}", val, (tile.prev_sides or {}).get(side)),
                        (f"p{port}", tile.inputs.get(side) or "0" * len(val), (tile.prev_inputs or {}).get(side))]
                for r, (name, hx, prev) in enumerate(rows, start=1):
                    block.blit(font.render(name, True, DIM), (0, r * lh))
                    for i, ch in enumerate(hx):
                        col = HI_ON if prev and prev[i] != ch else PROTO_FG
                        block.blit(font.render(ch, True, col), ((i + 3) * cw, r * lh))
            band = gap * 2 + 3 * lh   # keep corners free so neighbouring labels don't collide
            if side == "N":
                s.blit(block, (e + band, e + gap))
            elif side == "S":
                s.blit(block, (e + band, h - e - gap - block.get_height()))
            elif side == "W":
                rb = pygame.transform.rotate(block, 90)
                s.blit(rb, (e + gap, h - e - band - rb.get_height()))
            else:
                rb = pygame.transform.rotate(block, -90)
                s.blit(rb, (w - e - gap - rb.get_width(), e + band))
        return e + gap + 3 * lh

    # -------------------------------------------------------------- interior

    def _interior(self, s, r: pygame.Rect, tile: TileView, font, bold):
        lh = font.get_linesize()
        cw = font.size("M")[0]
        cols = max(8, r.w // cw)
        y = r.y

        lab = bold.render(tile.label, True, tile.accent or FG)
        s.blit(lab, (r.x, y))
        if tile.accent:
            pygame.draw.rect(s, tile.accent, pygame.Rect(r.x - pad_accent(r), r.y, 3, lh))
        tk = font.render(f"tick {tile.tick}", True, DIM)
        s.blit(tk, (r.right - tk.get_width(), y))
        sd = font.render(f"seed {tile.seed}", True, DIM)
        if tile.score is not None:
            col = HI_ON if tile.score and tile.score == tile.score_max else (FG if tile.score else DIM)
            sd = font.render(f"score {tile.score}/{tile.score_max}", True, col)
        sx = r.x + lab.get_width() + 2 * cw
        if sx + sd.get_width() + 2 * cw < r.right - tk.get_width():
            s.blit(sd, (sx, y))
        y += lh + 3
        for line in tile.extra[: max(0, (r.h // lh) // 3)]:
            s.blit(font.render(line[:cols], True, HI_ON if "**" in line else PROTO_FG), (r.x, y))
            y += lh
        pygame.draw.line(s, FAINT, (r.x, y), (r.right, y))
        y += 4

        # protocol, pinned to the bottom
        words = len(tile.protocol.split())
        title = f"{'-' if tile.show_protocol else '+'} protocol  {words} words"
        plines = []
        if tile.show_protocol:
            plines = wrap_block(tile.protocol or "(empty)", cols)
            room = max(1, int((r.bottom - y) * 0.6) // lh - 1)
            if len(plines) > room:
                plines = plines[: room - 1] + ["..."]
        ph = (len(plines) + 1) * lh + 5
        py = r.bottom - ph
        pygame.draw.line(s, FAINT, (r.x, py), (r.right, py))
        s.blit(font.render(title, True, DIM), (r.x, py + 4))
        for i, line in enumerate(plines):
            s.blit(font.render(line, True, PROTO_FG), (r.x, py + 4 + (i + 1) * lh))

        # log, newest at the bottom
        bottom = py - 4
        avail = max(0, (bottom - y) // lh)
        if not tile.log:
            s.blit(font.render("(no log yet)", True, FAINT), (r.x, bottom - lh))
            return
        rows, pw = self._log_rows(tile.log, cols, avail)
        ly = bottom - len(rows) * lh
        for prefix, text, col in rows:
            if prefix:
                s.blit(font.render(prefix, True, DIM), (r.x, ly))
            if text:
                s.blit(font.render(text, True, col), (r.x + pw * cw, ly))
            ly += lh

    @staticmethod
    def _log_rows(log, cols, avail):
        recent = log[-(avail * 2 + 4):]
        pw = len(f"t{max(t for t, _, _ in recent)} ")
        rows, last_t = [], None
        for t, kind, text in recent:
            col = ERR if kind == "error" else FG
            if kind == "error":
                text = "! " + text
            for n, piece in enumerate(wrap_block(text, max(4, cols - pw))):
                rows.append((f"t{t}" if (t != last_t and n == 0) else "", piece, col))
            last_t = t
        return rows[-avail:] if avail else [], pw
