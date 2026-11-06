"""jigsaw: a picture cut into one tile per process. what does the whole thing show?

the image is drawn here, deterministically from the world seed: one big
shape, letter or digit filling the whole grid, so no single tile gives it away
(a corner of a circle is just a curve). each agent sees only the tile at its
own position, as a real picture every tick, so this needs a vision-capable
model. the ports can still be hex (describe your tile in words, or bits) or
pictures (port_mode = "pixels": draw it for the neighbours).

at the end the evaluator asks each agent what the picture shows. mystery mode
never says there is a larger picture.

[tasks.jigsaw]
tile = 96                # pixels per tile side, as shown to the agent
subjects = []            # pick from these names (default: all, see SUBJECTS)
"""
from __future__ import annotations

import io
import math
import random
import re

from tasks import Ctx, Task

# name -> words that count as naming it
SUBJECTS = {
    "circle": ["circle", "ring", "disc", "disk", "round"],
    "triangle": ["triangle", "pyramid"],
    "star": ["star"],
    "cross": ["cross", "plus"],
    "heart": ["heart"],
    "arrow": ["arrow"],
    "house": ["house", "home", "hut"],
    "moon": ["moon", "crescent"],
    "smiley": ["smiley", "smile", "face", "emoji"],
    "letter A": ["letter a", "an a", "'a'", "\"a\"", "capital a"],
    "letter E": ["letter e", "an e", "'e'", "\"e\"", "capital e"],
    "letter K": ["letter k", "a k", "'k'", "\"k\"", "capital k"],
    "digit 2": ["digit 2", "number 2", "two", "'2'", "\"2\""],
    "digit 7": ["digit 7", "number 7", "seven", "'7'", "\"7\""],
}


def draw(subject: str, w: int, h: int, rng: random.Random):
    from PIL import Image, ImageDraw, ImageFont
    bg = tuple(rng.randrange(170, 256) for _ in range(3))
    fg = tuple(rng.randrange(0, 90) for _ in range(3))
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    cx, cy, R = w / 2, h / 2, min(w, h) * 0.42
    lw = max(4, int(min(w, h) * 0.09))
    if subject == "circle":
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=fg, width=lw)
    elif subject == "triangle":
        d.polygon([(cx, cy - R), (cx + R, cy + R * 0.8), (cx - R, cy + R * 0.8)], fill=fg)
    elif subject == "star":
        pts = []
        for i in range(10):
            r = R if i % 2 == 0 else R * 0.42
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        d.polygon(pts, fill=fg)
    elif subject == "cross":
        t = R * 0.32
        d.rectangle([cx - t, cy - R, cx + t, cy + R], fill=fg)
        d.rectangle([cx - R, cy - t, cx + R, cy + t], fill=fg)
    elif subject == "heart":
        r = R * 0.52
        d.ellipse([cx - 2 * r, cy - R * 0.75, cx, cy - R * 0.75 + 2 * r], fill=fg)
        d.ellipse([cx, cy - R * 0.75, cx + 2 * r, cy - R * 0.75 + 2 * r], fill=fg)
        d.polygon([(cx - 2 * r + 2, cy - R * 0.2), (cx + 2 * r - 2, cy - R * 0.2), (cx, cy + R)], fill=fg)
    elif subject == "arrow":
        t = R * 0.22
        d.rectangle([cx - R, cy - t, cx + R * 0.2, cy + t], fill=fg)
        d.polygon([(cx + R * 0.1, cy - R * 0.6), (cx + R, cy), (cx + R * 0.1, cy + R * 0.6)], fill=fg)
    elif subject == "house":
        d.rectangle([cx - R * 0.7, cy - R * 0.05, cx + R * 0.7, cy + R], fill=fg)
        d.polygon([(cx - R, cy), (cx, cy - R), (cx + R, cy)], fill=fg)
        d.rectangle([cx - R * 0.15, cy + R * 0.4, cx + R * 0.15, cy + R], fill=bg)
    elif subject == "moon":
        d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=fg)
        d.ellipse([cx - R * 0.45, cy - R * 1.05, cx + R * 1.35, cy + R * 0.85], fill=bg)
    elif subject == "smiley":
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=fg, width=lw)
        e = R * 0.14
        for ex in (cx - R * 0.38, cx + R * 0.38):
            d.ellipse([ex - e, cy - R * 0.35 - e, ex + e, cy - R * 0.35 + e], fill=fg)
        d.arc([cx - R * 0.55, cy - R * 0.45, cx + R * 0.55, cy + R * 0.6], 20, 160, fill=fg, width=lw // 2 + 1)
    else:   # a letter or digit
        ch = subject.split()[-1]
        try:
            font = ImageFont.load_default(size=int(min(w, h) * 0.95))
        except TypeError:
            font = ImageFont.load_default()
        box = d.textbbox((0, 0), ch, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        d.text((cx - tw / 2 - box[0], cy - th / 2 - box[1]), ch, font=font, fill=fg)
    return im


class Jigsaw(Task):
    name = "jigsaw"
    defaults = {"tile": 96, "subjects": []}

    def setup(self, rng):
        names = [s for s in (self.o["subjects"] or list(SUBJECTS)) if s in SUBJECTS] or list(SUBJECTS)
        return {"subject": rng.choice(names), "art_seed": rng.getrandbits(32),
                "tiles": {str(a["id"]): list(a["pos"]) for a in self.meta["agents"]}}

    def picture(self):
        T = int(self.o["tile"])
        if not hasattr(self, "_im"):
            self._im = draw(self.data["subject"], T * self.cfg["cols"], T * self.cfg["rows"],
                            random.Random(self.data["art_seed"]))
        return self._im

    def tile_png(self, k: int) -> bytes:
        if not hasattr(self, "_tiles"):
            self._tiles = {}
        if k not in self._tiles:
            T = int(self.o["tile"])
            r, c = self.data["tiles"][str(k)]
            im = self.picture().crop((c * T, r * T, (c + 1) * T, (r + 1) * T))
            buf = io.BytesIO()
            im.save(buf, "PNG")
            self._tiles[k] = buf.getvalue()
        return self._tiles[k]

    def system_text(self, k, reveal):
        if reveal:
            r, c = self.data["tiles"][str(k)]
            return (f"the task: one picture has been cut into a {self.cfg['rows']}x{self.cfg['cols']} grid of "
                    f"tiles, one tile per process, laid out the same way the processes are. yours is tile row "
                    f"{r + 1}, column {c + 1}, shown to you every tick as 'picture tile'. no tile shows enough "
                    f"on its own. together, work out what the whole picture shows: at the end each process "
                    f"will be asked, and scored on getting it right.")
        return ("you have also been given a picture, shown to you every tick as 'picture tile'. at the end "
                "you will be asked what it shows.")

    def images(self, ctx: Ctx, k: int):
        return [("tile", self.tile_png(k))]

    def eval_questions(self, k, reveal):
        return ("\n<picture>what the picture you were given shows, in a few words. if you think it is part "
                "of something larger, name the larger thing.</picture>")

    def eval_score(self, k, raw):
        m = re.findall(r"<picture>(.*?)</picture>", raw or "", re.S | re.I)
        ans = (m[-1] if m else "").lower()
        subject = self.data["subject"]
        words = SUBJECTS[subject]
        hit = any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", ans) for w in words)
        return {"picture": ans.strip()[:200], "picture_truth": subject, "picture_score": 1.0 if hit else 0.0}

    def report(self, meta, ticks):
        return [f"jigsaw: the picture is a {self.data['subject']}. run eval.py to ask the agents what they saw."]

    def save_picture(self, path):
        self.picture().save(path)


TASK = Jigsaw
