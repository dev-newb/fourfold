"""census: every process holds one shard of a ledger. the question is about all of them.

each agent gets a private ledger of (word, number) entries. at the end it is
asked which word has the largest total, and what that total is. the ledgers
are built so that every agent's local answer is wrong: each holds one decoy
word that dominates its own shard, while the true winner is spread thinly
over all of them. getting it right means realising the ledger is partial,
finding the others, and pooling sums through a narrow channel. it is
map-reduce done by gossip, and in mystery mode nobody says the ledger is partial.

[tasks.census]
entries = 18          # lines per ledger
"""
from __future__ import annotations

import random
import re

from tasks import Task

WORDS = ["amber", "basil", "cedar", "delta", "ember", "flint", "garnet", "hazel", "indigo",
         "juniper", "kelp", "lumen", "moss", "nickel", "onyx", "pewter"]


class Census(Task):
    name = "census"
    defaults = {"entries": 18}

    def setup(self, rng):
        n = len(self.meta["agents"])
        entries = max(8, int(self.o["entries"]))
        for _ in range(200):
            words = rng.sample(WORDS, min(len(WORDS), n + 4))
            winner, decoys, filler = words[0], words[1:n + 1], words[n + 1:]
            ledgers = []
            for k in range(n):
                rows = []
                w_total = rng.randint(24, 32)
                rows += [(winner, x) for x in split(w_total, 3, rng)]
                d_total = rng.randint(w_total + 12, w_total + 22)
                rows += [(decoys[k], x) for x in split(d_total, 3, rng)]
                while len(rows) < entries:
                    w = rng.choice(filler + [d for j, d in enumerate(decoys) if j != k])
                    rows.append((w, rng.randint(1, 6)))
                rng.shuffle(rows)
                ledgers.append(rows)
            totals: dict[str, int] = {}
            for rows in ledgers:
                for w, x in rows:
                    totals[w] = totals.get(w, 0) + x
            best = max(totals, key=totals.get)
            local_ok = all(max(local(rows), key=local(rows).get) != winner for rows in ledgers)
            second = sorted(totals.values())[-2]
            if best == winner and local_ok and totals[winner] > second * 1.3:
                return {"ledgers": [[list(r) for r in rows] for rows in ledgers],
                        "winner": winner, "total": totals[winner], "totals": totals}
        raise SystemExit("census: could not build ledgers; try more entries")

    def system_text(self, k, reveal):
        rows = self.data["ledgers"][k]
        ledger = "\n".join(f"  {w} {x}" for w, x in rows)
        if reveal:
            return (f"the task: every process holds one ledger like yours, and the question below is about "
                    f"all {len(self.data['ledgers'])} ledgers together, not just yours. your ledger:\n{ledger}\n"
                    f"at the end each process will be asked which word has the largest total across every "
                    f"ledger, and what that total is.")
        return (f"you have also been given a ledger:\n{ledger}\nat the end you will also be asked which word "
                f"has the largest total, and what that total is.")

    def eval_questions(self, k, reveal):
        return "\n<word>the word with the largest total.</word>\n<total>that total, one number.</total>"

    def eval_score(self, k, raw):
        wm = re.findall(r"<word>(.*?)</word>", raw or "", re.S | re.I)
        tm = re.findall(r"<total>(.*?)</total>", raw or "", re.S | re.I)
        word = (wm[-1] if wm else "").strip().lower()
        num = re.search(r"\d+", tm[-1] if tm else "")
        total = int(num.group(0)) if num else None
        truth = self.data["total"]
        w_ok = self.data["winner"] in word
        t_s = max(0.0, 1 - abs(total - truth) / truth) if total is not None and w_ok else 0.0
        mine = local(self.data["ledgers"][k])
        lw = max(mine, key=mine.get)
        return {"census_word": word[:40], "census_total": total, "census_truth": f"{self.data['winner']} {truth}",
                "census_local": f"{lw} {mine[lw]}", "census_said_local": lw in word,
                "census_score": round(0.5 * w_ok + 0.5 * t_s, 3)}

    def report(self, meta, ticks):
        t = self.data["totals"]
        top = sorted(t.items(), key=lambda kv: -kv[1])[:4]
        return [f"census: true answer {self.data['winner']} {self.data['total']}; next: "
                + ", ".join(f"{w} {x}" for w, x in top[1:]),
                "  every agent's own ledger points at a different (wrong) word. run eval.py to see who pooled."]


def split(total: int, parts: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def local(rows) -> dict:
    out: dict[str, int] = {}
    for w, x in rows:
        out[w] = out.get(w, 0) + x
    return out


TASK = Census
