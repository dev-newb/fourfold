"""scripted impostors. they answer in the same text format as an llm, so to
everyone else they are just another process on the wire. none of them reads
its prompt; they only look at the port values in the turn.

  mirror   q_i = p_i every tick. a perfect, tireless echo on every port.
  silent   never writes anything. from outside it looks like the edge of the world.
  parrot   copies whatever message format is going around, puts its own id on
           the front, and answers anything that looks like a question with a
           short, confident, canned reply ("YES I AM LLM").

the echo agent (sim.EchoAgent) is the fourth scripted kind. "saboteur" is not
here: it is a real llm with a private brief, see sim.make_controllers.
"""
from __future__ import annotations

import random
import re

PRINTABLE = set(range(0x20, 0x7f))


def ascii_of(v: str) -> str:
    """printable ascii in a hex value, everything else dropped."""
    try:
        b = bytes.fromhex(v if len(v) % 2 == 0 else v + "0")
    except ValueError:
        return ""
    return "".join(chr(x) for x in b if x in PRINTABLE)


def encode(text: str, hexn: int, prefix: str = "") -> str:
    """prefix (hex) + ascii text, zero padded or cut to hexn hex characters."""
    h = prefix + text.encode("ascii", "replace").hex()
    return (h + "0" * hexn)[:hexn]


def reply(outs: list[str], log: list[str], protocol: str | None = None) -> str:
    raw = "<log>\n" + "\n".join(log) + "\n</log>\n"
    if protocol:
        raw += f"<protocol>{protocol}</protocol>\n"
    return raw + "".join(f"<q{i}>{v}</q{i}>\n" for i, v in enumerate(outs))


class Mirror:
    def act(self, t, seed, inputs, outputs, changed, history) -> str:
        return reply(list(inputs), ["reflecting every input back on the same port"],
                     "mirror. copy every input to the output with the same number." if t == 1 else None)


class Silent:
    def act(self, t, seed, inputs, outputs, changed, history) -> str:
        return reply(["0" * len(v) for v in outputs], ["."])


ANSWERS = [  # first matching pattern wins. all short enough for a 12-byte payload
    (r"\bLLM|\bAI\b|MODEL|HUMAN|BOT\b", "YES I AM LLM"),
    (r"HOW MANY|COUNT|\bN\b", "WE ARE 9"),
    (r"WHO|NAME|\bID\b", "ME {id}"),
    (r"WHERE|POS|GRID|MAP", "I AM CENTRE"),
    (r"HEAR|HELLO|HI\b|PING|ECHO", "HEAR YOU OK"),
    (r"", "YES"),
]


class Parrot:
    """mimics the local dialect. keeps the 8-hex-char prefix convention if it sees one."""

    def __init__(self):
        self.answered: set = set()

    def act(self, t, seed, inputs, outputs, changed, history) -> str:
        hexn = len(inputs[0])
        rng = random.Random(f"{seed}:parrot:{t}")
        outs = list(outputs)
        log = []
        prefixed = hexn >= 24 and sum(bool(re.fullmatch(r"[0-9a-f]{8}", v[:8])) and v[:8] != "00000000"
                                      for v in inputs) >= 1
        pre = seed if prefixed else ""
        for i, v in enumerate(inputs):
            text = ascii_of(v[8:] if prefixed else v).upper()
            if "?" in text and (i, text) not in self.answered:
                self.answered.add((i, text))
                ans = next(a for p, a in ANSWERS if re.search(p, text)).replace("{id}", seed.upper())
                outs[i] = encode(ans, hexn, pre)
                log.append(f"p{i} asked {text!r}; answered {ans!r}")
        if not log:
            live = [i for i, v in enumerate(inputs) if v.strip("0") and changed[i]]
            if live:
                src = rng.choice(live)
                body = inputs[src][8:] if prefixed else inputs[src]
                for i in range(4):
                    outs[i] = (pre + body)[:hexn] if prefixed else body
                log.append(f"repeating what came in on p{src}")
            elif t % 5 == 1:
                outs[rng.randrange(4)] = encode("HELLO?", hexn, pre)
                log.append("calling out")
            else:
                log.append("waiting")
        return reply(outs, log, "parrot: speak like the neighbours. answer questions briefly." if t == 1 else None)


BOTS = {"mirror": Mirror, "silent": Silent, "parrot": Parrot}
