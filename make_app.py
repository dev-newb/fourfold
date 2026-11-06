"""build a double-clickable macOS app that opens the launcher.

  python make_app.py                  # -> ~/Applications/Fourfold.app
  python make_app.py --dest /Applications

the app is a thin wrapper: it runs launcher.py with this checkout's .venv, so
pulling new code needs no rebuild. the icon is drawn here: a 3x3 grid of
tiles with bit strips, in the launcher's four model colours.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import random
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLORS = [(90, 200, 255), (255, 150, 60), (225, 100, 225), (120, 225, 130)]


def icon(size: int):
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pad = size * 0.06
    d.rounded_rectangle([pad, pad, size - pad, size - pad], radius=size * 0.18, fill=(8, 8, 8, 255))
    rng = random.Random(4)
    g = (size - 2 * pad) * 0.07
    t = (size - 2 * pad - 4 * g) / 3
    for r in range(3):
        for c in range(3):
            x0, y0 = pad + g + c * (t + g), pad + g + r * (t + g)
            k = r * 3 + c
            col = COLORS[(k // 2) % 4] if k % 2 == 0 else (235, 235, 235)
            d.rectangle([x0, y0, x0 + t, y0 + t], fill=(14, 14, 14, 255))
            strip = max(1, t * 0.07)
            n = 12
            for i in range(n):
                if rng.random() < 0.55:
                    a, b = x0 + i * t / n, x0 + (i + 0.8) * t / n
                    d.rectangle([a, y0, b, y0 + strip], fill=col)
                if rng.random() < 0.55:
                    a, b = y0 + i * t / n, y0 + (i + 0.8) * t / n
                    d.rectangle([x0 + t - strip, a, x0 + t, b], fill=col)
                if rng.random() < 0.4:
                    a, b = x0 + i * t / n, x0 + (i + 0.8) * t / n
                    d.rectangle([a, y0 + t - strip, b, y0 + t], fill=(120, 120, 120))
    return im


def build_icns(dest: Path):
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "fourfold.iconset"
        iconset.mkdir()
        for s in (16, 32, 128, 256, 512):
            icon(s).save(iconset / f"icon_{s}x{s}.png")
            icon(s * 2).save(iconset / f"icon_{s}x{s}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(dest)], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(Path.home() / "Applications"))
    args = ap.parse_args()
    app = Path(args.dest) / "Fourfold.app"
    if app.exists():
        shutil.rmtree(app)
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "Resources").mkdir()
    py = ROOT / ".venv" / "bin" / "python"
    exe = app / "Contents" / "MacOS" / "Fourfold"
    exe.write_text(f"""#!/bin/bash
# opens the fourfold launcher from {ROOT}
cd "{ROOT}" || exit 1
PY="{py}"
[ -x "$PY" ] || PY="$(command -v python3)"
exec "$PY" launcher.py >> "{ROOT}/launches/app.log" 2>&1
""")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    build_icns(app / "Contents" / "Resources" / "fourfold.icns")
    with open(app / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump({
            "CFBundleName": "Fourfold",
            "CFBundleDisplayName": "Fourfold",
            "CFBundleIdentifier": "com.dev-newb.fourfold",
            "CFBundleExecutable": "Fourfold",
            "CFBundleIconFile": "fourfold",
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "1.0",
            "NSHighResolutionCapable": True,
        }, f)
    (ROOT / "launches").mkdir(exist_ok=True)
    os.utime(app)
    print(f"built {app}")


if __name__ == "__main__":
    main()
