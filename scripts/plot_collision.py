"""Plot docs/collision.csv (from: collide 1500 docs/collision.csv).

usage: python scripts/plot_collision.py [csv] [output_png]
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/collision.csv")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/collision.png")

snaps = defaultdict(lambda: ([], [], None))
with open(src) as fh:
    for r in csv.DictReader(fh):
        xs, ys, _ = snaps[int(r["snapshot"])]
        xs.append(float(r["x"]))
        ys.append(float(r["y"]))
        snaps[int(r["snapshot"])] = (xs, ys, float(r["t"]))

fig, ax = plt.subplots(1, len(snaps), figsize=(16, 4.3), facecolor="black")
for a, (k, (xs, ys, t)) in zip(ax, sorted(snaps.items())):
    half = len(xs) // 2
    a.scatter(xs[:half], ys[:half], s=0.5, color="#ffb347", alpha=0.7)
    a.scatter(xs[half:], ys[half:], s=0.5, color="#7ec8ff", alpha=0.7)
    a.set(xlim=(-5, 5), ylim=(-5, 5), aspect="equal", xticks=[], yticks=[])
    a.set_facecolor("black")
    a.set_title(f"t = {t:g}", color="white")
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=100, facecolor="black")
print(f"wrote {out}")
