#!/usr/bin/env python3
"""Generate the hnsw crate logo.

The mark is the algorithm itself: a Hierarchical Navigable Small World graph is a
stack of layers, sparse at the top and dense at the bottom, with a search that
enters at the top and descends toward its target. That descent is drawn as a
highlighted path, because it is the thing the data structure exists to make fast.

Deterministic: the layout is seeded, so re-running produces a byte-identical PNG.

    uv run --with pillow scripts/generate_logo.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

# Rendered large and downscaled, which is cheaper than real antialiasing and
# gives smooth edges at the 200px width the READMEs use.
SUPERSAMPLE = 4
SIZE = 1000
W = H = SIZE * SUPERSAMPLE

# Rust-adjacent palette: warm accent for the active search path, cool slate for
# the graph itself, so the descent reads instantly against the structure.
BG = (0, 0, 0, 0)
NODE = (94, 110, 133, 255)
EDGE = (94, 110, 133, 110)
PATH_NODE = (222, 118, 58, 255)
PATH_EDGE = (222, 118, 58, 235)
TARGET = (240, 176, 84, 255)
LAYER_PLATE = (94, 110, 133, 26)

# Layer 0 is the bottom (dense); the last entry is the top (sparse), matching
# how the paper draws it.
LAYERS = [
    {"count": 22, "y": 0.80, "r": 0.0125, "spread": 0.86},
    {"count": 11, "y": 0.545, "r": 0.0155, "spread": 0.70},
    {"count": 5, "y": 0.295, "r": 0.019, "spread": 0.52},
    {"count": 2, "y": 0.075, "r": 0.023, "spread": 0.26},
]


def px(v: float) -> float:
    return v * W


def layer_nodes(rng: random.Random, layer: dict) -> list[tuple[float, float]]:
    """Evenly spaced along the layer with a small deterministic jitter, so the
    graph looks organic rather than like a lattice."""
    count = layer["count"]
    spread = layer["spread"]
    y = layer["y"]
    nodes = []
    for i in range(count):
        t = (i + 0.5) / count
        x = 0.5 + (t - 0.5) * spread
        jitter_x = (rng.random() - 0.5) * (spread / count) * 0.55
        jitter_y = (rng.random() - 0.5) * 0.018
        nodes.append((x + jitter_x, y + jitter_y))
    return nodes


def draw_disc(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, color) -> None:
    d.ellipse([px(cx - r), px(cy - r), px(cx + r), px(cy + r)], fill=color)


def draw_edge(d: ImageDraw.ImageDraw, a, b, color, width: float) -> None:
    d.line([px(a[0]), px(a[1]), px(b[0]), px(b[1])], fill=color, width=int(px(width)))


def nearest(nodes: list[tuple[float, float]], point) -> int:
    return min(
        range(len(nodes)),
        key=lambda i: (nodes[i][0] - point[0]) ** 2 + (nodes[i][1] - point[1]) ** 2,
    )


def main() -> None:
    rng = random.Random(0x48_4E_53_57)  # "HNSW"
    img = Image.new("RGBA", (W, H), BG)
    d = ImageDraw.Draw(img)

    layers = [layer_nodes(rng, spec) for spec in LAYERS]

    # Faint plates behind each layer to read as stacked planes.
    for spec in LAYERS:
        half = spec["spread"] / 2 + 0.06
        y = spec["y"]
        d.rounded_rectangle(
            [px(0.5 - half), px(y - 0.052), px(0.5 + half), px(y + 0.052)],
            radius=px(0.052),
            fill=LAYER_PLATE,
        )

    # The target sits in the dense bottom layer, off to one side so the descent
    # has somewhere to travel.
    target_idx = int(len(layers[0]) * 0.72)
    target = layers[0][target_idx]

    # The search path: enter at the top layer's first node, then at each level
    # step to the node nearest the target. This is greedy descent, which is what
    # the algorithm actually does.
    path: list[tuple[int, int]] = [(len(layers) - 1, 0)]
    for level in range(len(layers) - 2, -1, -1):
        path.append((level, nearest(layers[level], target)))
    path_set = set(path)

    # Intra-layer edges: connect each node to its neighbours, denser lower down.
    for level, nodes in enumerate(layers):
        reach = 2 if level == 0 else 1
        for i in range(len(nodes)):
            for j in range(i + 1, min(i + 1 + reach, len(nodes))):
                on_path = (level, i) in path_set and (level, j) in path_set
                draw_edge(
                    d,
                    nodes[i],
                    nodes[j],
                    PATH_EDGE if on_path else EDGE,
                    0.0045 if on_path else 0.0028,
                )

    # Inter-layer links: every node in an upper layer drops to its nearest node
    # in the layer below, which is the "same element, lower level" relation.
    for level in range(len(layers) - 1, 0, -1):
        for i, node in enumerate(layers[level]):
            below = nearest(layers[level - 1], node)
            on_path = (level, i) in path_set and (level - 1, below) in path_set
            draw_edge(
                d,
                node,
                layers[level - 1][below],
                PATH_EDGE if on_path else EDGE,
                0.0055 if on_path else 0.0022,
            )

    # The descent itself, drawn over the graph so it stays legible.
    for (la, ia), (lb, ib) in zip(path, path[1:]):
        draw_edge(d, layers[la][ia], layers[lb][ib], PATH_EDGE, 0.0075)

    # Nodes last, so edges tuck underneath.
    for level, nodes in enumerate(layers):
        r = LAYERS[level]["r"]
        for i, node in enumerate(nodes):
            if (level, i) in path_set:
                draw_disc(d, node[0], node[1], r * 1.5, PATH_NODE)
            else:
                draw_disc(d, node[0], node[1], r, NODE)

    # Target: a filled disc with a halo ring, the one thing the eye lands on.
    tr = LAYERS[0]["r"]
    d.ellipse(
        [
            px(target[0] - tr * 2.9),
            px(target[1] - tr * 2.9),
            px(target[0] + tr * 2.9),
            px(target[1] + tr * 2.9),
        ],
        outline=TARGET,
        width=int(px(0.0055)),
    )
    draw_disc(d, target[0], target[1], tr * 1.75, TARGET)

    out = Image.new("RGBA", (SIZE, SIZE), BG)
    out.paste(img.resize((SIZE, SIZE), Image.LANCZOS), (0, 0))

    dest = Path(__file__).resolve().parent.parent / "hnsw-logo.png"
    out.save(dest, "PNG", optimize=True)

    alpha = out.getchannel("A")
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
    print(f"size: {out.size}, mode: {out.mode}")
    print(f"alpha range: min={alpha.getextrema()[0]} max={alpha.getextrema()[1]}")
    print(f"layers: {[spec['count'] for spec in LAYERS]} (bottom to top)")
    print(f"search path: {path}")


if __name__ == "__main__":
    main()
