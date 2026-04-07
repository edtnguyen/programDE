#!/usr/bin/env python3
"""Stitch QQ plot PNGs into a multi-page PDF.

Expected input filenames (from scripts/ipsc_ec_qq_sweep.py):
  qq_{day}.k_{K}.dt_{DT}.png

Output:
- One PDF page per (k, day)
- Within each page: different dt values shown side-by-side
- Page order: all days for k=smallest, then next k (k asc, day asc)

Example:
  python scripts/qq_pngs_to_pdf.py \
    --in-dir results/ipsc_ec_qq \
    --out-pdf results/ipsc_ec_qq/qq_by_k_day.pdf
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # safe on clusters

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PAT = re.compile(
    r"^qq_(?P<day>d\d+)\.k_(?P<k>\d+)\.dt_(?P<dt>[0-9_]+)\.png$",
    flags=re.IGNORECASE,
)


def _day_num(day: str) -> int:
    m = re.match(r"^d(\d+)$", day.strip().lower())
    if m:
        return int(m.group(1))
    return 10**9


def _dt_key(dt: str) -> Tuple[int, ...]:
    # dt strings look like "0_2"; sort numerically, fallback to large key.
    try:
        return tuple(int(x) for x in dt.split("_"))
    except Exception:
        return (10**9,)


def _load_png(path: Path):
    # Avoid Pillow dependency explicit import; matplotlib will use what's available.
    return plt.imread(str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-pdf", type=Path, required=True)
    ap.add_argument(
        "--glob",
        default="qq_*.png",
        help="Glob pattern under --in-dir to include (default: qq_*.png).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI used when writing PDF (default: 200).",
    )
    ap.add_argument(
        "--panel-width",
        type=float,
        default=6.0,
        help="Width (inches) per dt panel (default: 6.0).",
    )
    ap.add_argument(
        "--panel-height",
        type=float,
        default=6.0,
        help="Height (inches) per page (default: 6.0).",
    )
    args = ap.parse_args()

    groups: Dict[Tuple[int, str], Dict[str, Path]] = defaultdict(dict)  # (k, day) -> dt -> path

    for p in sorted(args.in_dir.glob(args.glob)):
        m = PAT.match(p.name)
        if not m:
            continue
        day = m.group("day").lower()
        k = int(m.group("k"))
        dt = m.group("dt")
        groups[(k, day)][dt] = p

    if not groups:
        raise SystemExit(
            f"No matching files found under {args.in_dir} with glob '{args.glob}'. "
            "Expected names like qq_d0.k_100.dt_0_2.png"
        )

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Order: k asc, day asc
    ordered_keys: List[Tuple[int, str]] = sorted(groups.keys(), key=lambda x: (x[0], _day_num(x[1]), x[1]))

    with PdfPages(args.out_pdf) as pdf:
        for k, day in ordered_keys:
            dt_map = groups[(k, day)]
            dts = sorted(dt_map.keys(), key=_dt_key)
            n = len(dts)

            fig, axes = plt.subplots(1, n, figsize=(args.panel_width * n, args.panel_height))
            if n == 1:
                axes = [axes]

            for ax, dt in zip(axes, dts):
                img = _load_png(dt_map[dt])
                ax.imshow(img)
                ax.set_title(f"dt={dt}")
                ax.axis("off")

            fig.suptitle(f"{day} | k={k}", y=0.98)
            fig.tight_layout()
            pdf.savefig(fig, dpi=args.dpi)
            plt.close(fig)

    print(f"[saved] {args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

