#!/usr/bin/env python3
"""Check the compiled PDF for content outside the text block.

``Overfull \\hbox`` in the log catches too little. A table rule, a listing
frame or a figure can sit in the margin without producing one --- the frame on
the prompt listings hung 5.4pt into both margins at zero overfull boxes --- so
this measures every drawn element against the page geometry instead.

AAAI: \\textwidth 7.0in with \\oddsidemargin -0.25in on letter paper, so the
text block runs from 0.75in to 7.75in, i.e. 54pt to 558pt.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pymupdf

LEFT, RIGHT = 54.0, 558.0
TOLERANCE = 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", nargs="?", type=Path, default=Path("/tmp/cpo_build/paper_draft.pdf"))
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()

    doc = pymupdf.open(args.pdf)
    bad = []
    for index, page in enumerate(doc, start=1):
        items = [(s["bbox"], s["text"][:50], "text")
                 for block in page.get_text("dict")["blocks"]
                 for line in block.get("lines", [])
                 for s in line.get("spans", [])]
        items += [(tuple(d["rect"]), "", "rule") for d in page.get_drawings()]
        for (x0, _, x1, _), text, kind in items:
            over = max(x1 - RIGHT, LEFT - x0)
            if over > TOLERANCE:
                bad.append((round(over, 1), index, kind, round(x0, 1), round(x1, 1), text))

    log = args.log or args.pdf.with_suffix(".log")
    counts = {}
    if log.exists():
        body = log.read_text(errors="replace")
        for key in ("Overfull \\hbox", "Overfull \\vbox", "Missing character"):
            counts[key] = body.count(key)
        # A dropped glyph leaves only this line in the log and a gap in the PDF.
        for m in set(re.findall(r"Missing character: There is no (.) .*? in font (\S+)!", body)):
            print(f"  ! dropped glyph {m[0]!r} in font {m[1]}")

    print(f"pages: {len(doc)}")
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"elements outside the text block: {len(bad)}")
    for row in sorted(bad, reverse=True)[:20]:
        print(f"  p{row[1]:>2} {row[2]:5s} over {row[0]:6.1f}pt x=[{row[3]},{row[4]}] {row[5]}")

    failed = bool(bad) or counts.get("Overfull \\hbox", 0) or counts.get("Missing character", 0)
    print("FAIL" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
