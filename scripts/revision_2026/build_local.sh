#!/bin/zsh
# Compile the manuscript locally with tectonic, for typesetting checks only.
#
# Overleaf builds with pdfLaTeX; tectonic is XeTeX. Three differences have to
# be patched or the local build measures the wrong thing, so the manuscript is
# copied and patched rather than edited:
#
#   \pdfinfo, \pdfpagewidth   pdfTeX primitives; XeTeX has neither. aaai2026.sty
#                             only stubs \pdfinfo under its `submission` option,
#                             which this paper does not use.
#   font encoding             XeTeX defaults to TU, for which no TUptm.fd exists,
#                             so times.sty and courier.sty fall back to Latin
#                             Modern without an error and every measured width is
#                             wrong. Forcing T1 loads the same Type1 Nimbus
#                             Roman / Nimbus Mono metrics pdfLaTeX uses.
#
# Install once:  conda install -c conda-forge tectonic
# Check after:   grep -c 'Overfull \\hbox' /tmp/cpo_build/paper_draft.log
#                and check_layout.py for content outside the text block.
set -e
SRC="${1:-$(cd "$(dirname "$0")/../../../overleaf-paper" && pwd)}"
WORK=${WORK:-/tmp/cpo_src}
OUT=${OUT:-/tmp/cpo_build}
TECTONIC=${TECTONIC:-/opt/anaconda3/bin/tectonic}
PYTHON=${PYTHON:-/opt/anaconda3/bin/python}

rm -rf "$WORK"; mkdir -p "$WORK" "$OUT"
/usr/bin/rsync -a --exclude '.git' --exclude 'output' --exclude 'tmp' "$SRC/" "$WORK/"

"$PYTHON" - "$WORK/paper_draft.tex" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text(encoding="utf-8")
shim = r"""% --- local-build shim (tectonic/XeTeX only; NOT part of the manuscript) ---
\providecommand{\pdfinfo}[1]{}
\ifx\pdfpagewidth\undefined\newdimen\pdfpagewidth\newdimen\pdfpageheight\fi
\usepackage[T1]{fontenc}
% --- end shim ---
"""
anchor = "\\pdfinfo{"
assert anchor in s, "anchor \\pdfinfo{ not found; update the shim"
p.write_text(s.replace(anchor, shim + anchor, 1), encoding="utf-8")
PY

cd "$WORK"
"$TECTONIC" -X compile paper_draft.tex --outdir "$OUT" --keep-logs --keep-intermediates
