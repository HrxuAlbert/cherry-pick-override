#!/usr/bin/env python3
"""Generate the prompt appendix from the hashed prompt templates.

The SHA-256 check is a build guard, not reader-facing content. It ran in the
manuscript for a while, printed above each listing; but the paper carries no
pointer to the code release, so a reader had a digest and nowhere to check it
against. It stays here, where a mismatch is a hard error and the appendix
cannot be regenerated from a prompt that was not the one executed.

The appendix used to be a hand-transcribed copy of the prompts, rewrapped by
hand to 62 columns to fit a two-column measure. The content was faithful ---
it matches the templates exactly once whitespace is normalised --- but a
reader had no way to check that, and the wrapping was indistinguishable from
newlines that were actually in the prompt.

This script removes both problems. It reads the template files the runner
itself loads, verifies each against the SHA-256 recorded in the executed
manifest, and emits the appendix with the template's own line structure
intact. Lines too long for the measure are wrapped by ``listings`` and marked
with a visible continuation arrow, so a prompt newline and a typographic wrap
are never confused.

Re-run after any change to a prompt template. The hashes below are the values
recorded in the run manifests; a mismatch means the manuscript would document
a prompt that was not the one executed, so it is a hard error.
"""

from __future__ import annotations

def _cpo_workspace(_f=__file__):
    """Resolve the release root.

    In the author's tree these scripts live at
    <root>/Writing/V0.2/revision_plan/scripts/, so parents[4] is the root. In
    this release they live at <repo>/scripts/revision_2026/, so walk up until a
    directory containing outputs/revision_2026 is found and fall back to the
    original rule.
    """
    import os, pathlib as _p
    env = os.environ.get("CPO_WORKSPACE")
    if env:
        return _p.Path(env).resolve()
    here = _p.Path(_f).resolve()
    for parent in here.parents:
        if (parent / "outputs" / "revision_2026").is_dir():
            return parent
    return here.parents[4]

import argparse
import hashlib
from pathlib import Path

def _first_existing(*candidates):
    """First candidate that exists, else the first, so a failure names the
    canonical location. These paths differ between the author's working
    tree and the public release."""
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


WORKSPACE = _cpo_workspace()
# The prompt templates sit at different depths in the author's tree and in
# the public release. Try both rather than hardcoding one; --help alone does
# not exercise this path, which is how a release shipped with it broken.
_PROMPT_CANDIDATES = (
    WORKSPACE / "Writing/V0.2/code_release/scripts/option_a_exp/prompts/judges",
    WORKSPACE / "scripts/option_a_exp/prompts/judges",
)
PROMPTS = next((p for p in _PROMPT_CANDIDATES if p.is_dir()), _PROMPT_CANDIDATES[0])
# Default output: the manuscript tree when present, otherwise beside the
# release so the script has somewhere to write without --output.
_OUTPUT_CANDIDATES = (
    WORKSPACE / "overleaf-paper/sections/appendix_prompts.tex",
    WORKSPACE / "appendix_prompts.tex",
)
# Pick by whether the *parent* exists: an output file does not exist yet, so
# testing the file itself always fell through to the manuscript tree and then
# failed to write.
OUTPUT = next((p for p in _OUTPUT_CANDIDATES if p.parent.is_dir()),
              _OUTPUT_CANDIDATES[0])


# (file, SHA-256 as recorded in the executed manifest)
JUDGE = ("honest_4opt_strong.txt",
         "1343845271fe50b3d55ea1f2364f0fa6bbaeb2c65984998f62802320aef7a350")
VALIDATOR = ("certificate_strict_fewshot.txt",
             "0ca38822936d9fae481b6f4ea6699202656c56e211955ca71c350ea7af42ac26")


def load_verified(name: str, expected: str) -> str:
    path = PROMPTS / name
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise SystemExit(
            f"{name}: SHA-256 mismatch\n  expected {expected}\n  actual   {actual}\n"
            "The manuscript would document a prompt that was not executed."
        )
    return raw.decode("utf-8")


ESCAPE_OPEN, ESCAPE_CLOSE = "(*@", "@*)"

# Characters with no glyph in Courier. Left as raw bytes they are dropped from
# the PDF with only a "Missing character" line in the log, so each one is
# rendered through listings' escapeinside instead.
NON_ASCII_MATH = {"\u2192": r"\rightarrow"}


def listing(body: str) -> str:
    assert "\\end{lstlisting}" not in body
    for delim in (ESCAPE_OPEN, ESCAPE_CLOSE):
        assert delim not in body, f"escape delimiter {delim!r} occurs in the prompt"
    for char, macro in NON_ASCII_MATH.items():
        body = body.replace(char, f"{ESCAPE_OPEN}${macro}${ESCAPE_CLOSE}")
    leftover = {c for c in body if ord(c) > 127}
    assert not leftover, f"unhandled non-ASCII in prompt: {[hex(ord(c)) for c in leftover]}"
    return ("\\begin{lstlisting}[style=promptstyle]\n"
            + body.rstrip("\n") + "\n\\end{lstlisting}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()

    judge = load_verified(*JUDGE)
    validator = load_verified(*VALIDATOR)

    tex = r"""% GENERATED FILE --- do not edit by hand.
% Produced by Writing/V0.2/revision_plan/scripts/export_prompt_appendix.py
% from the prompt templates the runner loads, each verified against the
% SHA-256 recorded in the executed run manifest. Re-run that script instead
% of editing this file.
\section{Prompts}
\label{sec:appendix:prompts}

Each prompt below is the byte content of the template file the runner loads,
named above its listing; the same files are in the code release under
\texttt{scripts/option\_a\_exp/prompts/judges/}, where four prompts are
shipped and the name identifies which one this is.
\verb|<<CLAIM>>| and \verb|<<EVIDENCE>>| are the
substitution points for the claim text and the rendered gold
question--answer evidence block. Line breaks are the template's own, except
where a line exceeds the measure and is wrapped by the typesetter; those
wraps carry a \textcolor{promptrule}{$\hookrightarrow$} marker, so a wrap is
never mistaken for a newline in the prompt.

\subsection{Typed four-option verdict prompt}
\label{sec:appendix:prompts:judge}

This template is shared by all panel judges and by all four contemporary
individual judges. Note the four enumerated trigger conditions
\mbox{(a)--(d)}, the imperative sentence forbidding collapse to
\textsc{Refutes}, and the three demonstrations, each of which resolves
to \textsc{Conflicting}. As discussed in \S\ref{sec:method}, this makes
every directional-overcommitment rate in the paper a conservative lower
bound.

\smallskip
\noindent\texttt{__JUDGE_NAME__}

__JUDGE_BODY__

\subsection{Certificate validator prompt}
\label{sec:appendix:prompts:validator}

The validator runs independently of the panel and is used only as the
structural channel of the two-channel reference probe
(\S\ref{sec:method:controllers}). Its \texttt{final\_verdict} field is
not consumed by the probe; only the per-subclaim
\texttt{evidence\_state} assignments are.

\smallskip
\noindent\texttt{__VALIDATOR_NAME__}

__VALIDATOR_BODY__
"""
    tex = (tex
           .replace("__JUDGE_NAME__", JUDGE[0].replace("_", r"\_"))
           .replace("__JUDGE_BODY__", listing(judge))
           .replace("__VALIDATOR_NAME__", VALIDATOR[0].replace("_", r"\_"))
           .replace("__VALIDATOR_BODY__", listing(validator)))

    args.output.write_text(tex, encoding="utf-8")
    print(f"wrote {args.output}")
    for name, sha in (JUDGE, VALIDATOR):
        body = load_verified(name, sha)
        lines = body.split("\n")
        over = [l for l in lines if len(l) > 93]
        print(f"  {name}: {len(lines)} lines, longest {max(len(l) for l in lines)} chars, "
              f"{len(over)} will wrap at \\footnotesize (93-char measure)")


if __name__ == "__main__":
    main()
