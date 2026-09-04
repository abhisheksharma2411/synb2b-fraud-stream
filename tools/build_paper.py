#!/usr/bin/env python3
"""Compile paper/paper_T13.tex under both anonymisation settings and report pages.

The PDF is a build check, not a deliverable: `.gitignore` excludes it. What this
asserts is that the single file compiles with pdflatex alone, both ways, inside the
page limit, and that the anonymous build leaks nothing.
"""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper", "paper_T13.tex")
BUILD = os.path.join(ROOT, "paper", ".build")

LEAKS = [
    r"Abhishek\s+Sharma",
    r"abhisheksharma2411",
    r"our\s+(?:earlier|prior|previous)\s+(?:work|dataset|paper|benchmark)",
    r"\bAcknowledg",
    r"abhicse24",
    r"orcid",
]


def build(anon: bool):
    src = open(PAPER, encoding="utf-8").read()
    want = r"\anontrue" if anon else r"\anonfalse"
    patched, n = re.subn(r"\\anon(?:true|false)", want, src, count=1)
    if n == 0:
        sys.exit("no \\anontrue/\\anonfalse toggle found in the paper")
    os.makedirs(BUILD, exist_ok=True)
    name = "anon" if anon else "named"
    tmp = os.path.join(BUILD, f"{name}.tex")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(patched)
    ok = True
    for _ in range(3):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", BUILD, tmp],
            capture_output=True, text=True, cwd=os.path.join(ROOT, "paper"))
    pdf = os.path.join(BUILD, f"{name}.pdf")
    if not os.path.exists(pdf):
        print(f"[{name}] pdflatex produced no PDF")
        print((r.stdout or "")[-3000:])
        return None, False
    pages = len(re.findall(rb"/Type\s*/Page(?![sA-Za-z])", open(pdf, "rb").read()))
    log = open(os.path.join(BUILD, f"{name}.log"), encoding="utf-8",
               errors="replace").read()
    for w in ("Overfull \\hbox", "Undefined control sequence", "LaTeX Error"):
        c = log.count(w)
        if c:
            print(f"[{name}] {c} x {w}")
            if w != "Overfull \\hbox":
                ok = False
    # leak check on the anonymous source
    if anon:
        body = re.sub(r"(?m)^\s*%.*$", "", patched)
        for pat in LEAKS:
            hits = re.findall(pat, body, re.I)
            if hits:
                print(f"[anon] LEAK: {pat!r} -> {set(hits)}")
                ok = False
    return pages, ok


if __name__ == "__main__":
    if not shutil.which("pdflatex"):
        sys.exit("pdflatex not found on PATH")
    allok = True
    for anon in (False, True):
        pages, ok = build(anon)
        tag = "anon" if anon else "named"
        print(f"{tag:6s} pages={pages} compile_ok={ok} "
              f"within_limit={bool(pages and pages <= 6)}")
        allok &= bool(ok and pages and pages <= 6)
    print("BUILD OK" if allok else "BUILD FAILED")
    sys.exit(0 if allok else 1)
