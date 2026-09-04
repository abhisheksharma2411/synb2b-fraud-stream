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


def _count_pages(pdf: str) -> int:
    """Page count from pdfinfo when poppler is installed, else from the page objects.

    The regex alone reads zero on a PDF whose page tree lands in a compressed object
    stream, which is what pdfTeX emits here.
    """
    if shutil.which("pdfinfo"):
        r = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True)
        m = re.search(r"(?m)^Pages:\s+(\d+)", r.stdout or "")
        if m:
            return int(m.group(1))
    with open(pdf, "rb") as f:
        blob = f.read()
    return len(re.findall(rb"/Type\s*/Page(?![sA-Za-z])", blob))

def resolve_anon(tex: str, anon: bool) -> str:
    """Expand \\ifanon ... \\else ... \\fi so the leak scan sees what the reader sees.

    The deliverable is one file carrying both variants, so scanning the raw source
    would flag the named branch every time and could never pass. What has to be clean
    is the text the anonymous build actually renders.
    """
    pat = re.compile(r"\\ifanon(.*?)(?:\\else(.*?))?\\fi", re.S)
    prev = None
    while prev != tex:
        prev = tex
        tex = pat.sub(lambda m: (m.group(1) if anon else (m.group(2) or "")), tex)
    return tex


def pdf_text(pdf: str) -> str:
    """Rendered text of the PDF, or an empty string if no extractor is installed."""
    if not shutil.which("pdftotext"):
        return ""
    r = subprocess.run(["pdftotext", "-q", pdf, "-"], capture_output=True, text=True)
    return r.stdout or ""


def build(anon: bool):
    src = open(PAPER, encoding="utf-8").read()
    want = r"\anontrue" if anon else r"\anonfalse"
    # a function repl, because "\\anonfalse" as a replacement string would be read
    # as the BEL escape and inject U+0007 into the source.
    patched, n = re.subn(r"\\anon(?:true|false)", lambda _m: want, src, count=1)
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
    pages = _count_pages(pdf)
    log = open(os.path.join(BUILD, f"{name}.log"), encoding="utf-8",
               errors="replace").read()
    for w in ("Overfull \\hbox", "Undefined control sequence", "LaTeX Error"):
        c = log.count(w)
        if c:
            print(f"[{name}] {c} x {w}")
            if w != "Overfull \\hbox":
                ok = False
    # leak check: the resolved anonymous source, and the text the PDF renders
    if anon:
        body = re.sub(r"(?m)^\s*%.*$", "", resolve_anon(patched, anon=True))
        rendered = pdf_text(pdf)
        for pat in LEAKS:
            for label, hay in (("source", body), ("rendered pdf", rendered)):
                if not hay:
                    continue
                hits = re.findall(pat, hay, re.I)
                if hits:
                    print(f"[anon] LEAK in {label}: {pat!r} -> {set(hits)}")
                    ok = False
        if not rendered:
            print("[anon] note: pdftotext unavailable, checked the source only")
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
