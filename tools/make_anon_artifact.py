#!/usr/bin/env python3
"""Build an anonymous artifact for double-blind review.

A reviewer cannot check the estimator, the feasibility oracle or the flag numbers if
the artifact is withheld, and "URL withheld for review" gives them nothing. This
exports the tracked tree with no git history, strips every author identifier, sets the
paper's toggle to the anonymous build, and refuses to finish if anything leaks.

    python tools/make_anon_artifact.py [--out DIR]
"""
from __future__ import annotations
import argparse, os, re, shutil, subprocess, sys, tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (pattern, replacement) applied to every text file in the export
SUBS = [
    (r"Abhishek\s+Sharma", "Anonymous Author"),
    (r"\bAbhishek\b", "Anonymous"),
    (r"\bSharma,\s*Abhishek\b", "Anonymous Author"),
    (r"\bA\.\s*Sharma\b", "Anonymous"),
    (r"\bSharma\b", "Anonymous"),
    (r"abhisheksharma2411", "anonymous-author"),
    (r"abhicse24@gmail\.com", "anonymous@example.org"),
    # bare form too: build_paper.py's own leak-detector list names the identifiers
    (r"abhicse24", "anonymous"),
    (r"0009-0007-1103-2103", "0000-0000-0000-0000"),
    (r"https?://github\.com/anonymous-author/[\w.-]+", "URL withheld for review"),
]
# these must not survive anywhere in the export
LEAKS = [r"Abhishek", r"Sharma", r"abhicse24", r"abhisheksharma2411", r"0009-0007-1103-2103"]
TEXT_EXT = {".md", ".tex", ".py", ".txt", ".cff", ".json", ".yml", ".yaml", ".toml", ""}
# This script is not part of the research artifact, and it cannot anonymise itself: its
# own identifiers sit inside \b...\b regex escapes, where the word-boundary anchors it
# would match with cannot fire. Excluding it is simpler and leaves nothing to explain.
EXCLUDE = {"tools/make_anon_artifact.py"}
SKIP_NAMES = {"Makefile", "LICENSE", "DATA_LICENSE"}


def is_text(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in TEXT_EXT or os.path.basename(path) in SKIP_NAMES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "..", "anon-artifact"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    stage = os.path.join(out, "synb2b-fraud-stream-anonymous")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(stage, exist_ok=True)

    # git archive: tracked files only, and no history to mine for an author
    tar = subprocess.run(["git", "archive", "HEAD"], cwd=ROOT, capture_output=True, check=True)
    tmp = os.path.join(out, "_export.tar")
    open(tmp, "wb").write(tar.stdout)
    with tarfile.open(tmp) as t:
        t.extractall(stage)
    os.remove(tmp)

    for rel in EXCLUDE:
        fp = os.path.join(stage, rel)
        if os.path.exists(fp):
            os.remove(fp)

    n_files = n_edits = 0
    for dirpath, _, names in os.walk(stage):
        for nm in names:
            fp = os.path.join(dirpath, nm)
            if not is_text(fp):
                continue
            try:
                txt = open(fp, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            orig = txt
            for pat, rep in SUBS:
                txt = re.sub(pat, rep, txt)
            if txt != orig:
                open(fp, "w", encoding="utf-8").write(txt)
                n_edits += 1
            n_files += 1

    # ship the paper already flipped to the anonymous build
    paper = os.path.join(stage, "paper", "main.tex")
    if os.path.exists(paper):
        t = open(paper, encoding="utf-8").read()
        t = re.sub(r"\\anon(?:true|false)", lambda _m: r"\anontrue", t, count=1)
        open(paper, "w", encoding="utf-8").write(t)

    open(os.path.join(stage, "ANONYMOUS.md"), "w").write(
        "# Anonymous artifact\n\n"
        "Author identifiers, the repository URL and the git history are removed for\n"
        "double-blind review. `paper/main.tex` ships with the toggle set to the\n"
        "anonymous build. The script that produced this export is omitted; it is\n"
        "packaging, not part of the method.\n\n"
        "Everything else is the submitted tree. To reproduce:\n\n"
        "    make venv && make test\n"
        "    make reproduce-full     # writes results/results.json\n"
        "    make numbers            # re-derives every number the paper prints\n"
        "    make diagnostics        # estimand gap, ESS, propensity error, flag scoring\n"
        "    make audit && make paper\n\n"
        "`make reproduce-small` is a reduced pass for a quick check.\n")

    # refuse to finish if anything leaked
    found = {}
    for dirpath, _, names in os.walk(stage):
        for nm in names:
            fp = os.path.join(dirpath, nm)
            try:
                txt = open(fp, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for pat in LEAKS:
                for m in re.finditer(pat, txt, re.I):
                    found.setdefault(pat, []).append(os.path.relpath(fp, stage))
    if found:
        print("LEAK: anonymisation incomplete")
        for pat, files in found.items():
            print(f"  {pat}: {sorted(set(files))[:6]}")
        return 1

    base = os.path.join(out, "synb2b-fraud-stream-anonymous")
    shutil.make_archive(base, "gztar", root_dir=out, base_dir=os.path.basename(stage))
    shutil.make_archive(base, "zip", root_dir=out, base_dir=os.path.basename(stage))
    print(f"scanned {n_files} text files, rewrote {n_edits}")
    print("no author identifier survives anywhere in the export")
    for ext in (".tar.gz", ".zip"):
        p = base + ext
        print(f"  {p}  ({os.path.getsize(p)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
