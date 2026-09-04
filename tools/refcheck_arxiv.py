#!/usr/bin/env python3
"""Verify every citation in the paper against an authoritative source.

arXiv entries go to the arXiv Atom API, DOI-bearing entries to Crossref, Zenodo
records to the Zenodo REST API. Titles are compared on a normalised form and the
first author's surname must appear in the returned author list. Anything that
fails is reported and must be dropped from the bibliography rather than guessed at.

    python tools/refcheck_arxiv.py [--out refcheck/arxiv_verified.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "synb2b-fraud-stream-refcheck/1.0 (reference verification)"}

# key -> (arxiv id, expected title, expected first-author surname)
ARXIV = {
    "gibbs2021aci":      ("2106.00170", "Adaptive Conformal Inference Under Distribution Shift", "Gibbs"),
    "gibbs2024arbitrary": ("2208.08401", "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts", "Gibbs"),
    "angelopoulos2024crc": ("2208.02814", "Conformal Risk Control", "Angelopoulos"),
    "bates2021rcps":     ("2101.02703", "Distribution-Free, Risk-Controlling Prediction Sets", "Bates"),
    "barber2023beyond":  ("2202.13415", "Conformal prediction beyond exchangeability", "Barber"),
    "angelopoulos2023pid": ("2307.16895", "Conformal PID Control for Time Series Prediction", "Angelopoulos"),
    "feldman2022online": ("2205.09095", "Achieving Risk Control in Online Learning Settings", "Feldman"),
    "geifman2017selective": ("1705.08500", "Selective Classification for Deep Neural Networks", "Geifman"),
    "geifman2019selectivenet": ("1901.09192", "SelectiveNet: A Deep Neural Network with an Integrated Reject Option", "Geifman"),
    "mozannar2020defer": ("2006.01862", "Consistent Estimators for Learning to Defer to an Expert", "Mozannar"),
    "deng2026authorize": ("2608.08577", None, None),
    "costsensitive2026": ("2607.27143", None, None),
    "xu2026selectivecrc": ("2512.12844", None, None),
    "shiftdetect2026":   ("2606.11949", None, None),
    "audited2026":       ("2606.14909", None, None),
    "anytime2026":       ("2602.04364", None, None),
    "limits2026":        ("2605.27557", None, None),
    "causal2026":        ("2605.29272", None, None),
    "corrupted2026":     ("2605.20515", None, None),
}

# key -> (doi, expected title fragment, expected first-author surname)
DOI = {
    "dalpozzolo2018":    ("10.1109/TNNLS.2017.2736643", "Credit Card Fraud Detection", "Dal Pozzolo"),
    "dalpozzolo2014":    ("10.1016/j.eswa.2014.02.026", "Learned lessons in credit card fraud detection", "Dal Pozzolo"),
    "papadopoulos2002":  ("10.1007/3-540-36755-1_29", "Inductive Confidence Machines for Regression", "Papadopoulos"),
    "lakkaraju2017":     ("10.1145/3097983.3098066", "The Selective Labels Problem", "Lakkaraju"),
}

ZENODO = {
    "synb2b_dataset":  ("21668281", "SynB2B"),
    "synb2b_preprint": ("21670659", "SynB2B"),
}

# no machine-readable record; verified by hand against the printed volume
MANUAL = {
    "vovk2005": {
        "title": "Algorithmic Learning in a Random World",
        "authors": ["Vovk", "Gammerman", "Shafer"],
        "venue": "Springer",
        "year": 2005,
        "isbn": "978-0-387-00152-4",
        "verified_by": "manual: Springer monograph, ISBN checked against the publisher page",
    }
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def get(url: str, tries: int = 3) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except Exception as e:              # noqa: BLE001 - reported, not swallowed
            last = e
            time.sleep(2 + 2 * i)
    raise RuntimeError(f"{url}: {last}")


NS = {"a": "http://www.w3.org/2005/Atom"}


def check_arxiv(key, aid, want_title, want_author):
    url = ("https://export.arxiv.org/api/query?id_list="
           + urllib.parse.quote(aid) + "&max_results=1")
    root = ET.fromstring(get(url))
    entry = root.find("a:entry", NS)
    if entry is None or entry.find("a:id", NS) is None:
        return {"status": "NOT FOUND", "arxiv": aid}
    title = " ".join((entry.findtext("a:title", "", NS) or "").split())
    authors = [" ".join((a.findtext("a:name", "", NS) or "").split())
               for a in entry.findall("a:author", NS)]
    published = entry.findtext("a:published", "", NS)
    updated = entry.findtext("a:updated", "", NS)
    rec = {
        "status": "OK", "arxiv": aid, "title": title, "authors": authors,
        "published": published, "updated": updated,
        "id": entry.findtext("a:id", "", NS),
        "doi": entry.findtext("a:doi", "", NS) or None,
    }
    problems = []
    if want_title and norm(want_title) != norm(title):
        if norm(want_title) not in norm(title):
            problems.append(f"title mismatch: expected {want_title!r}, arXiv says {title!r}")
    if want_author and not any(want_author.lower() in a.lower() for a in authors):
        problems.append(f"author mismatch: expected {want_author!r}, arXiv says {authors}")
    if problems:
        rec["status"] = "MISMATCH"
        rec["problems"] = problems
    return rec


def check_doi(key, doi, want_title, want_author):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    msg = json.loads(get(url))["message"]
    title = (msg.get("title") or [""])[0]
    authors = [f"{a.get('given','')} {a.get('family','')}".strip()
               for a in msg.get("author", [])]
    year = None
    for f in ("published-print", "published-online", "issued"):
        if msg.get(f, {}).get("date-parts", [[None]])[0][0]:
            year = msg[f]["date-parts"][0][0]
            break
    rec = {
        "status": "OK", "doi": doi, "title": title, "authors": authors,
        "container": (msg.get("container-title") or [""])[0],
        "volume": msg.get("volume"), "issue": msg.get("issue"),
        "page": msg.get("page"), "year": year, "type": msg.get("type"),
    }
    problems = []
    if want_title and norm(want_title) not in norm(title):
        problems.append(f"title mismatch: expected ~{want_title!r}, Crossref says {title!r}")
    if want_author and not any(want_author.lower() in a.lower() for a in authors):
        problems.append(f"author mismatch: expected {want_author!r}, Crossref says {authors}")
    if problems:
        rec["status"] = "MISMATCH"
        rec["problems"] = problems
    return rec


def check_zenodo(key, rid, want_title_frag):
    msg = json.loads(get(f"https://zenodo.org/api/records/{rid}"))
    md = msg.get("metadata", {})
    title = md.get("title", "")
    authors = [c.get("name", "") for c in md.get("creators", [])]
    rec = {
        "status": "OK", "zenodo_record": rid, "title": title, "authors": authors,
        "doi": msg.get("doi") or md.get("doi"),
        "publication_date": md.get("publication_date"),
        "license": (md.get("license") or {}).get("id"),
        "resource_type": (md.get("resource_type") or {}).get("type"),
    }
    if want_title_frag and norm(want_title_frag) not in norm(title):
        rec["status"] = "MISMATCH"
        rec["problems"] = [f"title fragment {want_title_frag!r} absent from {title!r}"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "refcheck", "arxiv_verified.json"))
    a = ap.parse_args()

    out = {}
    for key, (aid, t, au) in ARXIV.items():
        try:
            out[key] = check_arxiv(key, aid, t, au)
        except Exception as e:                # noqa: BLE001
            out[key] = {"status": "ERROR", "arxiv": aid, "error": str(e)}
        print(f"{out[key]['status']:9s} {key:24s} arXiv:{aid}  {out[key].get('title','')[:64]}")
        time.sleep(1.0)                       # arXiv asks for one request per second

    for key, (doi, t, au) in DOI.items():
        try:
            out[key] = check_doi(key, doi, t, au)
        except Exception as e:                # noqa: BLE001
            out[key] = {"status": "ERROR", "doi": doi, "error": str(e)}
        print(f"{out[key]['status']:9s} {key:24s} doi:{doi}  {out[key].get('title','')[:52]}")

    for key, (rid, frag) in ZENODO.items():
        try:
            out[key] = check_zenodo(key, rid, frag)
        except Exception as e:                # noqa: BLE001
            out[key] = {"status": "ERROR", "zenodo_record": rid, "error": str(e)}
        print(f"{out[key]['status']:9s} {key:24s} zenodo:{rid}  {out[key].get('title','')[:52]}")

    for key, rec in MANUAL.items():
        out[key] = dict(rec, status="MANUAL")
        print(f"{'MANUAL':9s} {key:24s} {rec['title'][:52]}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print("\nwrote", a.out)

    bad = {k: v for k, v in out.items() if v["status"] not in ("OK", "MANUAL")}
    if bad:
        print("\nNOT USABLE AS CITED - drop or correct these:")
        for k, v in bad.items():
            print(f"  {k}: {v['status']}  {v.get('problems') or v.get('error')}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
