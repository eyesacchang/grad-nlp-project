"""Literature scraper: Semantic Scholar, ACL Anthology, OpenReview, ArXiv.

Searches for papers related to cross-lingual alignment, layer-wise fine-tuning,
and multilingual representation learning. Outputs papers_found.csv.

Usage:
    python scripts/search_papers.py --output papers_found.csv
    python scripts/search_papers.py --sources semantic_scholar acl arxiv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import csv
import re
import json
import gzip
import urllib.request
import urllib.parse
from xml.etree import ElementTree


SEARCH_TERMS = [
    "cross-lingual alignment multilingual",
    "layer-wise learning rate transformer",
    "multilingual fine-tuning contrastive",
    "representation drift fine-tuning",
    "contrastive multilingual sentence embedding",
    "XLM-R fine-tuning cross-lingual",
    "discriminative fine-tuning transformer",
    "language-neutral representations multilingual",
    "Procrustes multilingual alignment",
    "centered kernel alignment representation similarity",
    "InfoNCE multilingual",
    "cross-lingual transfer geometry",
]

TARGET_VENUES = {
    "NeurIPS", "Neural Information Processing Systems",
    "ICLR", "International Conference on Learning Representations",
    "ICML", "International Conference on Machine Learning",
    "ACL", "EMNLP", "NAACL", "EACL", "CoNLL",
}


# ── Semantic Scholar ──────────────────────────────────────────────────────────

def search_semantic_scholar(query, fields, limit=100, delay=1.1):
    """Paginate through Semantic Scholar results for a query."""
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    results = []
    offset = 0

    while True:
        params = urllib.parse.urlencode({
            "query": query,
            "fields": ",".join(fields),
            "limit": limit,
            "offset": offset,
        })
        url = f"{base_url}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  [S2 error] {e}")
            break

        papers = data.get("data", [])
        results.extend(papers)

        if len(papers) < limit or len(results) >= 500:
            break
        offset += limit
        time.sleep(delay)

    return results


def collect_semantic_scholar():
    fields = ["title", "abstract", "authors", "year", "venue",
              "externalIds", "citationCount", "openAccessPdf"]
    all_papers = {}

    for term in SEARCH_TERMS:
        print(f"  [S2] Querying: {term!r}")
        papers = search_semantic_scholar(term, fields)
        for p in papers:
            pid = p.get("paperId", "")
            if pid and pid not in all_papers:
                all_papers[pid] = p
        time.sleep(1.0)

    return list(all_papers.values())


# ── ACL Anthology ─────────────────────────────────────────────────────────────

def collect_acl_anthology(min_year=2020):
    """Download and parse ACL Anthology BibTeX for recent papers."""
    url = "https://aclanthology.org/anthology+abstracts.bib.gz"
    print("  [ACL] Downloading anthology BibTeX (this may take a moment)...")

    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = gzip.decompress(resp.read()).decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [ACL error] {e}")
        return []

    keywords = set()
    for term in SEARCH_TERMS:
        for word in term.lower().split():
            if len(word) > 4:
                keywords.add(word)

    pattern = re.compile(
        r"@\w+\{([^,]+),\s*(.*?)\n\}", re.DOTALL
    )
    papers = []
    for match in pattern.finditer(raw):
        body = match.group(2)
        title_m = re.search(r'title\s*=\s*[{"](.+?)[}"]', body, re.IGNORECASE | re.DOTALL)
        abstract_m = re.search(r'abstract\s*=\s*[{"](.+?)[}"]', body, re.IGNORECASE | re.DOTALL)
        year_m = re.search(r'year\s*=\s*[{"]?(\d{4})[}"]?', body)
        author_m = re.search(r'author\s*=\s*[{"](.+?)[}"]', body, re.IGNORECASE | re.DOTALL)
        venue_m = re.search(r'booktitle\s*=\s*[{"](.+?)[}"]', body, re.IGNORECASE | re.DOTALL)
        url_m = re.search(r'url\s*=\s*[{"](.+?)[}"]', body)

        if not title_m or not year_m:
            continue
        year = int(year_m.group(1))
        if year < min_year:
            continue

        title = title_m.group(1).strip()
        abstract = abstract_m.group(1).strip() if abstract_m else ""
        text = (title + " " + abstract).lower()

        if any(kw in text for kw in keywords):
            papers.append({
                "title": title,
                "abstract": abstract,
                "authors": author_m.group(1).strip() if author_m else "",
                "year": year,
                "venue": venue_m.group(1).strip() if venue_m else "ACL Anthology",
                "url": url_m.group(1).strip() if url_m else "",
                "citation_count": "",
                "source": "acl_anthology",
            })

    return papers


# ── OpenReview (ICLR) ─────────────────────────────────────────────────────────

def collect_openreview(years=None):
    if years is None:
        years = [2022, 2023, 2024, 2025]

    keywords = set()
    for term in SEARCH_TERMS:
        for word in term.lower().split():
            if len(word) > 4:
                keywords.add(word)

    papers = []
    for year in years:
        url = (f"https://api2.openreview.net/notes"
               f"?invitation=ICLR.cc/{year}/Conference/-/Blind_Submission"
               f"&details=replyCount&limit=1000")
        print(f"  [OpenReview] Fetching ICLR {year}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  [OpenReview error] {e}")
            continue

        for note in data.get("notes", []):
            content = note.get("content", {})
            title = content.get("title", "")
            abstract = content.get("abstract", "")
            if isinstance(title, dict):
                title = title.get("value", "")
            if isinstance(abstract, dict):
                abstract = abstract.get("value", "")

            text = (title + " " + abstract).lower()
            if any(kw in text for kw in keywords):
                authors = content.get("authors", [])
                if isinstance(authors, dict):
                    authors = authors.get("value", [])
                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": ", ".join(authors) if isinstance(authors, list) else str(authors),
                    "year": year,
                    "venue": f"ICLR {year}",
                    "url": f"https://openreview.net/forum?id={note.get('id', '')}",
                    "citation_count": "",
                    "source": "openreview",
                })
        time.sleep(1.0)

    return papers


# ── ArXiv ─────────────────────────────────────────────────────────────────────

def collect_arxiv(min_year=2022):
    papers = []
    base_url = "https://export.arxiv.org/api/query"

    for term in SEARCH_TERMS[:6]:  # ArXiv is slower; limit queries
        encoded = urllib.parse.quote(term)
        url = (f"{base_url}?search_query=all:{encoded}+AND+(cat:cs.CL+OR+cat:cs.LG)"
               f"&max_results=100&sortBy=submittedDate&sortOrder=descending")
        print(f"  [ArXiv] Querying: {term!r}")
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                xml_data = resp.read()
        except Exception as e:
            print(f"  [ArXiv error] {e}")
            time.sleep(3)
            continue

        root = ElementTree.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            abstract = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
            published = entry.findtext("atom:published", "", ns) or ""
            year_str = published[:4]
            if not year_str.isdigit() or int(year_str) < min_year:
                continue

            arxiv_id = (entry.findtext("atom:id", "", ns) or "").split("/abs/")[-1]
            authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]

            papers.append({
                "title": title,
                "abstract": abstract,
                "authors": ", ".join(authors),
                "year": int(year_str),
                "venue": "ArXiv",
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "citation_count": "",
                "source": "arxiv",
            })

        time.sleep(3)

    return papers


# ── Deduplication and Export ──────────────────────────────────────────────────

def normalize_title(title):
    return re.sub(r"[^a-z0-9]", "", title.lower())


def deduplicate(all_papers):
    seen = {}
    for p in all_papers:
        key = normalize_title(p.get("title", ""))
        if key not in seen:
            seen[key] = p
        else:
            # Prefer entry with citation count
            if p.get("citation_count") and not seen[key].get("citation_count"):
                seen[key] = p
    return list(seen.values())


def flatten_s2_paper(p):
    authors = p.get("authors", [])
    author_str = ", ".join(a.get("name", "") for a in authors) if authors else ""
    ext_ids = p.get("externalIds") or {}
    arxiv_id = ext_ids.get("ArXiv", "")
    url = p.get("openAccessPdf", {}) or {}
    url = url.get("url", f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")
    return {
        "title": p.get("title", ""),
        "abstract": p.get("abstract", "") or "",
        "authors": author_str,
        "year": p.get("year", ""),
        "venue": p.get("venue", ""),
        "url": url,
        "citation_count": p.get("citationCount", ""),
        "source": "semantic_scholar",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="papers_found.csv")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["semantic_scholar", "acl", "openreview", "arxiv"],
        choices=["semantic_scholar", "acl", "openreview", "arxiv"],
    )
    args = parser.parse_args()

    all_papers = []

    if "semantic_scholar" in args.sources:
        print("\n[1/4] Collecting from Semantic Scholar...")
        s2_papers = collect_semantic_scholar()
        all_papers.extend(flatten_s2_paper(p) for p in s2_papers)
        print(f"  Found {len(s2_papers)} papers")

    if "acl" in args.sources:
        print("\n[2/4] Collecting from ACL Anthology...")
        acl_papers = collect_acl_anthology()
        all_papers.extend(acl_papers)
        print(f"  Found {len(acl_papers)} papers")

    if "openreview" in args.sources:
        print("\n[3/4] Collecting from OpenReview (ICLR)...")
        or_papers = collect_openreview()
        all_papers.extend(or_papers)
        print(f"  Found {len(or_papers)} papers")

    if "arxiv" in args.sources:
        print("\n[4/4] Collecting from ArXiv...")
        ax_papers = collect_arxiv()
        all_papers.extend(ax_papers)
        print(f"  Found {len(ax_papers)} papers")

    print(f"\nTotal before dedup: {len(all_papers)}")
    deduped = deduplicate(all_papers)
    print(f"Total after dedup:  {len(deduped)}")

    # Sort by citation count descending
    deduped.sort(key=lambda x: int(x.get("citation_count") or 0), reverse=True)

    fieldnames = ["title", "authors", "year", "venue", "citation_count", "url", "abstract", "source"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(deduped)

    print(f"\nSaved {len(deduped)} papers to {args.output}")


if __name__ == "__main__":
    main()
