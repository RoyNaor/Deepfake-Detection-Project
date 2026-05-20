"""
Academic paper search tool — searches real papers from 2023 to present.

Two backends:
  - semantic   : Semantic Scholar API (free, no key, indexes same papers as Google Scholar)
  - scholarly  : Unofficial Google Scholar scraper (pip install scholarly)

Usage:
    python scripts/search_papers.py "deepfake detection audio"
    python scripts/search_papers.py "voice spoofing anti-spoofing" --limit 10
    python scripts/search_papers.py "wav2vec deepfake" --year-start 2024
    python scripts/search_papers.py "speech synthesis detection" --backend scholarly
    python scripts/search_papers.py "deepfake audio" --bibtex --save results/papers.json
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.parse
import urllib.error


# ---------------------------------------------------------------------------
# Backend: Semantic Scholar (free, no key needed for up to 100 req/5min)
# API docs: https://api.semanticscholar.org/api-docs/
# ---------------------------------------------------------------------------

_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "title,authors,year,abstract,externalIds,citationCount,url,openAccessPdf,venue"


def _s2_search(query: str, year_start: int, limit: int) -> list[dict]:
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": _S2_FIELDS,
        "year": f"{year_start}-",
    }
    url = _S2_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "DeepfakeResearch/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode()).get("data", [])
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("[rate-limited] waiting 10s...", file=sys.stderr)
            time.sleep(10)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode()).get("data", [])
        raise


def _s2_to_common(paper: dict) -> dict:
    authors = [a["name"] for a in paper.get("authors", [])]
    ids = paper.get("externalIds") or {}
    pdf = (paper.get("openAccessPdf") or {}).get("url", "")
    return {
        "title": paper.get("title", ""),
        "authors": authors,
        "year": paper.get("year"),
        "abstract": paper.get("abstract", ""),
        "venue": paper.get("venue", ""),
        "doi": ids.get("DOI", ""),
        "arxiv": ids.get("ArXiv", ""),
        "url": paper.get("url", ""),
        "pdf": pdf,
        "citations": paper.get("citationCount", 0),
    }


# ---------------------------------------------------------------------------
# Backend: scholarly (pip install scholarly) — real Google Scholar scraper
# ---------------------------------------------------------------------------

def _scholarly_search(query: str, year_start: int, limit: int) -> list[dict]:
    try:
        from scholarly import scholarly
    except ImportError:
        print("scholarly is not installed. Run:  pip install scholarly", file=sys.stderr)
        sys.exit(1)

    results = []
    search_gen = scholarly.search_pubs(query)

    for _ in range(limit * 3):  # over-fetch since we filter by year
        try:
            pub = next(search_gen)
        except StopIteration:
            break

        bib = pub.get("bib", {})
        year_str = bib.get("pub_year", "0")
        try:
            year = int(year_str)
        except (ValueError, TypeError):
            year = 0

        if year < year_start:
            continue

        authors_raw = bib.get("author", "")
        authors = [a.strip() for a in authors_raw.split(" and ")] if authors_raw else []

        results.append({
            "title": bib.get("title", ""),
            "authors": authors,
            "year": year,
            "abstract": bib.get("abstract", ""),
            "venue": bib.get("venue", ""),
            "doi": "",
            "arxiv": "",
            "url": pub.get("pub_url", ""),
            "pdf": pub.get("eprint_url", ""),
            "citations": pub.get("num_citations", 0),
        })

        if len(results) >= limit:
            break

        time.sleep(1)  # be polite to Google Scholar

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_citation(paper: dict, index: int) -> str:
    authors = paper["authors"]
    if len(authors) > 3:
        author_str = ", ".join(authors[:3]) + " et al."
    else:
        author_str = ", ".join(authors) if authors else "Unknown"

    year = paper["year"] or "n.d."
    lines = [f"[{index}] {author_str} ({year}). {paper['title']}."]
    if paper["venue"]:
        lines.append(f"     Venue:     {paper['venue']}")
    if paper["doi"]:
        lines.append(f"     DOI:       https://doi.org/{paper['doi']}")
    if paper["arxiv"]:
        lines.append(f"     arXiv:     https://arxiv.org/abs/{paper['arxiv']}")
    if paper["pdf"]:
        lines.append(f"     PDF:       {paper['pdf']}")
    elif paper["url"]:
        lines.append(f"     URL:       {paper['url']}")
    lines.append(f"     Citations: {paper['citations']}")
    if paper["abstract"]:
        snippet = paper["abstract"]
        if len(snippet) > 220:
            snippet = snippet[:220].rsplit(" ", 1)[0] + "..."
        lines.append(f"     Abstract:  {snippet}")
    return "\n".join(lines)


def _format_bibtex(paper: dict) -> str:
    authors = paper["authors"]
    first_last = (authors[0].split()[-1] if authors else "unknown").lower()
    first_last = "".join(c for c in first_last if c.isalnum())
    year = paper["year"] or "0000"
    title_words = paper["title"].split()
    key_word = "".join(c for c in title_words[0].lower() if c.isalnum()) if title_words else "paper"
    cite_key = f"{first_last}{year}{key_word}"

    entry_type = "article" if paper["venue"] else "misc"
    lines = [f"@{entry_type}{{{cite_key},"]
    lines.append(f'  title         = {{{paper["title"]}}},')
    lines.append(f'  author        = {{{" and ".join(authors)}}},')
    lines.append(f'  year          = {{{year}}},')
    if paper["venue"]:
        lines.append(f'  journal       = {{{paper["venue"]}}},')
    if paper["doi"]:
        lines.append(f'  doi           = {{{paper["doi"]}}},')
    if paper["arxiv"]:
        lines.append(f'  eprint        = {{{paper["arxiv"]}}},')
        lines.append( '  archivePrefix = {arXiv},')
    if paper["url"] and not paper["doi"]:
        lines.append(f'  url           = {{{paper["url"]}}},')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Search academic papers (2023–present) for quoting and citation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", help="Search query, e.g. 'deepfake audio detection'")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max results (default: 10)")
    parser.add_argument("--year-start", type=int, default=2023,
                        help="Earliest year to include (default: 2023)")
    parser.add_argument("--backend", choices=["semantic", "scholarly"], default="semantic",
                        help="Search backend (default: semantic)")
    parser.add_argument("--bibtex", action="store_true",
                        help="Print BibTeX entries for all results")
    parser.add_argument("--save", metavar="FILE",
                        help="Save raw results as JSON to FILE")
    args = parser.parse_args()

    print(f'\nSearching for: "{args.query}"  |  year >= {args.year_start}  |  backend: {args.backend}\n')
    print("-" * 72)

    if args.backend == "semantic":
        raw = _s2_search(args.query, args.year_start, args.limit)
        papers = [_s2_to_common(p) for p in raw]
        raw_for_save = raw
    else:
        papers = _scholarly_search(args.query, args.year_start, args.limit)
        raw_for_save = papers

    if not papers:
        print("No results found. Try a different query or --backend.")
        return

    print(f"Found {len(papers)} papers:\n")
    for i, paper in enumerate(papers, 1):
        print(_format_citation(paper, i))
        print()

    if args.bibtex:
        print("\n" + "=" * 72)
        print("BibTeX entries:\n")
        for paper in papers:
            print(_format_bibtex(paper))
            print()

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(raw_for_save, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {args.save}")


if __name__ == "__main__":
    main()
