"""
Fetch open-access cardiovascular risk papers from PubMed Central (PMC).

Uses the NCBI Entrez API via Biopython to search for and download
full-text articles related to heart failure, cardiovascular risk prediction,
and key biomarkers (ejection fraction, serum creatinine, etc.).
"""

import os
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from Bio import Entrez

# Entrez requires an email
Entrez.email = "cardiorisk-ml@research.edu"

# Directory to save fetched papers
PAPERS_DIR = Path(__file__).parent.parent.parent / "data" / "papers"
METADATA_FILE = PAPERS_DIR / "metadata.json"

# Search queries covering key cardiovascular risk topics
SEARCH_QUERIES = [
    # Core topic
    '"heart failure" AND "machine learning" AND mortality prediction',
    '"heart failure" AND "risk prediction" AND clinical',
    '"cardiovascular risk" AND "machine learning" AND biomarkers',
    
    # Key biomarkers from the project
    '"ejection fraction" AND "heart failure" AND prognosis',
    '"serum creatinine" AND "heart failure" AND mortality',
    '"serum creatinine" AND "cardiovascular" AND "kidney"',
    
    # ML methods in cardiology
    '"random forest" AND "cardiovascular" AND prediction',
    '"xgboost" AND "heart failure"',
    '"logistic regression" AND "heart failure" AND "risk score"',
    '"deep learning" AND "cardiovascular" AND prediction',
    
    # Clinical risk scores and models
    '"heart failure" AND "risk score" AND validation',
    '"Framingham" AND "heart failure" AND risk',
    '"BNP" OR "NT-proBNP" AND "heart failure" AND prognosis',
    
    # Data leakage and methodology
    '"data leakage" AND "machine learning" AND clinical',
    '"survival analysis" AND "heart failure" AND "Cox"',
    
    # Population-specific
    '"heart failure" AND "South Asian" AND risk',
    '"cardiorenal syndrome" AND prognosis',
    
    # Feature importance and interpretability
    '"SHAP" AND "cardiovascular" AND prediction',
    '"feature importance" AND "heart failure" AND "machine learning"',
    '"explainable AI" AND "cardiovascular"',
    
    # Calibration and fairness
    '"calibration" AND "risk prediction" AND cardiovascular',
    '"fairness" AND "machine learning" AND clinical prediction',
    
    # Broader cardiovascular ML
    '"atrial fibrillation" AND "machine learning" AND prediction',
    '"hypertension" AND "machine learning" AND "risk prediction"',
]


def search_pmc(query: str, max_results: int = 10) -> list[str]:
    """Search PubMed Central for open-access articles."""
    try:
        handle = Entrez.esearch(
            db="pmc",
            term=query + ' AND "open access"[filter]',
            retmax=max_results,
            sort="relevance",
        )
        record = Entrez.read(handle)
        handle.close()
        return record.get("IdList", [])
    except Exception as e:
        print(f"  Search error for '{query[:50]}...': {e}")
        return []


def fetch_article_text(pmc_id: str) -> dict | None:
    """Fetch full-text XML from PMC and extract text content."""
    try:
        handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="xml")
        xml_data = handle.read()
        handle.close()

        # Parse XML
        if isinstance(xml_data, bytes):
            xml_data = xml_data.decode("utf-8")

        root = ET.fromstring(xml_data)

        # Extract metadata
        article = root.find(".//article")
        if article is None:
            article = root

        # Title
        title_elem = article.find(".//article-title")
        title = "".join(title_elem.itertext()).strip() if title_elem is not None else "Unknown Title"

        # Authors
        authors = []
        for contrib in article.findall(".//contrib[@contrib-type='author']"):
            surname = contrib.find(".//surname")
            given = contrib.find(".//given-names")
            if surname is not None:
                name = surname.text or ""
                if given is not None and given.text:
                    name = f"{given.text} {name}"
                authors.append(name)

        # Journal
        journal_elem = article.find(".//journal-title")
        journal = journal_elem.text.strip() if journal_elem is not None and journal_elem.text else "Unknown Journal"

        # Year
        year_elem = article.find(".//pub-date/year")
        year = year_elem.text if year_elem is not None and year_elem.text else "Unknown"

        # PMID
        pmid_elem = article.find(".//article-id[@pub-id-type='pmid']")
        pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else None

        # Abstract
        abstract_parts = []
        for abs_elem in article.findall(".//abstract//p"):
            text = "".join(abs_elem.itertext()).strip()
            if text:
                abstract_parts.append(text)
        abstract = "\n".join(abstract_parts)

        # Body text
        body_parts = []
        for p_elem in article.findall(".//body//p"):
            text = "".join(p_elem.itertext()).strip()
            if text and len(text) > 20:  # Skip very short fragments
                body_parts.append(text)
        body = "\n\n".join(body_parts)

        if not abstract and not body:
            return None

        return {
            "pmc_id": pmc_id,
            "pmid": pmid,
            "title": title,
            "authors": authors[:5],  # Limit to first 5 authors
            "journal": journal,
            "year": year,
            "abstract": abstract,
            "body": body,
            "text": f"Title: {title}\n\nAbstract:\n{abstract}\n\nFull Text:\n{body}",
        }

    except Exception as e:
        print(f"  Fetch error for PMC{pmc_id}: {e}")
        return None


def fetch_papers(max_papers: int = 100, per_query: int = 8) -> list[dict]:
    """
    Fetch cardiovascular risk papers from PubMed Central.
    
    Args:
        max_papers: Maximum total number of papers to fetch
        per_query: Maximum papers per search query
        
    Returns:
        List of paper dictionaries with text and metadata
    """
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if we already have papers
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            existing = json.load(f)
        if len(existing) >= max_papers * 0.8:  # Allow 20% margin
            print(f"Already have {len(existing)} papers cached. Skipping fetch.")
            return existing

    papers = []
    seen_ids = set()

    print(f"Fetching up to {max_papers} papers from PubMed Central...")

    for i, query in enumerate(SEARCH_QUERIES):
        if len(papers) >= max_papers:
            break

        print(f"  [{i+1}/{len(SEARCH_QUERIES)}] Searching: {query[:60]}...")
        pmc_ids = search_pmc(query, max_results=per_query)

        for pmc_id in pmc_ids:
            if len(papers) >= max_papers:
                break
            if pmc_id in seen_ids:
                continue
            seen_ids.add(pmc_id)

            article = fetch_article_text(pmc_id)
            if article and len(article["text"]) > 500:  # Minimum content threshold
                papers.append(article)
                
                # Save individual paper
                paper_file = PAPERS_DIR / f"PMC{pmc_id}.txt"
                with open(paper_file, "w", encoding="utf-8") as f:
                    f.write(article["text"])

                print(f"    ✓ PMC{pmc_id}: {article['title'][:70]}...")

            # Be respectful of NCBI rate limits
            time.sleep(0.4)

        time.sleep(1)  # Pause between queries

    # Save metadata
    metadata = [
        {
            "pmc_id": p["pmc_id"],
            "pmid": p.get("pmid"),
            "title": p["title"],
            "authors": p["authors"],
            "journal": p["journal"],
            "year": p["year"],
            "text_length": len(p["text"]),
        }
        for p in papers
    ]
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Fetched {len(papers)} papers. Saved to {PAPERS_DIR}")
    return papers


if __name__ == "__main__":
    papers = fetch_papers(max_papers=100)
    print(f"\nTotal papers fetched: {len(papers)}")
    for p in papers[:5]:
        print(f"  - {p['title'][:80]}...")
