"""
Build and manage a FAISS vector store from cardiovascular risk papers.

Chunks paper text, embeds with Google's embedding model, and stores
in a local FAISS index for fast similarity search.
"""

import os
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Paths
PAPERS_DIR = Path(__file__).parent.parent.parent / "data" / "papers"
METADATA_FILE = PAPERS_DIR / "metadata.json"
VECTOR_STORE_DIR = Path(__file__).parent.parent.parent / "data" / "vector_store"


def _get_embeddings():
    """Get the Google Generative AI embeddings model."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )


def _load_papers() -> list[Document]:
    """Load paper text files and create LangChain Documents with metadata."""
    documents = []

    # Load metadata for enriching documents
    metadata_lookup = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            for entry in json.load(f):
                metadata_lookup[entry["pmc_id"]] = entry

    # Load text files
    if not PAPERS_DIR.exists():
        raise FileNotFoundError(
            f"Papers directory not found: {PAPERS_DIR}\n"
            "Run the paper fetcher first: python -m src.rag.paper_fetcher"
        )

    txt_files = list(PAPERS_DIR.glob("PMC*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No paper files found in {PAPERS_DIR}\n"
            "Run the paper fetcher first: python -m src.rag.paper_fetcher"
        )

    for txt_file in txt_files:
        pmc_id = txt_file.stem.replace("PMC", "")
        text = txt_file.read_text(encoding="utf-8")

        meta = metadata_lookup.get(pmc_id, {})
        doc_metadata = {
            "pmc_id": pmc_id,
            "title": meta.get("title", "Unknown"),
            "authors": ", ".join(meta.get("authors", [])),
            "journal": meta.get("journal", "Unknown"),
            "year": meta.get("year", "Unknown"),
            "source": f"PMC{pmc_id}",
        }

        documents.append(Document(page_content=text, metadata=doc_metadata))

    return documents


def _chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # ~375 tokens
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        # Propagate metadata to all chunks
        for chunk in doc_chunks:
            chunk.metadata = {**doc.metadata}
        chunks.extend(doc_chunks)

    return chunks


def build_store(max_papers: int | None = None) -> FAISS:
    """
    Build the FAISS vector store from downloaded papers.
    
    Args:
        max_papers: Optional limit on number of papers to index (for testing)
        
    Returns:
        The built FAISS vector store
    """
    print("Loading papers...")
    documents = _load_papers()
    if max_papers:
        documents = documents[:max_papers]
    print(f"  Loaded {len(documents)} papers")

    print("Chunking documents...")
    chunks = _chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    print("Building embeddings and FAISS index...")
    embeddings = _get_embeddings()

    # Process in batches to respect rate limits
    batch_size = 50
    store = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} ({len(batch)} chunks)...")

        if store is None:
            store = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            store.merge_from(batch_store)

    # Save to disk
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VECTOR_STORE_DIR))
    print(f"  Saved vector store to {VECTOR_STORE_DIR}")

    # Save stats
    stats = {
        "num_papers": len(documents),
        "num_chunks": len(chunks),
        "papers": [d.metadata.get("title", "Unknown") for d in documents],
    }
    stats_file = VECTOR_STORE_DIR / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    return store


def load_store() -> FAISS:
    """Load a previously built FAISS vector store from disk."""
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_STORE_DIR}\n"
            "Build it first: python -m src.rag.vector_store"
        )

    embeddings = _get_embeddings()
    store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return store


def get_store_stats() -> dict:
    """Get statistics about the current vector store."""
    stats_file = VECTOR_STORE_DIR / "stats.json"
    if stats_file.exists():
        with open(stats_file, "r") as f:
            return json.load(f)
    return {"num_papers": 0, "num_chunks": 0, "papers": []}


if __name__ == "__main__":
    store = build_store()
    print("\nTesting similarity search...")
    results = store.similarity_search("heart failure prediction ejection fraction", k=3)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('title', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")
