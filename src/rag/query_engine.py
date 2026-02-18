"""
RAG query engine for cardiovascular risk research Q&A.

Takes a user question, retrieves relevant paper chunks from the
FAISS vector store, and generates an answer using Google Gemini
with proper source citations.
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from .vector_store import load_store, get_store_stats


# System prompt for the RAG assistant
SYSTEM_PROMPT = """You are a cardiovascular research assistant for the CardioRisk ML project. 
You answer questions about heart failure, cardiovascular risk prediction, and related clinical topics 
using evidence from a curated collection of peer-reviewed research papers.

IMPORTANT RULES:
1. Base your answers ONLY on the provided paper excerpts. Do not fabricate information.
2. Cite your sources using the paper title and PMC ID when referencing specific findings.
3. If the provided excerpts don't contain enough information to answer the question, say so explicitly.
4. Use clear, academic language appropriate for a clinical/ML audience.
5. When discussing statistical findings, include specific numbers when available.
6. Distinguish between strong evidence (multiple papers, large studies) and limited evidence (single small study).
7. Keep answers focused and well-structured with key points highlighted.
"""

# Example questions for the UI
EXAMPLE_QUESTIONS = [
    "What are the strongest predictors of heart failure mortality?",
    "How does ejection fraction relate to heart failure outcomes?",
    "What role does serum creatinine play in cardiovascular risk prediction?",
    "How do machine learning models compare to traditional risk scores for heart failure?",
    "What is data leakage and how does it affect clinical prediction models?",
    "What are the limitations of using small datasets for clinical ML?",
    "How does the cardiorenal syndrome affect heart failure prognosis?",
    "What ML approaches have been used for cardiovascular risk prediction?",
]


def _get_llm():
    """Get the Google Generative AI chat model."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.3,
        max_output_tokens=2048,
    )


def _format_context(docs: list[Document]) -> str:
    """Format retrieved documents into a context string."""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "Unknown")
        pmc_id = doc.metadata.get("pmc_id", "N/A")
        journal = doc.metadata.get("journal", "Unknown")
        year = doc.metadata.get("year", "Unknown")

        context_parts.append(
            f"--- Paper {i} ---\n"
            f"Title: {title}\n"
            f"Source: PMC{pmc_id} | {journal} ({year})\n"
            f"Excerpt:\n{doc.page_content}\n"
        )
    return "\n".join(context_parts)


def ask_question(question: str, k: int = 8) -> dict:
    """
    Answer a question using RAG from the cardiovascular papers.
    
    Args:
        question: The user's question
        k: Number of relevant paper chunks to retrieve
        
    Returns:
        Dict with 'answer', 'sources', and 'retrieved_docs' keys
    """
    # Load vector store and retrieve relevant chunks
    store = load_store()
    retrieved_docs = store.similarity_search(question, k=k)

    # Format context
    context = _format_context(retrieved_docs)

    # Build the prompt
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Here are relevant excerpts from cardiovascular research papers:\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"Please provide a thorough, evidence-based answer citing the specific papers above."
    )

    # Generate answer
    llm = _get_llm()
    response = llm.invoke(prompt)
    answer = response.content

    # Extract unique sources
    seen_sources = set()
    sources = []
    for doc in retrieved_docs:
        pmc_id = doc.metadata.get("pmc_id", "")
        if pmc_id not in seen_sources:
            seen_sources.add(pmc_id)
            sources.append({
                "title": doc.metadata.get("title", "Unknown"),
                "pmc_id": pmc_id,
                "journal": doc.metadata.get("journal", "Unknown"),
                "year": doc.metadata.get("year", "Unknown"),
                "authors": doc.metadata.get("authors", ""),
            })

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_docs": retrieved_docs,
    }


if __name__ == "__main__":
    print("CardioRisk RAG Query Engine")
    print("=" * 50)
    
    question = "What are the main risk factors for heart failure mortality?"
    print(f"\nQuestion: {question}\n")
    
    result = ask_question(question)
    
    print("Answer:")
    print(result["answer"])
    print(f"\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  - {s['title']} (PMC{s['pmc_id']})")
