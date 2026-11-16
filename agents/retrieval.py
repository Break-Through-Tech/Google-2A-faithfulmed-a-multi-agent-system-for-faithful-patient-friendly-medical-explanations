"""Optional query adapter for the MedlinePlus Chroma index notebook."""

import os
from pathlib import Path
from typing import Any

import chromadb
from google import genai
from google.genai import types


INDEX_DIR = Path(__file__).resolve().parents[1] / "chroma" / "medlineplus"
COLLECTION_NAME = "medlineplus_glossary_and_guidelines"
EMBED_MODEL = "gemini-embedding-001"


def search_medlineplus(query: str, top_k: int = 3) -> dict[str, Any]:
    """Search the optional local MedlinePlus glossary and guideline index.

    The index must first be built with the project MedlinePlus RAG notebook.
    """
    if not query.strip():
        raise ValueError("Retrieval query cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least one.")
    if not INDEX_DIR.is_dir():
        return {
            "status": "index_not_built",
            "message": (
                "The local MedlinePlus index is not built. Run "
                "agents/medlineplus_rag_index.ipynb before using retrieval."
            ),
            "passages": [],
        }

    database = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = database.get_collection(COLLECTION_NAME)
    collection_size = collection.count()
    if collection_size == 0:
        return {
            "status": "index_empty",
            "message": "The MedlinePlus collection exists but contains no records.",
            "passages": [],
        }

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY to query the MedlinePlus index."
        )
    client = genai.Client(api_key=api_key)
    embedding_response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[types.Content(parts=[types.Part(text=query)])],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    if embedding_response is None or not embedding_response.embeddings:
        raise RuntimeError("Gemini returned no embedding for the retrieval query.")
    query_embedding = embedding_response.embeddings[0].values
    if query_embedding is None:
        raise RuntimeError("Gemini returned an embedding without vector values.")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection_size),
    )
    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []
    passages = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        passages.append(
            {
                "text": document,
                "source_type": metadata.get("source_type", ""),
                "title": metadata.get("title", ""),
                "url": metadata.get("url", ""),
                "distance": distance,
            }
        )
    return {
        "status": "available",
        "message": f"Retrieved {len(passages)} MedlinePlus passages.",
        "passages": passages,
    }
