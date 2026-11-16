import json
import os
import re
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple
 
import chromadb
from dotenv import load_dotenv
from google import genai
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.genai import types
from pydantic import BaseModel, Field
 
try:
    from .verifier import MODEL, make_verifier_input
except ImportError:  # running as a plain script from inside agents/
    from verifier import MODEL, make_verifier_input
 
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
 
CHROMA_DIR = str(REPO_ROOT / "chroma" / "medlineplus")
COLLECTION_NAME = "medlineplus_glossary_and_guidelines"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMS = 768  # index and queries must use the same dimensionality
 
RESULTS_PER_QUERY = 3
MAX_CONTEXT_CHUNKS = 8
MAX_QUERY_ATOMS = 6
FALLBACK_QUERY_CHARS = 1000
TEMPERATURE = 0.2
 
MAX_DISTANCE: Optional[float] = None
 

SOURCE_FILTER: Optional[str] = None
 
ATOMS_KEY = "clinical_atoms"
 
TRANSIENT = ("429", "500", "503", "RESOURCE_EXHAUSTED", "INTERNAL", "UNAVAILABLE")
 
 
class TermSource(BaseModel):
    term: str = Field(description="The medical term explained using a reference passage.")
    source_id: str = Field(description="ID of the passage used, e.g. 'S2'.")
 
 
class SimplifierOutput(BaseModel):
    simplified_text: str = Field(
        description="The patient-friendly rewrite, as plain text paragraphs (no markdown)."
    )
    definitions_used: List[TermSource] = Field(
        default_factory=list,
        description="One entry per medical term you explained using a reference passage. "
        "Empty if none.",
    )
 
 
_client: Optional[genai.Client] = None
_collection = None
 
 
def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY in your .env file (repo root).")
        _client = genai.Client(api_key=api_key)
    return _client
 
 
def _get_collection():
    global _collection
    if _collection is None:
        try:
            _collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(
                COLLECTION_NAME
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not open collection '{COLLECTION_NAME}' at {CHROMA_DIR}. "
                "Run agents/medlineplus_rag_index.ipynb first to build the index."
            ) from exc
    return _collection
 
 
def _with_retry(fn, tries: int = 4, label: str = "call"):
    """Retry transient API errors (429/500/503) with exponential backoff."""
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:
            if not any(k in str(exc) for k in TRANSIENT) or i == tries - 1:
                raise
            wait = 5 * 2**i
            print(f"[simplifier] {label} transient error, retrying in {wait}s "
                  f"({i + 1}/{tries})")
            time.sleep(wait)
 
 
def _embed_queries(queries: List[str]) -> List[List[float]]:
    """Embed all queries in ONE API call (task_type=RETRIEVAL_QUERY)."""
    response = _with_retry(
        lambda: _get_client().models.embed_content(
            model=EMBED_MODEL,
            contents=queries,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY", output_dimensionality=EMBED_DIMS
            ),
        ),
        label="embedding",
    )
    return [e.values for e in response.embeddings]
 
 
def retrieve_context(
    queries: List[str],
    results_per_query: int = RESULTS_PER_QUERY,
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    max_distance: Optional[float] = MAX_DISTANCE,
) -> List[dict]:
    """Retrieve passages for several queries, de-duplicated and interleaved so every
    query gets coverage before any gets a second result. Each passage gets a source_id
    (S1, S2, ...)."""
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        return []
 
    kwargs = {"query_embeddings": _embed_queries(queries), "n_results": results_per_query}
    if SOURCE_FILTER:
        kwargs["where"] = {"source_type": SOURCE_FILTER}
    res = _get_collection().query(**kwargs)
 
    chunks: List[dict] = []
    seen = set()
    for rank in range(results_per_query):
        for qi in range(len(queries)):
            ids = res["ids"][qi]
            if rank >= len(ids) or ids[rank] in seen:
                continue
            dist = res["distances"][qi][rank]
            if max_distance is not None and dist > max_distance:
                continue
            seen.add(ids[rank])
            meta = res["metadatas"][qi][rank] or {}
            chunks.append(
                {
                    "source_id": f"S{len(chunks) + 1}",
                    "chunk_id": ids[rank],
                    "title": meta.get("title", ""),
                    "source_type": meta.get("source_type", ""),
                    "url": meta.get("url", ""),
                    "text": res["documents"][qi][rank],
                    "distance": dist,
                }
            )
            if len(chunks) >= max_chunks:
                return chunks
    return chunks
 
 

def _coerce_atoms(raw: Any) -> List[dict]:
    """Accept atoms as a list, a JSON string, or a {"atoms": [...]} dict and normalise
    to [{"id", "concept", "importance"}] (the shape verifier.py wants)."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        raw = raw.get("clinical_atoms") or raw.get("atoms") or []
    if not isinstance(raw, list):
        return []
 
    atoms = []
    for i, item in enumerate(raw, start=1):
        if isinstance(item, str):
            concept, atom_id, importance = item, f"a{i}", None
        elif isinstance(item, dict):
            concept = item.get("concept") or item.get("text") or item.get("name")
            atom_id = item.get("id") or f"a{i}"
            importance = item.get("importance")
        else:
            continue
        if not concept:
            continue
        atoms.append(
            {
                "id": str(atom_id),
                "concept": str(concept),
                "importance": str(importance) if importance is not None else None,
            }
        )
    return atoms
 
 
def _grade(value: Any) -> int:
    try:
        return max(1, min(12, int(value)))
    except (TypeError, ValueError):
        return 8
 
 
def build_queries(original_text: str, atoms: List[dict]) -> List[str]:
    """One query per clinical atom. With no atoms, one query per sentence (a blended
    whole-note query lands between topics and retrieves poorly)."""
    concepts = [a["concept"] for a in atoms][:MAX_QUERY_ATOMS]
    if concepts:
        return concepts
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", original_text)
        if len(s.strip()) > 15
    ]
    return sentences[:MAX_QUERY_ATOMS] or [original_text[:FALLBACK_QUERY_CHARS]]
 
 
def format_context(chunks: List[dict]) -> str:
    if not chunks:
        return "(no reference passages retrieved)"
    return "\n\n".join(
        f"[{c['source_id']}] {c['title']} ({c['source_type']})\n{c['text']}" for c in chunks
    )
 
 
def _term_in_text(term: str, text: str) -> bool:
    term, text = term.lower().strip(), text.lower()
    if not term:
        return False
    if term in text:
        return True
    words = [w for w in re.findall(r"[a-z]+", term) if len(w) > 3]
    return bool(words) and all(w in text for w in words)
 
 
def verify_definitions(
    definitions: List[TermSource], chunks: List[dict]
) -> Tuple[List[dict], List[str]]:
    """Keep only claimed (term, source) pairs where the term really appears in that
    passage. Returns (verified definitions, verified source IDs in order)."""
    by_id = {c["source_id"]: c for c in chunks}
    verified, sources = [], []
    for d in definitions:
        chunk = by_id.get(d.source_id)
        if chunk and _term_in_text(d.term, chunk["title"] + " " + chunk["text"]):
            verified.append({"term": d.term, "source_id": d.source_id})
            if d.source_id not in sources:
                sources.append(d.source_id)
    return verified, sources
 
 
def build_instruction(
    original_text: str, atoms: List[dict], chunks: List[dict], target_grade_level: int
) -> str:
    atom_lines = (
        "\n".join(
            f"- [{a['id']}] {a['concept']}"
            + (f" (importance: {a['importance']})" if a.get("importance") else "")
            for a in atoms
        )
        or "(none supplied)"
    )
    return f"""You rewrite clinical text as a clear, patient-friendly explanation.
 
RULES
1. SOURCE TEXT (below) is the only source of patient facts. Everything you say about
   this patient must be stated in it.
2. REFERENCE PASSAGES (MedlinePlus, below) are only for explaining medical terms in plain
   words. Never add symptoms, diagnoses, causes, risks, treatments or advice from them
   unless the source text says the same thing. Ignore passages that are not relevant;
   if none fit a term, explain it from general knowledge in one plain, neutral phrase.
3. Keep exactly: certainty ("suspected" stays "suspected"), negation ("no fever" stays
   "no fever"), numbers, units, dates, durations, medication names and doses, and
   follow-up instructions.
4. Keep every clinically important point. The clinical atoms below are hints for what
   matters; check them against the source text and do not assume they are complete or
   correct.
5. Do not add advice, reassurance, warnings or next steps that are not in the source.
   Do not introduce people the source does not mention (do not write "your doctor"
   unless the source refers to a doctor). Address the patient as "you".
6. Write at about a grade {target_grade_level} reading level: short sentences, everyday
   words. Replace medical jargon with plain words. Keep a technical term only if the
   patient will hear it again (a drug or test name), and give its plain meaning in
   parentheses right after. Explain every diagnosis in plain words.
7. Output plain text paragraphs. No markdown, headings or bullet symbols.
8. In definitions_used, add one entry (term, source_id) for each term whose explanation
   you took from a reference passage. Do not list passages you did not use. Use an
   empty list if none.
 
A reviewer will check your text for faithfulness to the source, omissions, unsupported
additions, and reading level.
 
SOURCE TEXT
\"\"\"
{original_text}
\"\"\"
 
CLINICAL ATOMS
{atom_lines}
 
REFERENCE PASSAGES
{format_context(chunks)}
"""
 
 
def _gen_config(system_instruction: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        response_schema=SimplifierOutput,
    )
 
 
def simplify(
    original_text: str,
    clinical_atoms: Any = None,
    target_grade_level: int = 8,
    use_rag: bool = True,
) -> dict:
    """Run the Simplifier once. use_rag=False is the no-retrieval ablation variant.
 
    Returns simplified_text, sources_used (verified), definitions_used (verified),
    retrieved_chunks, and verifier_input (JSON string for the Verifier)."""
    atoms = _coerce_atoms(clinical_atoms)
    grade = _grade(target_grade_level)
    chunks = retrieve_context(build_queries(original_text, atoms)) if use_rag else []
 
    response = _with_retry(
        lambda: _get_client().models.generate_content(
            model=MODEL,
            contents="Write the patient-friendly version now.",
            config=_gen_config(build_instruction(original_text, atoms, chunks, grade)),
        ),
        label="generate_content",
    )
    out = SimplifierOutput.model_validate_json(response.text)
    definitions, sources = verify_definitions(out.definitions_used, chunks)
    return {
        "simplified_text": out.simplified_text,
        "sources_used": sources,
        "definitions_used": definitions,
        "retrieved_chunks": chunks,
        "verifier_input": make_verifier_input(
            original_text=original_text,
            simplified_text=out.simplified_text,
            clinical_atoms=atoms,
            target_grade_level=grade,
        ),
    }
 
 

def _user_text(ctx: CallbackContext) -> str:
    content = getattr(ctx, "user_content", None)
    if content and content.parts:
        return "\n".join(p.text for p in content.parts if getattr(p, "text", None)).strip()
    return ""
 
 
def retrieve_for_simplifier(callback_context: CallbackContext) -> Optional[types.Content]:
    """before_agent_callback: deterministic retrieval, outside the model."""
    state = callback_context.state
    original_text = state.get("original_text") or _user_text(callback_context)
    if not original_text:
        raise ValueError("Simplifier needs original_text in state or in the user message.")
    state["original_text"] = original_text
 
    atoms = _coerce_atoms(state.get(ATOMS_KEY))
    if state.get("use_rag", True):
        state["retrieved_chunks"] = retrieve_context(build_queries(original_text, atoms))
    else:
        state["retrieved_chunks"] = []
    return None
 
 
def simplifier_instruction(ctx: ReadonlyContext) -> str:
    """Instruction provider; building the prompt ourselves means braces in clinical text
    can never break ADK's state templating."""
    s = ctx.state
    return build_instruction(
        original_text=s.get("original_text", ""),
        atoms=_coerce_atoms(s.get(ATOMS_KEY)),
        chunks=s.get("retrieved_chunks") or [],
        target_grade_level=_grade(s.get("target_grade_level")),
    )
 
 
def prepare_verifier_input(callback_context: CallbackContext) -> Optional[types.Content]:
    """after_agent_callback: verify definitions and build the Verifier's input JSON."""
    state = callback_context.state
    out = state.get("simplified_output")
    if isinstance(out, str):
        out = json.loads(out)
    if not out or not out.get("simplified_text"):
        raise RuntimeError("Simplifier produced no simplified_text.")
 
    defs = [TermSource(**d) for d in out.get("definitions_used", [])]
    state["definitions_verified"], state["sources_used"] = verify_definitions(
        defs, state.get("retrieved_chunks") or []
    )
    state["verifier_input"] = make_verifier_input(
        original_text=state.get("original_text", ""),
        simplified_text=out["simplified_text"],
        clinical_atoms=_coerce_atoms(state.get(ATOMS_KEY)),
        target_grade_level=_grade(state.get("target_grade_level")),
    )
    return None
 
 
simplifier_agent = LlmAgent(
    name="simplifier_agent",
    model=MODEL,
    description="Rewrites clinical text in plain language, grounded in MedlinePlus.",
    instruction=simplifier_instruction,
    generate_content_config=types.GenerateContentConfig(temperature=TEMPERATURE),
    output_schema=SimplifierOutput,
    output_key="simplified_output",
    before_agent_callback=retrieve_for_simplifier,
    after_agent_callback=prepare_verifier_input,
)
 
 
if __name__ == "__main__":
    demo_note = (
        "The patient reports a dry cough for three days. No fever or shortness of breath "
        "is reported. A viral upper respiratory infection is suspected. The plan is rest, "
        "fluids, and follow-up if symptoms worsen or persist."
    )
    result = simplify(demo_note)
    print("RETRIEVED:")
    for c in result["retrieved_chunks"]:
        print(f"  [{c['source_id']}] {c['title']} ({c['source_type']}, d={c['distance']:.3f})")
    print("\nSIMPLIFIED:\n" + result["simplified_text"])
    print("\nVERIFIED DEFINITIONS:", result["definitions_used"])
    print("SOURCES USED:", result["sources_used"])
 