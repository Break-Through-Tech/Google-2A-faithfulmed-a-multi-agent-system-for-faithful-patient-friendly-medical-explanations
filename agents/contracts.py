"""Shared input contract and adapters for the FaithfulMed agent pipeline."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ClinicalAtom(BaseModel):
    id: str
    text: str
    type: str
    source_quote: str
    importance: str | None = None

    @field_validator("id", "text", "type", "source_quote")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Clinical atom fields cannot be empty")
        return value.strip()


class RetrievedPassage(BaseModel):
    text: str
    source: str

    @field_validator("text", "source")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Retrieved passage text and source are required")
        return value.strip()


class PipelineCase(BaseModel):
    """Canonical source and evidence state shared across pipeline agents."""

    source_text: str
    atoms: list[ClinicalAtom] = Field(default_factory=list)
    retrieved_context: list[RetrievedPassage] = Field(default_factory=list)
    protected_facts: list[str] = Field(default_factory=list)
    target_grade: int = Field(default=8, ge=1, le=12)

    @field_validator("source_text")
    @classmethod
    def require_source_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Source text cannot be empty")
        return value.strip()

    @classmethod
    def from_extractor_output(
        cls,
        source_text: str,
        extractor_output: dict[str, Any],
        target_grade: int = 8,
        retrieved_context: list[dict[str, str]] | None = None,
        protected_facts: list[str] | None = None,
    ) -> "PipelineCase":
        """Validate Extractor notebook output and assign stable per-case atom IDs."""
        raw_atoms = extractor_output.get("atoms")
        if not isinstance(raw_atoms, list):
            raise ValueError("Extractor output must contain an atoms list")

        atoms = []
        for index, raw_atom in enumerate(raw_atoms, start=1):
            if not isinstance(raw_atom, dict):
                raise ValueError(f"Extractor atom {index} must be an object")
            atom = ClinicalAtom(
                id=f"atom-{index:03d}",
                text=raw_atom.get("text", ""),
                type=raw_atom.get("type", ""),
                source_quote=raw_atom.get("source_quote", ""),
            )
            if " ".join(atom.source_quote.split()) not in " ".join(source_text.split()):
                raise ValueError(
                    f"Extractor atom {atom.id} source_quote is not present in source_text"
                )
            atoms.append(atom)

        return cls(
            source_text=source_text,
            atoms=atoms,
            retrieved_context=[
                RetrievedPassage.model_validate(passage)
                for passage in retrieved_context or []
            ],
            protected_facts=protected_facts or [],
            target_grade=target_grade,
        )

    def to_adk_state(self) -> dict[str, Any]:
        """Return state values used by Extractor, Simplifier, and Readability agents."""
        return {
            "source_text": self.source_text,
            "atoms": [atom.model_dump() for atom in self.atoms],
            "retrieved_context": [passage.model_dump() for passage in self.retrieved_context],
            "protected_facts": self.protected_facts,
            "target_grade": self.target_grade,
        }

    def verifier_atoms(self) -> list[dict[str, str | None]]:
        """Adapt canonical Extractor atoms to the existing Verifier schema."""
        return [
            {
                "id": atom.id,
                "concept": atom.text,
                "importance": atom.importance,
            }
            for atom in self.atoms
        ]

    def refiner_state(self, candidate_text: str, verifier_feedback: dict[str, Any]) -> dict[str, Any]:
        """Adapt shared state and candidate output to the existing Refiner contract."""
        if not candidate_text.strip():
            raise ValueError("Candidate text cannot be empty")
        return {
            "original_text": self.source_text,
            "simplified_text": candidate_text.strip(),
            "verifier_feedback": verifier_feedback,
        }
