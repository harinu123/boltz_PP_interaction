"""Lightweight data structures mirroring the Boltz hackathon template."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional


class TaskType(str, Enum):
    """Enumeration of supported hackathon task types."""

    PROTEIN_COMPLEX = "protein_complex"
    PROTEIN_LIGAND = "protein_ligand"


@dataclass
class Protein:
    """Protein sequence metadata used by hackathon prediction scripts."""

    id: str
    sequence: str
    msa: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Protein":
        return cls(
            id=str(data["id"]),
            sequence=str(data["sequence"]),
            msa=data.get("msa"),
        )


@dataclass
class SmallMolecule:
    """Ligand metadata for protein–ligand tasks."""

    id: str
    smiles: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SmallMolecule":
        return cls(id=str(data["id"]), smiles=data.get("smiles"))


@dataclass
class Datapoint:
    """Description of a single hackathon datapoint."""

    datapoint_id: str
    task_type: TaskType
    proteins: list[Protein]
    ligands: Optional[list[SmallMolecule]] = None
    ground_truth: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Datapoint":
        task = data.get("task_type")
        try:
            task_type = TaskType(task)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported task_type: {task!r}") from exc

        proteins_raw: Iterable[dict[str, Any]] = data.get("proteins", [])
        proteins = [Protein.from_dict(p) for p in proteins_raw]

        ligands_raw = data.get("ligands")
        ligands = (
            [SmallMolecule.from_dict(l) for l in ligands_raw]
            if ligands_raw
            else None
        )

        ground_truth = data.get("ground_truth")

        return cls(
            datapoint_id=str(data["datapoint_id"]),
            task_type=task_type,
            proteins=proteins,
            ligands=ligands,
            ground_truth=ground_truth,
        )

    @classmethod
    def from_json(cls, payload: str) -> "Datapoint":
        """Parse a JSON string describing a datapoint."""

        data = json.loads(payload)
        if not isinstance(data, dict):  # pragma: no cover - defensive
            raise TypeError("Expected datapoint JSON to decode into a dict")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the datapoint."""

        data: dict[str, Any] = {
            "datapoint_id": self.datapoint_id,
            "task_type": self.task_type.value,
            "proteins": [protein.__dict__ for protein in self.proteins],
        }
        if self.ligands is not None:
            data["ligands"] = [ligand.__dict__ for ligand in self.ligands]
        if self.ground_truth is not None:
            data["ground_truth"] = self.ground_truth
        return data

    def to_json(self) -> str:
        """Serialise the datapoint to a JSON string."""

        return json.dumps(self.to_dict())


__all__ = [
    "Datapoint",
    "Protein",
    "SmallMolecule",
    "TaskType",
]

