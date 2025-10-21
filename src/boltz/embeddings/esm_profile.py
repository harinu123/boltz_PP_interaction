"""Utilities for generating ESM-backed residue probability profiles."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from boltz.data import const


class ESMProfileGenerator:
    """Generate per-residue amino-acid distributions using an ESM model."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        cache_dir: Path | str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - runtime dependency
            msg = (
                "transformers is required to compute ESM residue profiles. "
                "Install it with `pip install transformers`."
            )
            raise ImportError(msg) from exc

        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path("./esm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self._cache: Dict[str, np.ndarray] = {}
        self._token_id_map = self._build_token_id_map()

    def _build_token_id_map(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for letter in const.prot_letter_to_token:
            try:
                token_id = int(self.tokenizer.convert_tokens_to_ids(letter))
            except KeyError:
                continue
            if token_id == self.tokenizer.unk_token_id:
                continue
            mapping[letter] = token_id
        return mapping

    @staticmethod
    def _sequence_key(sequence: str) -> str:
        return hashlib.sha1(sequence.encode("utf-8")).hexdigest()

    def _cache_path(self, sequence: str) -> Path:
        return self.cache_dir / f"{self._sequence_key(sequence)}.npy"

    def _profile_from_cache(self, sequence: str) -> np.ndarray | None:
        key = self._sequence_key(sequence)
        if key in self._cache:
            return self._cache[key]

        path = self._cache_path(sequence)
        if path.exists():
            profile = np.load(path)
            self._cache[key] = profile
            return profile
        return None

    def get_profile(self, sequence: str) -> np.ndarray:
        """Return a per-residue distribution over Boltz amino-acid tokens."""

        sequence = sequence.upper()
        if not sequence:
            return np.zeros((0, const.num_tokens), dtype=np.float32)

        cached = self._profile_from_cache(sequence)
        if cached is not None:
            return cached

        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        logits = logits[1:-1]  # drop BOS/EOS
        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

        profile = np.zeros((len(sequence), const.num_tokens), dtype=np.float32)
        for letter, token_name in const.prot_letter_to_token.items():
            token_id = self._token_id_map.get(letter)
            if token_id is None:
                continue
            profile[:, const.token_ids[token_name]] = probs[:, token_id]

        for idx, letter in enumerate(sequence):
            if np.sum(profile[idx]) <= 0:
                token_name = const.prot_letter_to_token.get(letter, "UNK")
                profile[idx, const.token_ids[token_name]] = 1.0

        path = self._cache_path(sequence)
        np.save(path, profile)
        self._cache[self._sequence_key(sequence)] = profile
        return profile

    def build_profile_map(self, tokenized) -> Dict[Tuple[int, int], np.ndarray]:
        """Compute per-residue profiles for the full tokenized structure."""

        structure = tokenized.structure
        profile_map: Dict[Tuple[int, int], np.ndarray] = {}
        for chain in structure.chains:
            chain_id = int(chain["asym_id"])
            mol_type = int(chain["mol_type"])
            if mol_type != const.chain_type_ids["PROTEIN"]:
                continue

            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])
            residues = structure.residues[res_start:res_end]
            letters = []
            for residue in residues:
                res_name = str(residue["name"]).strip().upper()
                if len(res_name) == 1:
                    letter = res_name
                else:
                    letter = const.prot_token_to_letter.get(res_name, "X")
                letters.append(letter)

            sequence = "".join(letters)
            if not sequence:
                continue

            profile = self.get_profile(sequence)
            for offset, residue in enumerate(residues):
                res_idx = int(residue["res_idx"])
                profile_map[(chain_id, res_idx)] = profile[offset]

        return profile_map

    def profile_for_tokens(
        self, tokenized, profile_map: Dict[Tuple[int, int], np.ndarray]
    ) -> np.ndarray:
        """Project pre-computed residue profiles to the cropped token set."""

        tokens = tokenized.tokens
        num_tokens = len(tokens)
        profiles = np.zeros((num_tokens, const.num_tokens), dtype=np.float32)

        for i, token in enumerate(tokens):
            mol_type = int(token["mol_type"])
            if mol_type != const.chain_type_ids["PROTEIN"]:
                continue

            chain_id = int(token["asym_id"])
            res_idx = int(token["res_idx"])
            key = (chain_id, res_idx)
            if key in profile_map:
                profiles[i] = profile_map[key]
            else:
                res_type = int(token["res_type"])
                if 0 <= res_type < const.num_tokens:
                    profiles[i, res_type] = 1.0
        return profiles
