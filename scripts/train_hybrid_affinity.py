#!/usr/bin/env python3
"""Fine-tune a hybrid Boltz2 + ESM2 model for antibody-antigen affinity."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

import requests

from boltz.main import predict as boltz_predict

LOGGER = logging.getLogger("boltz.hybrid_affinity")

# Shared regex utilities
_CHAIN_DELIMITER = re.compile(r"[\s|,:;/]+")
_VALID_PROTEIN_TOKENS = set("ACDEFGHIKLMNPQRSTVWYBXZOU-")

_SABDAB_DATA_ID = 4167357
_SABDAB_DATA_URL = f"https://dataverse.harvard.edu/api/access/datafile/{_SABDAB_DATA_ID}"
_SABDAB_FILENAME = "protein_sabdab.csv"
_DEFAULT_DATA_DIR = Path(os.environ.get("HYBRID_DATA_DIR", "./data"))


@dataclass
class InteractionExample:
    """Container for a single antibody-antigen interaction example."""

    identifier: str
    antibody_heavy: str
    antibody_light: str
    antigen: str
    log_affinity: float
    split: str


def download_sabdab_dataset(data_dir: Path) -> Path:
    """Download the Protein_SAbDab CSV if it is not already cached."""

    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / _SABDAB_FILENAME
    if destination.exists():
        return destination

    tmp_path = destination.with_suffix(".tmp")
    LOGGER.info("Downloading Protein_SAbDab dataset to %s", destination)

    with requests.get(_SABDAB_DATA_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)

    tmp_path.replace(destination)
    return destination


def _load_sabdab_frame(csv_path: Path) -> pd.DataFrame:
    """Load the raw Protein_SAbDab CSV into a cleaned DataFrame."""

    frame = pd.read_csv(csv_path)
    unnamed_cols = [col for col in frame.columns if str(col).startswith("Unnamed")]  # pandas index column
    if unnamed_cols:
        frame = frame.drop(columns=unnamed_cols)
    return frame


def _split_indices(size: int, fractions: tuple[float, float, float], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random train/validation/test splits."""

    if size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    indices = np.arange(size)
    rng.shuffle(indices)

    train_end = int(fractions[0] * size)
    valid_end = train_end + int(fractions[1] * size)
    train_idx = indices[:train_end]
    valid_idx = indices[train_end:valid_end]
    test_idx = indices[valid_end:]
    return train_idx, valid_idx, test_idx


def _partition_frame(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Partition the Protein_SAbDab dataframe into train/valid/test splits."""

    train_idx, valid_idx, test_idx = _split_indices(len(frame), (0.7, 0.1, 0.2), seed)
    splits: dict[str, pd.DataFrame] = {}
    splits["train"] = frame.iloc[train_idx].reset_index(drop=True)
    splits["valid"] = frame.iloc[valid_idx].reset_index(drop=True)
    splits["test"] = frame.iloc[test_idx].reset_index(drop=True)
    return splits


def _parse_antibody_entry(entry: str) -> tuple[str, str]:
    """Parse a raw antibody column entry into heavy and light chain sequences."""

    if not entry or entry.lower() in {"nan", "none"}:
        return "", ""

    try:
        parsed = ast.literal_eval(entry)
    except (SyntaxError, ValueError):
        return split_antibody_sequence(entry)

    if isinstance(parsed, (list, tuple)):
        heavy = sanitize_sequence(str(parsed[0])) if parsed else ""
        light = sanitize_sequence(str(parsed[1])) if len(parsed) > 1 else ""
        return heavy, light

    return split_antibody_sequence(entry)


class ESMEmbedder:
    """Compute mean pooled embeddings from an ESM2 checkpoint."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        device: torch.device,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - handled at runtime
            msg = (
                "transformers is required for the hybrid affinity training script. "
                "Please install it with `pip install transformers`."
            )
            raise ImportError(msg) from exc

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        self._embedding_cache: dict[str, np.ndarray] = {}

    def _cache_path(self, sequence: str) -> Path:
        digest = hashlib.sha1(sequence.encode("utf-8"), usedforsecurity=False).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def embed(self, sequence: str) -> np.ndarray:
        """Embed a sequence using the frozen ESM model."""

        if not sequence:
            return np.zeros(self.model.config.hidden_size, dtype=np.float32)

        if sequence in self._embedding_cache:
            return self._embedding_cache[sequence]

        cache_path = self._cache_path(sequence)
        if cache_path.exists():
            embedding = np.load(cache_path)
            self._embedding_cache[sequence] = embedding
            return embedding

        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            padding=False,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state.squeeze(0)
            mask = inputs["attention_mask"].squeeze(0).bool()
            embedding = hidden[mask].mean(dim=0).cpu().numpy().astype(np.float32)

        np.save(cache_path, embedding)
        self._embedding_cache[sequence] = embedding
        return embedding


class BoltzAffinityPredictor:
    """Utility to run Boltz2 affinity predictions for protein complexes."""

    def __init__(
        self,
        cache_dir: Path,
        work_dir: Path,
        accelerator: str,
        devices: int,
        override: bool,
        sampling_steps: int,
        diffusion_samples: int,
        sampling_steps_affinity: int,
        diffusion_samples_affinity: int,
        max_parallel_samples: int,
        checkpoint: Optional[str],
        affinity_checkpoint: Optional[str],
        no_kernels: bool,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = work_dir
        self.accelerator = accelerator
        self.devices = devices
        self.override = override
        self.sampling_steps = sampling_steps
        self.diffusion_samples = diffusion_samples
        self.sampling_steps_affinity = sampling_steps_affinity
        self.diffusion_samples_affinity = diffusion_samples_affinity
        self.max_parallel_samples = max_parallel_samples
        self.checkpoint = checkpoint
        self.affinity_checkpoint = affinity_checkpoint
        self.no_kernels = no_kernels

        self.input_dir = self.work_dir / "boltz_inputs"
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def _yaml_path(self, identifier: str) -> Path:
        return self.input_dir / f"{identifier}.yaml"

    def _prediction_dir(self) -> Path:
        stem = self.input_dir.name
        return self.work_dir / f"boltz_results_{stem}" / "predictions"

    def _prediction_file(self, identifier: str) -> Path:
        return self._prediction_dir() / identifier / f"affinity_{identifier}.json"

    def _build_yaml(self, example: InteractionExample) -> dict:
        sequences = []
        chain_ids = []
        if example.antibody_heavy:
            sequences.append(
                {
                    "protein": {
                        "id": "H",
                        "sequence": example.antibody_heavy,
                        "msa": "empty",
                    }
                }
            )
            chain_ids.append("H")
        if example.antibody_light:
            sequences.append(
                {
                    "protein": {
                        "id": "L",
                        "sequence": example.antibody_light,
                        "msa": "empty",
                    }
                }
            )
            chain_ids.append("L")

        sequences.append(
            {
                "protein": {
                    "id": "ANT",
                    "sequence": example.antigen,
                    "msa": "empty",
                }
            }
        )

        properties = [
            {
                "affinity": {
                    "binder": "ANT",
                }
            }
        ]

        return {"sequences": sequences, "properties": properties}

    def prepare_inputs(self, examples: Iterable[InteractionExample]) -> list[str]:
        missing_predictions: list[str] = []
        for example in examples:
            yaml_path = self._yaml_path(example.identifier)
            if self.override or not yaml_path.exists():
                config = self._build_yaml(example)
                with yaml_path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(config, handle, sort_keys=False)

            prediction_file = self._prediction_file(example.identifier)
            if self.override or not prediction_file.exists():
                missing_predictions.append(example.identifier)

        return missing_predictions

    def run(self, examples: Iterable[InteractionExample]) -> None:
        missing = self.prepare_inputs(examples)
        if not missing and not self.override:
            LOGGER.info("All Boltz predictions already available; skipping inference.")
            return

        LOGGER.info("Running Boltz2 affinity predictions for %s complexes", len(missing))
        predict_fn = getattr(boltz_predict, "callback", boltz_predict)
        predict_fn(
            data=str(self.input_dir),
            out_dir=str(self.work_dir),
            cache=str(self.cache_dir),
            checkpoint=self.checkpoint,
            affinity_checkpoint=self.affinity_checkpoint,
            devices=self.devices,
            accelerator=self.accelerator,
            recycling_steps=1,
            sampling_steps=self.sampling_steps,
            diffusion_samples=self.diffusion_samples,
            sampling_steps_affinity=self.sampling_steps_affinity,
            diffusion_samples_affinity=self.diffusion_samples_affinity,
            max_parallel_samples=self.max_parallel_samples,
            step_scale=None,
            use_msa_server=False,
            subsample_msa=False,
            max_msa_seqs=1,
            override=self.override,
            model="boltz2",
            affinity_mw_correction=True,
            no_kernels=self.no_kernels,
        )

    def collect_features(self) -> dict[str, dict[str, float]]:
        predictions_dir = self._prediction_dir()
        if not predictions_dir.exists():
            msg = (
                "Boltz prediction directory not found. Ensure predictions were "
                "generated before collecting features."
            )
            raise FileNotFoundError(msg)

        outputs: dict[str, dict[str, float]] = {}
        for sample_dir in predictions_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            identifier = sample_dir.name
            affinity_file = sample_dir / f"affinity_{identifier}.json"
            if not affinity_file.exists():
                LOGGER.warning("Missing affinity output for %s", identifier)
                continue
            with affinity_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            outputs[identifier] = {
                "affinity_pred_value": float(data.get("affinity_pred_value", 0.0)),
                "affinity_probability_binary": float(
                    data.get("affinity_probability_binary", 0.0)
                ),
                "affinity_pred_value1": float(data.get("affinity_pred_value1", 0.0)),
                "affinity_pred_value2": float(data.get("affinity_pred_value2", 0.0)),
            }
        return outputs


class FeatureDataset(Dataset):
    """Simple tensor dataset wrapper."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class HybridAffinityRegressor(nn.Module):
    """Two-layer perceptron for affinity regression."""

    def __init__(self, input_dim: int, hidden_dim: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(-1)


def sanitize_sequence(sequence: str) -> str:
    sequence = sequence.strip().upper()
    return "".join(token for token in sequence if token in _VALID_PROTEIN_TOKENS)


def split_antibody_sequence(sequence: str) -> tuple[str, str]:
    sequence = sequence.strip()
    if not sequence:
        return "", ""

    parts = [part for part in _CHAIN_DELIMITER.split(sequence) if part]
    cleaned: list[str] = []
    for part in parts:
        if ":" in part and len(part) > 2 and part[1] == ":":
            part = part.split(":", maxsplit=1)[1]
        cleaned.append(sanitize_sequence(part))

    if not cleaned:
        return "", ""
    if len(cleaned) == 1:
        return cleaned[0], ""
    return cleaned[0], cleaned[1]


def resolve_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    msg = f"Could not locate any of the columns {list(candidates)} in dataframe."
    raise KeyError(msg)


def compute_log_affinity(value: float, already_log: bool) -> float:
    if already_log:
        return float(value)
    clipped = max(float(value), 1e-8)
    return float(math.log10(clipped))


def load_dataset(
    data_dir: Path = _DEFAULT_DATA_DIR,
    min_antigen_length: Optional[int] = None,
) -> dict[str, list[InteractionExample]]:
    csv_path = download_sabdab_dataset(data_dir)
    frame = _load_sabdab_frame(csv_path)

    if min_antigen_length is not None and min_antigen_length > 0:
        antigen_col = resolve_column(frame, ["X2", "Antigen", "antigen", "antigen_seq"])
        lengths = frame[antigen_col].map(lambda entry: len(sanitize_sequence(str(entry))))
        before = len(frame)
        mask = lengths >= int(min_antigen_length)
        frame = frame[mask].reset_index(drop=True)
        LOGGER.info(
            "Filtered Protein_SAbDab to antigen length >= %d: %d -> %d entries",
            min_antigen_length,
            before,
            len(frame),
        )

    splits = _partition_frame(frame)

    processed: dict[str, list[InteractionExample]] = {}
    for split_name, split_frame in splits.items():
        antibody_col = resolve_column(split_frame, ["X1", "Antibody", "antibody", "antibody_seq"])
        antigen_col = resolve_column(split_frame, ["X2", "Antigen", "antigen", "antigen_seq"])
        target_col = resolve_column(
            split_frame,
            [
                "Y",
                "Affinity",
                "affinity",
                "label",
                "Kd",
                "kd",
                "pKd",
                "pkd",
                "pAffinity",
                "paffinity",
            ],
        )

        id_column = None
        for candidate in ("ID", "id", "ID1", "id1", "complex_id", "pdb_id", "Index"):
            if candidate in split_frame.columns:
                id_column = candidate
                break

        examples: list[InteractionExample] = []
        for row_idx, row in split_frame.iterrows():
            if id_column:
                identifier = str(row[id_column])
            else:
                antibody_id = str(row.get("ID1", "")).strip()
                antigen_id = str(row.get("ID2", "")).strip()
                if antibody_id or antigen_id:
                    identifier = f"{antibody_id}_{antigen_id}".strip("_")
                else:
                    identifier = f"{split_name}_{row_idx}"

            heavy, light = _parse_antibody_entry(str(row[antibody_col]))
            antigen = sanitize_sequence(str(row[antigen_col]))
            log_affinity = compute_log_affinity(float(row[target_col]), already_log=False)

            examples.append(
                InteractionExample(
                    identifier=identifier,
                    antibody_heavy=heavy,
                    antibody_light=light,
                    antigen=antigen,
                    log_affinity=log_affinity,
                    split=split_name,
                )
            )

        processed[split_name] = examples

    return processed


def prepare_embeddings(
    embedder: ESMEmbedder, examples: Iterable[InteractionExample]
) -> dict[str, np.ndarray]:
    unique_sequences: set[str] = set()
    for example in examples:
        if example.antibody_heavy:
            unique_sequences.add(example.antibody_heavy)
        if example.antibody_light:
            unique_sequences.add(example.antibody_light)
        unique_sequences.add(example.antigen)

    embeddings: dict[str, np.ndarray] = {}
    for sequence in sorted(unique_sequences):
        embeddings[sequence] = embedder.embed(sequence)
    return embeddings


def build_feature_matrices(
    examples_by_split: dict[str, list[InteractionExample]],
    embeddings: dict[str, np.ndarray],
    boltz_outputs: dict[str, dict[str, float]],
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    matrices: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}

    # Determine embedding dimension
    example_embedding = next(iter(embeddings.values()))
    embed_dim = example_embedding.shape[0]

    for split_name, examples in examples_by_split.items():
        features: list[np.ndarray] = []
        targets: list[float] = []
        identifiers: list[str] = []

        for example in examples:
            heavy_vec = embeddings.get(example.antibody_heavy)
            if heavy_vec is None:
                heavy_vec = np.zeros(embed_dim, dtype=np.float32)
            light_vec = embeddings.get(example.antibody_light)
            if light_vec is None:
                light_vec = np.zeros(embed_dim, dtype=np.float32)
            antigen_vec = embeddings.get(example.antigen)
            if antigen_vec is None:
                antigen_vec = np.zeros(embed_dim, dtype=np.float32)

            if example.antibody_light:
                antibody_vec = 0.5 * (heavy_vec + light_vec)
            else:
                antibody_vec = heavy_vec

            boltz_feature = boltz_outputs.get(example.identifier, {})
            boltz_values = np.array(
                [
                    boltz_feature.get("affinity_pred_value", 0.0),
                    boltz_feature.get("affinity_probability_binary", 0.0),
                    boltz_feature.get("affinity_pred_value1", 0.0),
                    boltz_feature.get("affinity_pred_value2", 0.0),
                ],
                dtype=np.float32,
            )

            length_features = np.array(
                [
                    float(len(example.antibody_heavy)),
                    float(len(example.antibody_light)),
                    float(len(example.antigen)),
                ],
                dtype=np.float32,
            )

            combined = np.concatenate((antibody_vec, antigen_vec, length_features, boltz_values))
            features.append(combined)
            targets.append(example.log_affinity)
            identifiers.append(example.identifier)

        if not features:
            matrices[split_name] = (np.zeros((0, embed_dim * 2 + 7), dtype=np.float32), np.zeros(0), [])
            continue

        matrices[split_name] = (
            np.stack(features, axis=0).astype(np.float32),
            np.asarray(targets, dtype=np.float32),
            identifiers,
        )

    return matrices


def normalize_features(
    train_features: np.ndarray,
    matrices: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    epsilon = 1e-6
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0) + epsilon
    normalized: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for split_name, (features, targets, identifiers) in matrices.items():
        if features.size == 0:
            normalized[split_name] = (features, targets, identifiers)
            continue
        normalized_features = (features - mean) / std
        normalized[split_name] = (normalized_features, targets, identifiers)
    return normalized, mean, std


def train_model(
    model: HybridAffinityRegressor,
    train_data: FeatureDataset,
    val_data: FeatureDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> HybridAffinityRegressor:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_state: Optional[dict[str, torch.Tensor]] = None
    best_val = float("inf")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_features.size(0)

        train_loss = running_loss / max(len(train_data), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                predictions = model(batch_features)
                loss = criterion(predictions, batch_targets)
                val_loss += loss.item() * batch_features.size(0)
        val_loss /= max(len(val_data), 1)
        scheduler.step(val_loss)

        LOGGER.info(
            "Epoch %d/%d - train_loss=%.4f val_loss=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_dataset(model: nn.Module, dataset: FeatureDataset, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=256)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for features, _ in loader:
            outputs = model(features.to(device)).cpu().numpy()
            predictions.append(outputs)
    if not predictions:
        return np.zeros(0)
    return np.concatenate(predictions, axis=0)


def compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    if targets.size == 0:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan"), "pearson": float("nan")}
    residuals = predictions - targets
    mse = float(np.mean(residuals**2))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(mse))
    if targets.size > 1 and np.std(targets) > 1e-8 and np.std(predictions) > 1e-8:
        pearson = float(np.corrcoef(targets, predictions)[0, 1])
    else:
        pearson = float("nan")
    return {"mse": mse, "mae": mae, "rmse": rmse, "pearson": pearson}


def export_predictions(
    output_dir: Path,
    split_name: str,
    identifiers: list[str],
    targets: np.ndarray,
    predictions: np.ndarray,
) -> None:
    frame = pd.DataFrame(
        {
            "identifier": identifiers,
            "log_affinity": targets,
            "prediction": predictions,
        }
    )
    frame.to_csv(output_dir / f"predictions_{split_name}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("./hybrid_affinity_output"))
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")))
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--embedding-cache", type=Path, default=Path("./embedding_cache"))
    parser.add_argument("--esm-model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu"]) 
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--boltz-override", action="store_true")
    parser.add_argument("--boltz-sampling-steps", type=int, default=60)
    parser.add_argument("--boltz-diffusion-samples", type=int, default=1)
    parser.add_argument("--boltz-sampling-steps-affinity", type=int, default=120)
    parser.add_argument("--boltz-diffusion-samples-affinity", type=int, default=3)
    parser.add_argument("--boltz-max-parallel", type=int, default=2)
    parser.add_argument("--boltz-checkpoint", type=str, default=None)
    parser.add_argument("--boltz-affinity-checkpoint", type=str, default=None)
    parser.add_argument("--boltz-no-kernels", action="store_true")
    parser.add_argument(
        "--min-antigen-length",
        type=int,
        default=17,
        help=(
            "Discard complexes whose antigen sequence is shorter than this length. "
            "Set to 0 to keep every Protein_SAbDab entry."
        ),
    )
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache = args.embedding_cache.expanduser()
    embedding_cache.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.accelerator == "gpu":
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU execution.")

    LOGGER.info("Loading Protein_SAbDab dataset")
    dataset_dir = args.dataset_dir.expanduser()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    examples_by_split = load_dataset(
        dataset_dir,
        min_antigen_length=args.min_antigen_length,
    )
    all_examples = [example for split in examples_by_split.values() for example in split]

    LOGGER.info("Preparing ESM2 embeddings for %d unique sequences", len(all_examples))
    embedder = ESMEmbedder(args.esm_model, embedding_cache, device)
    embeddings = prepare_embeddings(embedder, all_examples)

    LOGGER.info("Running Boltz2 affinity inference")
    args.cache_dir = args.cache_dir.expanduser()
    predictor = BoltzAffinityPredictor(
        cache_dir=args.cache_dir,
        work_dir=args.output_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        override=args.boltz_override,
        sampling_steps=args.boltz_sampling_steps,
        diffusion_samples=args.boltz_diffusion_samples,
        sampling_steps_affinity=args.boltz_sampling_steps_affinity,
        diffusion_samples_affinity=args.boltz_diffusion_samples_affinity,
        max_parallel_samples=args.boltz_max_parallel,
        checkpoint=args.boltz_checkpoint,
        affinity_checkpoint=args.boltz_affinity_checkpoint,
        no_kernels=args.boltz_no_kernels,
    )
    predictor.run(all_examples)
    boltz_outputs = predictor.collect_features()

    LOGGER.info("Building feature matrices")
    matrices = build_feature_matrices(examples_by_split, embeddings, boltz_outputs)
    train_features, train_targets, train_ids = matrices.get("train", (np.zeros((0, 0)), np.zeros(0), []))
    val_features, val_targets, val_ids = matrices.get("valid", (np.zeros((0, 0)), np.zeros(0), []))
    test_features, test_targets, test_ids = matrices.get("test", (np.zeros((0, 0)), np.zeros(0), []))

    if train_features.size == 0 or val_features.size == 0:
        msg = "Training and validation splits must be non-empty."
        raise RuntimeError(msg)

    normalized_matrices, mean, std = normalize_features(train_features, matrices)
    norm_train_features, norm_train_targets, train_ids = normalized_matrices["train"]
    norm_val_features, norm_val_targets, val_ids = normalized_matrices.get("valid", (np.zeros((0, 0)), np.zeros(0), []))
    norm_test_features, norm_test_targets, test_ids = normalized_matrices.get("test", (np.zeros((0, 0)), np.zeros(0), []))

    train_dataset = FeatureDataset(norm_train_features, norm_train_targets)
    val_dataset = FeatureDataset(norm_val_features, norm_val_targets)
    model = HybridAffinityRegressor(
        input_dim=norm_train_features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    LOGGER.info("Training hybrid affinity regressor")
    model = train_model(
        model=model,
        train_data=train_dataset,
        val_data=val_dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    LOGGER.info("Evaluating on train/validation/test splits")
    train_predictions = predict_dataset(model, train_dataset, device)
    val_predictions = predict_dataset(model, val_dataset, device)
    test_dataset = FeatureDataset(norm_test_features, norm_test_targets)
    test_predictions = predict_dataset(model, test_dataset, device)

    metrics = {
        "train": compute_metrics(norm_train_targets, train_predictions),
        "valid": compute_metrics(norm_val_targets, val_predictions),
        "test": compute_metrics(norm_test_targets, test_predictions),
    }
    with (args.output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    export_predictions(args.output_dir, "train", train_ids, norm_train_targets, train_predictions)
    export_predictions(args.output_dir, "valid", val_ids, norm_val_targets, val_predictions)
    if test_ids:
        export_predictions(args.output_dir, "test", test_ids, norm_test_targets, test_predictions)

    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "config": vars(args),
    }, args.output_dir / "hybrid_model.pt")

    LOGGER.info("Hybrid affinity training complete. Metrics saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
