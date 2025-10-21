"""SAbDab fine-tuned Boltz2 inference harness for hackathon datasets."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple

import yaml

try:
    from hackathon_api import Datapoint, Protein, SmallMolecule
except ModuleNotFoundError as exc:  # pragma: no cover - imported at runtime
    msg = (
        "Could not import hackathon_api. Install the hackathon template or add "
        "it to PYTHONPATH before running this script."
    )
    raise SystemExit(msg) from exc

###############################################################################
# configuration helpers
###############################################################################


@dataclass(frozen=True)
class PredictionConfig:
    """Container for reusable Boltz CLI arguments."""

    diffusion_samples: int = 5
    recycling_steps: int = 3
    sampling_steps: int = 80
    step_scale: float = 1.15
    max_parallel_samples: int = 1

    def to_args(self) -> list[str]:
        return [
            "--diffusion_samples",
            str(self.diffusion_samples),
            "--recycling_steps",
            str(self.recycling_steps),
            "--sampling_steps",
            str(self.sampling_steps),
            "--step_scale",
            f"{self.step_scale:.2f}",
            "--max_parallel_samples",
            str(self.max_parallel_samples),
        ]


DEFAULT_COMPLEX_CONFIG = PredictionConfig()
DEFAULT_PROTEIN_LIGAND_CONFIG = PredictionConfig(diffusion_samples=8, sampling_steps=100)

###############################################################################
# participant hooks
###############################################################################


def prepare_protein_complex(
    datapoint_id: str,
    proteins: Sequence[Protein],
    input_dict: dict,
    msa_dir: Optional[Path] = None,
) -> List[tuple[dict, List[str]]]:
    """Return configurations for antibody–antigen complex generation."""

    _ = (datapoint_id, proteins, msa_dir)
    args = DEFAULT_COMPLEX_CONFIG.to_args()
    return [(input_dict, args)]


def prepare_protein_ligand(
    datapoint_id: str,
    protein: Protein,
    ligands: Sequence[SmallMolecule],
    input_dict: dict,
    msa_dir: Optional[Path] = None,
) -> List[tuple[dict, List[str]]]:
    """Return configurations for protein–ligand complex generation."""

    _ = (datapoint_id, protein, ligands, msa_dir)
    args = DEFAULT_PROTEIN_LIGAND_CONFIG.to_args()
    return [(input_dict, args)]


def post_process_protein_complex(
    datapoint: Datapoint,
    input_dicts: Sequence[dict[str, Any]],
    cli_args_list: Sequence[Sequence[str]],
    prediction_dirs: Sequence[Path],
) -> List[Path]:
    """Return ranked PDB paths for antibody–antigen predictions."""

    _ = (input_dicts, cli_args_list)
    all_pdbs: list[Path] = []
    for prediction_dir in prediction_dirs:
        glob_pattern = f"{datapoint.datapoint_id}_config_*_model_*.pdb"
        all_pdbs.extend(sorted(prediction_dir.glob(glob_pattern)))
    return sorted(all_pdbs)


def post_process_protein_ligand(
    datapoint: Datapoint,
    input_dicts: Sequence[dict[str, Any]],
    cli_args_list: Sequence[Sequence[str]],
    prediction_dirs: Sequence[Path],
) -> List[Path]:
    """Return ranked PDB paths for protein–ligand predictions."""

    _ = (datapoint, input_dicts, cli_args_list)
    all_pdbs: list[Path] = []
    for prediction_dir in prediction_dirs:
        glob_pattern = f"{datapoint.datapoint_id}_config_*_model_*.pdb"
        all_pdbs.extend(sorted(prediction_dir.glob(glob_pattern)))
    return sorted(all_pdbs)


###############################################################################
# boilerplate from the hackathon template with Boltz2 fine-tune extensions
###############################################################################

DEFAULT_OUT_DIR = Path("predictions")
DEFAULT_SUBMISSION_DIR = Path("submission")
DEFAULT_INPUTS_DIR = Path("inputs")
DEFAULT_CHECKPOINT_DIRS: tuple[Path, ...] = (
    Path.home() / "sabdab_finetune" / "output" / "checkpoints",
    Path.home() / "sabdab_finetune" / "output",
    Path.home() / "weights",
)


ap = argparse.ArgumentParser(
    description=(
        "Prediction harness that mirrors the Boltz hackathon template but "
        "defaults to the antibody–antigen fine-tuned Boltz2 checkpoint."
    ),
    epilog=(
        "Examples:\n"
        "  Single datapoint: python hackathon/predict_hackathon.py --input-json"
        " examples/specs/example_protein_ligand.json --msa-dir ./msa\n"
        "  Multiple datapoints: python hackathon/predict_hackathon.py --input-jsonl"
        " hackathon_data/datasets/abag_public/abag_public.jsonl --msa-dir ./msa"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

input_group = ap.add_mutually_exclusive_group(required=True)
input_group.add_argument("--input-json", type=str, help="Path to JSON datapoint for a single datapoint")
input_group.add_argument("--input-jsonl", type=str, help="Path to JSONL file with multiple datapoint definitions")

ap.add_argument("--msa-dir", type=Path, help="Directory containing MSA files (for computing relative paths in YAML)")
ap.add_argument("--submission-dir", type=Path, required=False, default=DEFAULT_SUBMISSION_DIR, help="Directory to place final submissions")
ap.add_argument("--intermediate-dir", type=Path, required=False, default=Path("hackathon_intermediate"), help="Directory to place generated input YAML files and predictions")
ap.add_argument("--group-id", type=str, required=False, default=None, help="Group ID to set for submission directory (sets group rw access if specified)")
ap.add_argument("--result-folder", type=Path, required=False, default=None, help="Directory to save evaluation results. If set, will automatically run evaluation after predictions.")
ap.add_argument("--checkpoint", type=Path, required=False, default=None, help="Path to a Boltz model checkpoint (.ckpt) to use for prediction")
ap.add_argument("--override", action="store_true", default=False, help="Force Boltz to overwrite/redo existing predictions")
ap.add_argument("--cache", type=Path, default=None, help="Override BOLTZ_CACHE location")
ap.add_argument("--max-samples", type=int, default=None, help="Limit the number of datapoints processed from a JSONL file.")

args = ap.parse_args()


def _prefill_input_dict(
    datapoint_id: str,
    proteins: Iterable[Protein],
    ligands: Optional[Sequence[SmallMolecule]] = None,
    msa_dir: Optional[Path] = None,
) -> dict:
    seqs: list[dict[str, Any]] = []
    for protein in proteins:
        msa_relative_path: Optional[str]
        if msa_dir and protein.msa:
            msa_path = Path(protein.msa)
            msa_full_path = msa_path if msa_path.is_absolute() else msa_dir / msa_path
            try:
                msa_relative_path = os.path.relpath(msa_full_path, Path.cwd())
            except ValueError:
                msa_relative_path = str(msa_full_path)
        else:
            msa_relative_path = protein.msa

        seqs.append(
            {
                "protein": {
                    "id": protein.id,
                    "sequence": protein.sequence,
                    "msa": msa_relative_path,
                }
            }
        )

    if ligands:
        for ligand in ligands:
            seqs.append({"ligand": {"id": ligand.id, "smiles": ligand.smiles}})

    return {"version": 1, "sequences": seqs}


def _iter_datapoints(jsonl_path: Path) -> Iterator[Datapoint]:
    with jsonl_path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield Datapoint.from_json(stripped)


def _resolve_checkpoint(explicit: Optional[Path]) -> Optional[Path]:
    if explicit:
        expanded = explicit.expanduser()
        if not expanded.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {expanded}")
        return expanded

    env_candidates = [
        os.environ.get("BOLTZ_FINETUNED_CKPT"),
        os.environ.get("BOLTZ_CHECKPOINT"),
    ]
    for candidate in env_candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            print(f"Using checkpoint from environment: {path}")
            return path
        print(f"WARNING: checkpoint path in environment is missing: {path}")

    discovered: list[Path] = []
    for root in DEFAULT_CHECKPOINT_DIRS:
        root = root.expanduser()
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".ckpt":
            discovered.append(root)
        else:
            discovered.extend(sorted(root.rglob("*.ckpt")))

    if discovered:
        latest = max(discovered, key=lambda p: p.stat().st_mtime)
        print(f"Auto-discovered checkpoint: {latest}")
        return latest

    return None


def _ensure_cache_path(override: Optional[Path]) -> Path:
    if override is not None:
        cache = override.expanduser()
    else:
        cache_env = os.environ.get("BOLTZ_CACHE")
        cache = Path(cache_env).expanduser() if cache_env else Path.home() / "boltz_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _run_boltz_and_collect(datapoint: Datapoint, checkpoint: Optional[Path]) -> None:
    out_dir = args.intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subdir = args.submission_dir / datapoint.datapoint_id
    subdir.mkdir(parents=True, exist_ok=True)

    base_input_dict = _prefill_input_dict(datapoint.datapoint_id, datapoint.proteins, datapoint.ligands, args.msa_dir)

    if datapoint.task_type == "protein_complex":
        configs = prepare_protein_complex(datapoint.datapoint_id, datapoint.proteins, base_input_dict, args.msa_dir)
    elif datapoint.task_type == "protein_ligand":
        configs = prepare_protein_ligand(datapoint.datapoint_id, datapoint.proteins[0], datapoint.ligands, base_input_dict, args.msa_dir)
    else:  # pragma: no cover - defensive template parity
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    all_input_dicts: list[dict[str, Any]] = []
    all_cli_args: list[list[str]] = []
    all_pred_subfolders: list[Path] = []

    input_dir = args.intermediate_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    for config_idx, (input_dict, cli_args) in enumerate(configs):
        yaml_path = input_dir / f"{datapoint.datapoint_id}_config_{config_idx}.yaml"
        with yaml_path.open("w") as handle:
            yaml.safe_dump(input_dict, handle, sort_keys=False)

        cache = _ensure_cache_path(args.cache)
        cmd = [
            sys.executable,
            "-m",
            "boltz.main",
            "predict",
            str(yaml_path),
            "--model",
            "boltz2",
            "--devices",
            "1",
            "--out_dir",
            str(out_dir),
            "--cache",
            str(cache),
            "--no_kernels",
            "--output_format",
            "pdb",
        ]

        if checkpoint is not None:
            cmd.extend(["--checkpoint", str(checkpoint)])

        if args.override:
            cmd.append("--override")

        cmd.extend(cli_args)
        print(f"Running config {config_idx}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

        pred_subfolder = out_dir / f"boltz_results_{datapoint.datapoint_id}_config_{config_idx}" / "predictions" / f"{datapoint.datapoint_id}_config_{config_idx}"
        all_input_dicts.append(input_dict)
        all_cli_args.append(list(cli_args))
        all_pred_subfolders.append(pred_subfolder)

    if datapoint.task_type == "protein_complex":
        ranked_files = post_process_protein_complex(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    else:
        ranked_files = post_process_protein_ligand(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)

    if not ranked_files:
        raise FileNotFoundError(f"No model files found for {datapoint.datapoint_id}")

    for i, file_path in enumerate(ranked_files[:5]):
        suffix = file_path.suffix or ".pdb"
        target = subdir / f"model_{i}{suffix}"
        shutil.copy2(file_path, target)
        print(f"Saved: {target}")

    if args.group_id:
        try:
            subprocess.run(["chgrp", "-R", args.group_id, str(subdir)], check=True)
            subprocess.run(["chmod", "-R", "g+rw", str(subdir)], check=True)
        except Exception as exc:  # pragma: no cover - best effort permissions
            print(f"WARNING: Failed to adjust group permissions: {exc}")


def _load_datapoint(path: Path) -> Datapoint:
    return Datapoint.from_json(path.read_text())


def _run_evaluation(input_file: str, task_type: str, submission_dir: Path, result_folder: Path) -> None:
    script_dir = Path(__file__).parent
    if task_type == "protein_complex":
        eval_script = script_dir / "evaluate_abag.py"
        cmd = ["python", str(eval_script), "--dataset-file", input_file, "--submission-folder", str(submission_dir), "--result-folder", str(result_folder)]
    elif task_type == "protein_ligand":
        eval_script = script_dir / "evaluate_asos.py"
        cmd = ["python", str(eval_script), "--dataset-file", input_file, "--submission-folder", str(submission_dir), "--result-folder", str(result_folder)]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    print(f"\n{'=' * 80}\nRunning evaluation for {task_type}...\nCommand: {' '.join(cmd)}\n{'=' * 80}\n")
    subprocess.run(cmd, check=True)
    print(f"\nEvaluation complete. Results saved to {result_folder}")


def _process_jsonl(jsonl_path: Path, checkpoint: Optional[Path]) -> None:
    print(f"Processing JSONL file: {jsonl_path}")
    for idx, datapoint in enumerate(_iter_datapoints(jsonl_path), start=1):
        if args.max_samples is not None and idx > args.max_samples:
            print("Reached --max-samples limit; stopping early.")
            break
        print(f"\n--- Processing line {idx} ({datapoint.datapoint_id}) ---")
        _run_boltz_and_collect(datapoint, checkpoint)


def _process_json(json_path: Path, checkpoint: Optional[Path]) -> Datapoint:
    print(f"Processing JSON file: {json_path}")
    datapoint = _load_datapoint(json_path)
    _run_boltz_and_collect(datapoint, checkpoint)
    return datapoint


def main() -> None:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    if checkpoint is None:
        print("No fine-tuned checkpoint found; Boltz will fall back to cache defaults.")

    task_type: Optional[str] = None
    input_file: Optional[str] = None

    if args.input_json:
        input_path = Path(args.input_json)
        datapoint = _process_json(input_path, checkpoint)
        input_file = args.input_json
        task_type = datapoint.task_type
    else:
        input_path = Path(args.input_jsonl)
        _process_jsonl(input_path, checkpoint)
        input_file = args.input_jsonl
        try:
            first = next(_iter_datapoints(input_path))
            task_type = first.task_type
        except StopIteration:
            task_type = None

    if args.result_folder and task_type and input_file:
        try:
            _run_evaluation(input_file, task_type, args.submission_dir, args.result_folder)
        except Exception as exc:  # pragma: no cover - evaluation optional
            print(f"WARNING: Evaluation failed: {exc}")


if __name__ == "__main__":
    main()
