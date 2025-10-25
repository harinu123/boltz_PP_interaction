"""Evaluation helper for the antibody–antigen hackathon challenge."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from hackathon_api import Datapoint
except ModuleNotFoundError:  # pragma: no cover - executed when run as module
    from .hackathon_api import Datapoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel CAPRI-Q evaluation runner (Python port)",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=str(Path.cwd() / "inputs"),
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--result-folder",
        type=str,
        default=str(Path.cwd() / "outputs"),
        help="Directory to store result files",
    )
    parser.add_argument(
        "--submission-folder",
        type=str,
        default=str(Path.cwd() / "predictions"),
        help="Directory containing prediction files",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=50,
        help="Number of parallel jobs to run",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=5,
        help="Number of samples to evaluate per structure",
    )
    return parser.parse_args()


def _run_evaluation(
    gt_dir: Path,
    gt_structures: dict[str, Any],
    structure_name: str,
    index: int,
    args: argparse.Namespace,
) -> Optional[pd.DataFrame]:
    output_subdir = Path(args.result_folder) / f"{structure_name}_{index}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    prediction_file = Path(args.submission_folder) / structure_name / f"model_{index}.pdb"
    if not prediction_file.exists():
        print(f"No prediction file {prediction_file} found. Skipping.")
        return None

    capriq_cmd = [
        "/capri-q/bin/capriq",
        "-a",
        "--dontwrite",
        "-t",
        f"/app/ground_truth/{gt_structures['structure_complex']}",
        "-u",
        f"/app/ground_truth/{gt_structures['structure_ab']}",
        "-u",
        f"/app/ground_truth/{gt_structures['structure_ligand']}",
        "-z",
        "/app/outputs/",
        "-p",
        "65",
        "-o",
        f"/app/outputs/{structure_name}_{index}_results.txt",
        "-l",
        f"/app/outputs/{structure_name}_{index}_errors.txt",
        "/app/predictions/prediction.pdb",
        "&&",
        "chown",
        "-R",
        f"{os.getuid()}:{os.getgid()}",
        "/app/outputs",
    ]

    docker_cmd = [
        "docker",
        "run",
        "--group-add",
        str(os.getgid()),
        "--rm",
        "--network",
        "none",
        "-v",
        f"{gt_dir.absolute()}:/app/ground_truth/",
        "-v",
        f"{output_subdir.absolute()}:/app/outputs",
        "-v",
        f"{prediction_file.absolute()}:/app/predictions/prediction.pdb",
        "gitlab-registry.in2p3.fr/cmsb-public/capri-q",
        "/bin/bash",
        "-c",
        " ".join(capriq_cmd),
    ]

    print(
        f"Evaluating {structure_name} model {index}... Prediction file: {prediction_file}",
        flush=True,
    )

    try:
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Docker run failed for {structure_name} model {index}. Error: {exc}", file=sys.stderr)
        return pd.DataFrame(
            {
                "structure_name": [structure_name],
                "structure_index": [index],
                "nclash": [None],
                "clash_fraction": [None],
                "classification": ["error"],
                "error": [str(exc)],
            }
        )

    result_file = output_subdir / f"{structure_name}_{index}_results.txt"
    if not result_file.exists():
        print(f"No result file {result_file} found. Skipping.")
        return pd.DataFrame(
            {
                "structure_name": [structure_name],
                "structure_index": [index],
                "nclash": [None],
                "clash_fraction": [None],
                "classification": ["error"],
                "error": ["Result file not found"],
            }
        )

    df = pd.read_csv(result_file, sep=r"\s+")
    df["clash_fraction"] = df["model"].str.replace("/", "").astype(float) / df["nclash"]
    df["nclash"] = df["model"].str.replace("/", "").astype(int)
    df.drop(columns=["model"], inplace=True)
    df["structure_name"] = structure_name
    df["structure_index"] = index
    return df


def _load_dataset(path: Path) -> list[Datapoint]:
    with path.open() as handle:
        return [Datapoint.from_json(line) for line in handle if line.strip()]


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset_file)
    dataset = _load_dataset(dataset_path)
    ground_truth_dir = dataset_path.parent / "ground_truth"

    result_frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=args.njobs) as executor:
        futures = []
        for datapoint in dataset:
            structure_name = datapoint.datapoint_id
            gt_structures = datapoint.ground_truth or {}
            for idx in range(args.nsamples):
                futures.append(
                    executor.submit(
                        _run_evaluation,
                        ground_truth_dir,
                        gt_structures,
                        structure_name,
                        idx,
                        args,
                    )
                )

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                result_frames.append(result)

    if not result_frames:
        print("No evaluation results generated.")
        return

    combined = pd.concat(result_frames, ignore_index=True)
    result_dir = Path(args.result_folder)
    result_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(result_dir / "combined_results.csv", index=False)

    success_total = 0
    good_classes = {"high", "medium", "acceptable"}
    bad_classes = {"incorrect", "error"}
    for label in [*good_classes, *bad_classes]:
        count = len(
            combined[
                (combined["structure_index"] == 0)
                & (combined["classification"].str.contains(label))
            ]
        )
        if label in good_classes:
            success_total += count
        print(f"Number of {label} classifications in top 1: {count}")

    print(f"Number of successful top 1 predictions: {success_total} out of {len(dataset)}")
    print("All evaluations completed.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

