"""Evaluation helper for the protein–ligand hackathon challenge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from hackathon_api import Datapoint
except ModuleNotFoundError:  # pragma: no cover - executed when run as module
    from .hackathon_api import Datapoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate docking submissions")
    parser.add_argument(
        "--dataset-file",
        type=str,
        required=True,
        help="Path to JSONL dataset description",
    )
    parser.add_argument(
        "--submission-folder",
        type=str,
        required=True,
        help="Directory containing submitted structures",
    )
    parser.add_argument(
        "--result-folder",
        type=str,
        required=True,
        help="Output directory for evaluation results",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset_file)
    submission_dir = Path(args.submission_folder)
    result_dir = Path(args.result_folder)
    result_dir.mkdir(parents=True, exist_ok=True)

    datapoints = [
        Datapoint.from_json(line)
        for line in dataset_path.read_text().splitlines()
        if line.strip()
    ]

    rows = []
    for datapoint in datapoints:
        folder = submission_dir / datapoint.datapoint_id
        if not folder.exists():
            rows.append(
                {
                    "datapoint_id": datapoint.datapoint_id,
                    "status": "missing_submission",
                    "details": json.dumps({}),
                }
            )
            continue

        pdbs = sorted(folder.glob("model_*.pdb"))
        rows.append(
            {
                "datapoint_id": datapoint.datapoint_id,
                "status": "found",
                "details": json.dumps({"num_models": len(pdbs)}),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(result_dir / "evaluation_summary.csv", index=False)
    print(f"Wrote evaluation summary to {result_dir / 'evaluation_summary.csv'}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

