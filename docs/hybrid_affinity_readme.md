# Hybrid Boltz2 + ESM2 Antibody–Antigen Affinity Pipeline

This walkthrough demonstrates how to reproduce the hybrid affinity fine-tuning workflow implemented in [`scripts/train_hybrid_affinity.py`](../scripts/train_hybrid_affinity.py). The pipeline combines frozen ESM-2 language embeddings with Boltz2 structure-driven affinity estimates to train a lightweight regression head on the Protein_SAbDab benchmark.

## 1. Environment Setup

1. Create and activate a fresh Python 3.10+ environment (conda, venv, or virtualenv).
2. Install the project in editable mode with the optional CUDA extras:
   ```bash
   pip install -e .[cuda]
   ```
   If you are running on CPU-only hardware, drop the `[cuda]` extra, noting that Boltz inference will run significantly slower.
3. Install the training-specific dependency for the language model encoder:
   ```bash
   pip install transformers pytorch-lightning==2.2.5
   ```
   The `transformers` package provides the ESM2 encoder. PyTorch Lightning is pinned to the version the script was validated against.

## 2. Download Model Weights

Boltz inference downloads weights on first use. Make sure the machine has internet access the first time you invoke the script. If you prefer to supply local checkpoints, pass `--boltz-checkpoint` and `--boltz-affinity-checkpoint` arguments that point to the corresponding `.ckpt` files.

For the ESM2 encoder (`facebook/esm2_t33_650M_UR50D` by default) you may need to authenticate with the Hugging Face Hub:

```bash
huggingface-cli login
```

Once downloaded, both the Boltz and ESM weights are cached under `~/.cache` and reused automatically.

The first invocation will also download the Protein_SAbDab CSV from the Harvard Dataverse into `--dataset-dir` (default `./data`).

## 3. Prepare Optional GPU Resources

By default the script requests GPU acceleration (`--accelerator gpu --devices 1`). Ensure CUDA is available (`nvidia-smi`) before launching. To run on CPU instead, pass `--accelerator cpu`.

## 4. Run the Training Script

Execute the end-to-end pipeline from the project root:

```bash
python scripts/train_hybrid_affinity.py \
  --output-dir ./hybrid_affinity_output \
  --embedding-cache ./embedding_cache \
  --esm-model facebook/esm2_t33_650M_UR50D \
  --epochs 75 \
  --batch-size 32
```

Key behaviors:
- The script downloads the Protein_SAbDab CSV directly from the Harvard Dataverse (ID 4167357), caches it under `--dataset-dir`, and logs the train/validation/test sizes.
- It keeps only complexes whose antigen sequence contains at least 17 amino acids (roughly the antibody–antigen subset used in TDC). Override this with `--min-antigen-length 0` to retain all 493 entries.
- It caches mean-pooled ESM embeddings for every antibody heavy/light chain and antigen sequence (`--embedding-cache`).
- It generates temporary YAML inputs for Boltz, runs affinity inference, and collects binder features (Boltz caches live under `--cache-dir`, defaulting to `~/.boltz`).
- It assembles hybrid feature matrices that concatenate Boltz outputs and language embeddings, normalizes them using the training split, and trains a two-layer MLP regressor (`--hidden-dim`, `--dropout`, `--learning-rate`, `--weight-decay`, `--epochs`).

You can further customize the Boltz sampler by adjusting `--boltz-sampling-steps`, `--boltz-diffusion-samples`, `--boltz-max-parallel`, and related flags. Run `python scripts/train_hybrid_affinity.py --help` to inspect the full set of options.

## 5. Inspect Outputs

All artifacts land in `--output-dir` (default `./hybrid_affinity_output`):

- `evaluation_metrics.json`: Train/validation/test MSE, MAE, RMSE, and Pearson correlation coefficients.
- `predictions_train.csv`, `predictions_valid.csv`, `predictions_test.csv`: Per-sample predictions with ground-truth log affinity.
- `hybrid_model.pt`: Serialized PyTorch state dict along with the feature normalization statistics and the CLI arguments used for the run.
- `boltz_inputs/` and `boltz_results_*/`: Intermediate files generated for Boltz inference. You can delete them after training to reclaim disk space.

## 6. Reusing the Trained Head

To deploy the trained regressor on new antibody–antigen complexes:

1. Generate ESM embeddings and Boltz affinity features for your new examples using the helper utilities in the script (`prepare_embeddings`, `BoltzAffinityPredictor.collect_features`).
2. Load `hybrid_model.pt`, apply the stored mean/std normalization to your feature vectors, and feed them through the saved network.

Refer to the inline documentation in [`scripts/train_hybrid_affinity.py`](../scripts/train_hybrid_affinity.py) for additional implementation details.
