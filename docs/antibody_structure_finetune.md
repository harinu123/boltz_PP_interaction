# Antibody–Antigen Structure Fine-Tuning

This guide outlines how to continue training the Boltz structure model on
antibody–antigen complexes while conditioning on ESM2 language-model
probabilities. The pipeline reuses the core training scripts but enables a
hybrid feature path where every residue receives an ESM-backed amino-acid
profile that is blended with the traditional MSA statistics.

## 1. Prepare the SAbDab structures

1. Create a workspace directory where all processed assets will live:
   ```bash
   mkdir -p ~/sabdab_finetune/{raw,targets,msas}
   ```

2. Download the Protein_SAbDab metadata used throughout the project:
   ```bash
   python - <<'PY'
   import pathlib
   import requests

   dest = pathlib.Path('~/sabdab_finetune/raw/protein_sabdab.csv').expanduser()
   dest.parent.mkdir(parents=True, exist_ok=True)

   url = 'https://dataverse.harvard.edu/api/access/datafile/4167357'
   with requests.get(url, timeout=60, stream=True) as resp:
       resp.raise_for_status()
       dest.write_bytes(resp.content)
   print(f"Saved Protein_SAbDab metadata to {dest}")
   PY
   ```

3. Fetch the mmCIF files for each complex listed in the CSV. The snippet below
   iterates over the metadata and downloads the corresponding entries directly
   from the RCSB archive:
   ```bash
   python - <<'PY'
   import pathlib
   import pandas as pd
   import requests

   base = pathlib.Path('~/sabdab_finetune/raw').expanduser()
   mmcif_dir = base / 'mmcif'
   mmcif_dir.mkdir(parents=True, exist_ok=True)

   df = pd.read_csv(base / 'protein_sabdab.csv')
   for pdb_id in df['ID1'].unique():
       pdb_id = str(pdb_id).lower()
       path = mmcif_dir / f"{pdb_id}.cif"
       if path.exists():
           continue
       url = f"https://files.rcsb.org/download/{pdb_id}.cif"
       resp = requests.get(url, timeout=60)
       resp.raise_for_status()
       path.write_bytes(resp.content)
       print(f"Downloaded {pdb_id}")
   PY
   ```

4. Process the mmCIF structures into Boltz training records. The standard
   preprocessing utilities that ship with Boltz can be repurposed for the
   antibody subset. If you have a precomputed clustering manifest you can pass
   it via `--clusters`, otherwise omit the flag and every chain will fall back
   to the default cluster id of `-1`:
   ```bash
   export BOLTZ_CACHE=~/boltz_cache
   python scripts/process/rcsb.py \
       --datadir ~/sabdab_finetune/raw/mmcif \
       --outdir  ~/sabdab_finetune/targets \
       --cache-dir "$BOLTZ_CACHE" \
       --max_file_size 2000000
   ```
   If a Redis instance is unavailable the script now falls back to the
   specified cache directory, pulling CCD components on demand and storing
   them under `molecules/`. This step writes `structures/*.npz` and
   `records/*.json` inside the `targets` directory, together with a
   consolidated `manifest.json`.

5. Generate (or reuse) single-sequence MSAs. When no evolutionary depth is
   available, Boltz will fall back to dummy alignments. You can therefore place
   empty placeholders for each chain ID:
   ```bash
   python - <<'PY'
   import json, numpy as np, pathlib
   from boltz.data.types import MSA, MSADeletion, MSAResidue, MSASequence

   msa_dir = pathlib.Path('~/sabdab_finetune/msas').expanduser()
   msa_dir.mkdir(parents=True, exist_ok=True)

   manifest_path = pathlib.Path('~/sabdab_finetune/targets/manifest.json').expanduser()
   manifest = json.load(manifest_path.open())
   seen = set()
   for record in manifest:
       for chain in record['chains']:
           msa_id = chain['msa_id']
           if not msa_id or msa_id in seen:
               continue
           seen.add(msa_id)
           msa = MSA(
               residues=np.zeros((0,), dtype=MSAResidue),
               deletions=np.zeros((0,), dtype=MSADeletion),
               sequences=np.array([(0, -1, 0, 0, 0, 0)], dtype=MSASequence),
           )
           np.savez_compressed(msa_dir / f"{msa_id}.npz", **msa.__dict__)
   PY
   ```

## 2. Launch fine-tuning with ESM guidance

1. Choose a Boltz checkpoint (e.g., the public `boltz2_conf.ckpt`) and point the
   new configuration to your processed directories. The default config expects
   the directories to be provided via environment variables, so you can export
   them without touching the YAML:
   ```bash
   export SABDAB_PROCESSED_TARGETS=~/sabdab_finetune/targets
   export SABDAB_PROCESSED_MSAS=~/sabdab_finetune/msas
   export BOLTZ_OUTPUT=~/sabdab_finetune/output
   export BOLTZ_PRETRAINED=~/weights/boltz2_conf.ckpt
   ```
   Alternatively, you can replace the placeholders (`SABDAB_PROCESSED_TARGETS`,
   `SABDAB_PROCESSED_MSAS`, etc.) directly in
   `scripts/train/configs/antibody_structure.yaml` with absolute paths.

2. Run the training script with the antibody-specific configuration:
   ```bash
   BOLTZ_OUTPUT=~/sabdab_finetune/output \
   BOLTZ_PRETRAINED=~/weights/boltz2_conf.ckpt \
   python scripts/train/train.py scripts/train/configs/antibody_structure.yaml
   ```

   The datamodule automatically loads the specified ESM2 checkpoint, projects
   residue-wise probability profiles, and mixes them with the standard MSA
   statistics using the ratio defined in the config (`data.esm_profile.mix`).

## 3. Practical considerations

- **Caching:** ESM logits are cached under `data.esm_profile.cache_dir`. If you
  restart training, profiles are reused rather than recomputed.
- **Device selection:** Set `data.esm_profile.device` to `cpu`, `cuda`, or leave
  it as `auto` to pick CUDA only when it is available.
- **Mixing coefficient:** Lowering `data.esm_profile.mix` increases the influence
  of the language-model probabilities relative to the traditional MSA profile.
- **Validation crop:** The antibody config enables `crop_validation` so the
  validation loop mirrors the training crop strategy, which better reflects
  inference behaviour on complex interfaces.

With these steps you can continue training Boltz on SAbDab antibody–antigen
complexes while injecting language-model priors that emphasise paratope and
epitope preferences directly in the structural featurisation stage.
