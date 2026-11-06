# USPTO reaction scorers

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

Training and inference code for the USPTO-Full experiments accompanying
*Trustworthy Retrosynthesis: Eliminating Hallucinations with a Diverse Ensemble of
Reaction Scorers*.

The release includes Reaction Prior (RP), Reaction Graph Plausibility (RGP), and
Reference Reaction Scorer (RRS), with model weights, training splits, and
configurations.

The [reaction benchmark](benchmark/README.md) contains 500 human-annotated
reactions with SMILES, plausibility categories, and issue categories. The
Syntheseus integration runs multi-step retrosynthesis with optional scorer
filters and the public pretrained RootAligned generator. **Only RootAligned is
officially supported.** Current search commands do not support Graph2Edits.

Use Python 3.10 or 3.11 and run commands from the repository root.

## Generator and threshold calibration

RP uses the original released implementation without changes to its scoring
formula, normalization, or weights.

Only Public RootAligned is officially supported. The selected thresholds in
`configs/decision_coverage90_rootaligned.json` preserved complete routes for
9 of 10 calibration targets using a historical eMolecules inventory from
2023-09-27. That commercial catalog is not distributed with this repository.
This calibration result is not a guarantee for new targets or inventories.

Tune thresholds for your targets, inventory, and search settings, following the
paper's constraint of preserving routes for at least 90% of calibration targets.
Validate coverage and chemical correctness independently. Other generators can
require different normalization, threshold tuning, or scorer retraining; their
outputs must not be assumed compatible with the existing scorers and cutoffs.

## Set up the models

Download the assets archive `uspto-reaction-scorers-assets.tar.gz` (483 MiB)
from [Google Drive](https://drive.google.com/file/d/1mpUbN1labZt34r8Tt7C7nNjlKVYFVCLk/view?usp=sharing).
From a terminal, `gdown` handles Google Drive's large-file confirmation page:

```bash
python -m pip install gdown
gdown 1mpUbN1labZt34r8Tt7C7nNjlKVYFVCLk -O uspto-reaction-scorers-assets.tar.gz
```

Verify the checksum, then extract the archive into the repository root:

```bash
shasum -a 256 uspto-reaction-scorers-assets.tar.gz
# 3acae81fabed9262d5f6dedc5da37fc583b80297bb0de030a44c444d9a76ca7b
tar -xzf uspto-reaction-scorers-assets.tar.gz -C .
```

The archive contains model weights, the reference index, training data, splits,
and configurations.

For GPU inference on Linux with CUDA 11.8, install the matching Torch build:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
```

Install the model dependencies and verify the weights:

```bash
python -m pip install -e '.[models]'
python tools/check_models.py --models rp rgp
```

## Score and evaluate reactions

Input CSVs need `substrates` and `product` columns containing atom-mapped SMILES.
For the paper's binary protocol, Safe bet is positive, Rather not and Nonsense
are negative, and Worthwhile is excluded. Prepare the benchmark labels, then
score and evaluate the reactions:

```bash
python - <<'PYTHON'
from pathlib import Path
import pandas as pd
f = pd.read_csv("benchmark/reactions.csv")
f = f.loc[f["category"].isin(["Safe bet", "Rather not", "Nonsense"])].copy()
f["class"] = (f["category"] == "Safe bet").astype(int)
f["robustness_test_name"] = f["issue"]
Path("runs").mkdir(exist_ok=True)
f.to_csv("runs/benchmark.csv", index=False)
PYTHON

python -m reaction_scorers.score --model rp --input runs/benchmark.csv --output runs/rp_scores.csv --device cpu
python -m reaction_scorers.score --model rgp --input runs/benchmark.csv --output runs/rgp_scores.csv --device cpu
python -m reaction_scorers.evaluate --results runs/rp_scores.csv
python -m reaction_scorers.evaluate --results runs/rgp_scores.csv
```

Scoring adds a `score` column. Evaluation reports ROC-AUC and average precision
(PR-AUC), excluding failed predictions. Use `--device cuda:0` for GPU inference.

### Reference scoring

```bash
python -m pip install -e '.[reference]'
python -m reaction_scorers.reference_sqlite --index models/rrs/reference.sqlite --input runs/benchmark.csv --output runs/rrs_scores.csv
python -m reaction_scorers.evaluate --results runs/rrs_scores.csv
```

To rebuild the reference index from the training corpus:

```bash
python -m tools.build_reference_index --data training/rp/data.zip --output rebuilt-reference.sqlite --workers 4
```

## Train RP or RGP

After extracting the assets, restore the training splits to CSV files:

```bash
python -m pip install -e '.[train]'
python tools/restore_training.py --data training/rp/data.zip --splits training/rp/splits.zip --output restored/rp
python tools/restore_training.py --data training/rgp/data.zip --splits training/rgp/splits.zip --output restored/rgp
```

Each restore command creates a new directory containing `train.csv`, `val.csv`,
and `test.csv`. Train using:

```bash
python -m reaction_scorers.train --model rp --train-csv restored/rp/train.csv --val-csv restored/rp/val.csv --output runs/rp
python -m reaction_scorers.train --model rgp --train-csv restored/rgp/train.csv --val-csv restored/rgp/val.csv --output runs/rgp
```

Training settings are recorded in `configs/{rp,rgp}_training.json`. Model
architectures and RGP feature dictionaries are in `models/`. Dataset and split
details are in `provenance/training_artifacts.json`.

## Check the implementation

```bash
python -m pip install -e '.[models,reference,train,retro,test]'
python -m pytest tests -q
```

[Inference validation results](validation/release_runtime.json) cover all 500
benchmark reactions.

## Retrosynthesis with Syntheseus

Syntheseus Retro* uses Public RootAligned, the only supported generator.
Install its dependencies and download the public USPTO-50K checkpoint:

```bash
python tools/setup_retro.py --download-generator
```

Download the [PaRoutes v2 n1 stock](https://zenodo.org/records/7341155), then
search for the first target with default RP/RGP/RRS filtering or without scorers:

```bash
mkdir -p cache
curl -fL https://zenodo.org/records/7341155/files/stock_n1.txt -o cache/stock_n1.txt
python -m reaction_scorers.retro --target-index 0 --inventory cache/stock_n1.txt --output runs/target-on.json
python -m reaction_scorers.retro --target-index 0 --inventory cache/stock_n1.txt --scorers off --output runs/target-off.json
```

Defaults: one augmentation, 10 predictions per expansion, 500 generator calls,
and 600 seconds, stopping at the first solution. JSON output includes routes,
settings, and scorer decisions. Filtering requires RP > `1.9536646643727844e-12`,
RGP > `0.12769793719053266`, and at least one reference match, as configured in
[configs/decision_coverage90_rootaligned.json](configs/decision_coverage90_rootaligned.json).

Tune thresholds for your targets, stock, and search settings using
`--decision path/to/tuned.json`. The defaults preserved routes for 9/10 targets
with a historical eMolecules inventory; recalibrate for PaRoutes or other settings.
Following the paper, preserve routes for at least 90% of calibration targets and
validate on independent targets. Other generators require compatibility checks
and threshold tuning, and may require scorer retraining.

See `python -m reaction_scorers.retro --help` for all options.

## Code guide

Paths below are relative to `reaction_scorers/`.

| Files | Purpose |
|---|---|
| `rp.py`, `transformer.py`, `tokenizer.py` | RP scoring, network, and tokenization |
| `graph/` | RGP graph representation and network |
| `score.py`, `loading.py`, `evaluate.py` | Model loading, CSV scoring, and metrics |
| `reference_keys.py`, `reference_sqlite.py` | Reference grouping and index lookup |
| `chemistry/` | Reaction processing, centers, and templates |
| `train.py`, `rp_dataset.py`, `training_data.py` | Training and data preparation |
| `retro.py`, `decision.py` | Public Syntheseus integration, filtering, and threshold calibration |
