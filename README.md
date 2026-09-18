# Strong-Modality Collapse: Code and Data

Code and pre-extracted feature embeddings for the experiments in an ICLR 2027 submission on strong-modality collapse in multimodal fusion. This repository is anonymized for double-blind review.

Four datasets are used in the paper: **MultiHuSE**, **MUStARD**, **UR-FUNNY**, **Food-101**. Each has its own top-level folder. Status:

| Dataset | Status |
|---|---|
| MUStARD | Added |
| MultiHuSE | Added |
| UR-FUNNY | Not yet added |
| Food-101 | Not yet added |

## Repository layout

```
.
├── mustard/
├── multihuse/
├── ur-funny/        (to be added)
└── food101/          (to be added)
```

Each dataset folder is self-contained: its own data subfolder, its own scripts, and its own `predictions_output/` created on first run. There is no shared code between folders, so each one can be read and run independently.

---

## mustard/

```
mustard/
├── mustard_dataset/
│   ├── mustard_dataset_master/
│   │   └── data/
│   │       ├── sarcasm_data.json         # Utterance identifiers and sarcasm labels
│   │       └── split_indices.p           # 5-fold cross-validation split indices
│   └── mustard_features_pkl_FIXED/
│       ├── context/                       # Pre-extracted context features (.pkl, per identifier)
│       │   ├── sarcasm/
│       │   └── not_sarcasm/
│       └── utterances/                    # Pre-extracted punchline/utterance features (.pkl, per identifier)
│           ├── sarcasm/
│           └── not_sarcasm/
├── extract_features_mustard_FIXED.py               # Feature extraction pipeline
├── baseline_text_only_mustard.py                   # Unimodal text baseline
├── baseline_audio_only_mustard.py                  # Unimodal audio baseline
├── baseline_video_only_mustard.py                  # Unimodal video baseline
├── baseline_early_fusion_mustard.py                # Early fusion baseline
├── baseline_late_fusion_mustard.py                 # Late fusion baseline
├── fusion_symmetric_mustard_baseline.py             # Symmetric fusion baseline (no distillation)
├── fusion_symmetric_mustard_distilled_4configs.py   # Symmetric fusion + MAKD, 4 encoder configurations
├── fusion_inverted_baseline_mustard_new.py          # Inverted Asymmetric Fusion (IAF), no distillation
├── fusion_inverted_cascade_mustard_new_4configs.py  # IAF + MAKD, 4 encoder configurations
├── dual_distillation_mustard.py                    # Dual-teacher MAKD (for MUStARD's co-dominant audio/video setting)
├── test_pathways_mustard_v2.py                     # Pathway isolation analysis across all 5 folds
├── causal_decomposition_mustard.py                 # 2x2 causal decomposition: cross-modal attention (on/off) x classifier state (frozen/trainable)
└── cka_analysis_mustard.py                          # Linear CKA representational similarity analysis
```

**Dataset.** 690 utterances, 5-fold cross-validation, split indices in `split_indices.p`. Labels and identifiers are in `sarcasm_data.json`; pre-extracted features are stored per identifier as individual `.pkl` files under `context/` (surrounding dialogue) and `utterances/` (the punchline), each split into `sarcasm/` and `not_sarcasm/`.

**Usage.** Run directly from within `mustard/`, so relative paths to `mustard_dataset/` resolve correctly:

```bash
cd mustard
python baseline_text_only_mustard.py
python fusion_inverted_cascade_mustard_new_4configs.py
# etc.
```

---

## multihuse/

```
multihuse/
├── multihuse_dataset/
│   ├── bert_text_embeddings.csv          # BERT text embeddings
│   ├── e5_text_embeddings.csv            # E5 text embeddings
│   ├── dasheng_audio_embeddings.csv      # Dasheng audio embeddings
│   ├── pe_core_visual_embeddings.csv     # PE-Core visual embeddings
│   ├── videomae_visual_embeddings.csv    # VideoMAE visual embeddings
│   └── splits_strict_v2/                 # 5-fold cross-validation splits, grouped by unique text ID
│       ├── fold_1_train.csv / fold_1_test.csv
│       ├── fold_2_train.csv / fold_2_test.csv
│       ├── fold_3_train.csv / fold_3_test.csv
│       ├── fold_4_train.csv / fold_4_test.csv
│       └── fold_5_train.csv / fold_5_test.csv
├── baseline_text_only.py                 # Unimodal text baseline
├── baseline_audio_only.py                # Unimodal audio baseline
├── baseline_video_only.py                # Unimodal video baseline
├── baseline_early_fusion.py              # Early fusion baseline
├── baseline_late_fusions.py              # Late fusion baseline
├── fusion_symmetric_baseline_models.py   # Symmetric fusion baseline (no distillation)
├── fusion_symmetric_makd.py              # Symmetric fusion + Modality-Aware Knowledge Distillation (MAKD)
├── fusion_iaf_baseline_models.py         # Inverted Asymmetric Fusion (IAF), no distillation
├── fusion_iaf_makd.py                    # IAF + MAKD
├── makd_multihuse.py                     # MAKD distillation routine
├── pathways_isolation_analysis.py        # Pathway isolation analysis (dominant-modality pathway accuracy)
├── causal_decomposition_multihuse.py     # 2x2 causal decomposition: cross-modal attention (on/off) x classifier state (frozen/trainable)
└── cka_analysis_multihuse.py             # Linear CKA representational similarity analysis
```

**Dataset.** Splits in `splits_strict_v2/` are 5-fold cross-validation folds grouped by unique text ID, so no text ID appears in both the train and test portion of the same fold. Feature embeddings are pre-extracted (frozen) from: Text (BERT, E5), Audio (Dasheng), Video (PE-Core, VideoMAE).

**Usage.** Run directly from within `multihuse/`, so relative paths to `multihuse_dataset/` resolve correctly:

```bash
cd multihuse
python baseline_text_only.py
python fusion_iaf_makd.py
# etc.
```

---

## ur-funny/ *(not yet added)*



## food101/ *(not yet added)*


---

## Requirements

```
python >= 3.9
torch
pandas
numpy
scikit-learn
```

A CUDA-capable GPU is recommended but not required; all scripts fall back to CPU automatically if CUDA is unavailable. No script takes command-line arguments; encoder combinations, fold selection, and hyperparameters are set as constants near the top of each script and can be edited there. Predictions and results are written to a `predictions_output/` directory created automatically alongside the scripts on first run.

## Citation

Citation details will be added after the review period.

## License

MIT License. See `LICENSE`.
