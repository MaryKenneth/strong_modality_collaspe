# MultiHuSE: Code and Data

Code and pre-extracted feature embeddings for the MultiHuSE experiments in an ICLR 2027 submission on strong-modality collapse in multimodal fusion. This repository is anonymized for double-blind review. Code for the remaining three datasets used in the paper (MUStARD, UR-FUNNY, Food-101) will be added to this repository before the camera-ready deadline.

## Contents

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

## Dataset

The splits in `splits_strict_v2/` are 5-fold cross-validation folds grouped by unique text ID, so no text ID appears in both the train and test portion of the same fold. This is intentional: it avoids the leakage risk present in some other splits of this dataset that are not grouped this way.

Feature embeddings are pre-extracted (frozen) from the following encoders:
- Text: BERT, E5
- Audio: Dasheng
- Video: PE-Core, VideoMAE

## Requirements

```
python >= 3.9
torch
pandas
numpy
scikit-learn
```

A CUDA-capable GPU is recommended but not required; all scripts fall back to CPU automatically if CUDA is unavailable.

## Usage

Each script is self-contained and run directly from within the `multihuse/` directory, so that the relative paths to `multihuse_dataset/` resolve correctly:

```bash
cd multihuse
python baseline_text_only.py
python fusion_iaf_makd.py
# etc.
```

There are no command-line arguments; encoder combinations, fold selection, and hyperparameters are set as constants near the top of each script and can be edited there. All scripts use a fixed random seed (999) for reproducibility.

Predictions and results are written to a `predictions_output/` directory created automatically alongside the scripts on first run.

## Citation

Citation details will be added after the review period.

## License

MIT License. See `LICENSE`.
