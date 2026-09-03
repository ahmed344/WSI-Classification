# Independent PANTHER classification pipeline

This package implements the original PANTHER workflow for the repository's
class-folder WSI feature layout. It has no imports from CLAM or any other model
pipeline. Its split policy intentionally reproduces CLAM's deterministic
slide-level split so their held-out results are directly comparable.

The stages are:

1. sample training tiles evenly across training slides and fit K-means prototypes;
2. fit a prototype-centred diagonal GMM independently to every slide by MAP-EM;
3. cache mixture weights (`pi`), means (`mean`), and diagonal variances
   (`variance`) in three separate slide-embedding files;
4. train one original bias-free linear classifier for each configured
   `output_type`, with `allcat` derived by concatenating the three caches;
5. evaluate at the native slide level and save reports, predictions, and confusion matrices;
6. reproduce the official categorical prototypical-assignment maps and GMM
   mixture-proportion plots for aligned tissue images.

From the repository root:

```bash
python -m panther.train_panther --config panther/config.yaml
python -m panther.evaluate_panther --config panther/config.yaml
python -m panther.visualize_assignments --config panther/config.yaml
python -m panther.compare_feature_models --config panther/config.yaml
pytest panther/test_integration.py -q
```

`compare_feature_models` trains the current configuration across every
feature extractor (`hoptimus`, `uni2h`, `genbio`) and prototype count
(`8`, `16`, `32`). Each of those nine runs fits the four configured heads
(`allcat`, `pi`, `mean`, `variance`) and writes
`feature_model_comparison.csv` plus `feature_model_comparison.md` under
`output_dir/panther/`. Matching completed runs are reused by default.

Each training run receives a dated directory under
`output_dir/panther/`. The directory contains the exact split manifest,
prototype pickle, `slide_embeddings_pi.pt`, `slide_embeddings_mean.pt`, and
`slide_embeddings_variance.pt`. Per-output checkpoints and histories are stored
under `models/<output_type>/`; the run-root `best_model.pth` is the multi-model
bundle whose primary head is the selected-epoch representation with the highest
validation balanced accuracy. Evaluation writes one directory per model
under `evaluation_results/<output_type>/`, plus a combined manifest in
`evaluation_results/`. Set `paths.run_dir` in the configuration to evaluate a
specific run; null selects the newest completed run.

The implementation follows Mahmood Lab's
[PANTHER repository](https://github.com/mahmoodlab/PANTHER) and the CVPR 2024
paper, *Morphological Prototyping for Unsupervised Slide Representation Learning
in Computational Pathology*.

The visualization command follows the official PANTHER notebook: tile colors
are the hard labels obtained from the slide GMM posterior, and the accompanying
bars are the fitted mixture weights. Because HE-MYO stores several cropped
tissue images for one slide, one slide-level GMM is fit to the concatenated bag
and its aligned assignments are rendered on each tissue crop. Outputs and a
machine-readable diagnostic manifest are written under the selected run's
`visualization_results/` directory.

