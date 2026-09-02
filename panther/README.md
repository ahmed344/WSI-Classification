# Independent PANTHER classification pipeline

This package implements the original PANTHER workflow for the repository's
class-folder WSI feature layout. It has no imports from CLAM or any other model
pipeline. Its split policy intentionally reproduces CLAM's deterministic
slide-level split so their held-out results are directly comparable.

The stages are:

1. sample training tiles evenly across training slides and fit K-means prototypes;
2. fit a prototype-centred diagonal GMM independently to every slide by MAP-EM;
3. concatenate mixture weights, means, and variances (`allcat`);
4. train the original bias-free linear downstream classifier;
5. evaluate at the native slide level and save reports, predictions, and confusion matrices;
6. reproduce the official categorical prototypical-assignment maps and GMM
   mixture-proportion plots for aligned tissue images.

From the repository root:

```bash
python -m panther.train_panther --config panther/config.yaml
python -m panther.evaluate_panther --config panther/config.yaml
python -m panther.visualize_assignments --config panther/config.yaml
pytest panther/test_integration.py -q
```

Each training run receives a dated directory under
`output_dir/panther/`. The directory contains the exact split manifest,
prototype pickle, cached slide representations, training history, final and
best-validation classifier states, the selected `best_model.pth`, and an
`evaluation_results/` directory after evaluation. Set `paths.run_dir` in the
configuration to evaluate a specific run; null selects the newest completed run.

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

