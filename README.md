# FRME-GNN — Stage 2

A0–A2 are unchanged from Stage 1. A3 adds fairness-aware training to the learned edge mask. Code uses PyTorch only; no PyG dependency.

## Experiments

| Run | Model | Loss |
| --- | --- | --- |
| A0 | GCN, all edges | Cross-entropy |
| A1 | Learned edge mask | Cross-entropy |
| A2 | A1 + A0 teacher | Cross-entropy + teacher KL + mask sparsity |
| A3 | A2 + fairness training | A2 + soft demographic-parity gap + soft equal-opportunity gap + mask-exposure disparity |

A3 is a **testable fairness regularization baseline**, sensitive-direction projector (novel method). The mask-exposure term compares the mean selected-edge weight incident to each *training* node, averaged over two groups. Low exposure disparity alone does not establish explanation fairness or mask correctness.

## Data

Use the exact **Bail, Credit, German** graphs and train/validation/test indices from the NIFTY Vanilla, NIFTY SSF, FairGNN (two run sets), and EDITS comparisons. Keep seeds **0–4**. `data/<dataset>.npz` must include:

- `x` (N,F), numeric node features
- `edge_index` (2,E or E,2), zero-based graph edges
- `y` (N), binary target labels 0/1
- `sensitive` (N), binary group labels 0/1
- `train_idx`, `val_idx`, `test_idx` (non-overlapping node IDs)

Optional files `<dataset>_seed0.npz`, ..., `<dataset>_seed4.npz` take priority when your original baselines used different splits by seed.

**Do not generate new splits for direct comparison with the prior baseline tables.** If you have no original indices, use `prepare_splits.py` to create one reproducible stratified 60/20/20 split, but rerun all older baselines on that same split. Stratification tries to preserve the label × group composition; it does not rebalance test labels artificially. Data and split files are not included in this repository.

## Run

```bash
pip install -r requirements.txt
python train.py --dataset bail --data-dir data --output results/bail
python train.py --dataset credit --data-dir data --output results/credit
python train.py --dataset german --data-dir data --output results/german
python summarize.py results/bail/stage2_results.csv --output results/bail/summary.csv
```

The default command trains A0→A3 for seeds 0–4. A2/A3 use the matching seed's frozen A0 prediction distribution on training nodes. For standalone A3 runs, supply an output folder containing A0 checkpoints first, and choose a new results folder after updating the code's teacher path if needed; the default workflow is recommended.

### Loss weights and ablation

```bash
python train.py --dataset bail --data-dir data --output results/bail_mask_ablation \
  --lambda-dp 1.0 --lambda-eo 1.0 --lambda-mask-fair 0.0
```

The mask-specific contribution is measured by comparing `--lambda-mask-fair 0` with a positive value, with everything else fixed. Select weights on validation data, never on test scores. A3's fairness losses require both groups and positive labels for both groups among training nodes; invalid training splits are rejected.

## Metrics and fairness interpretation

Each run writes test F1, ROC-AUC, demographic parity gap, equal opportunity gap, positive prediction rate, mask mean and mask fraction above 0.5. A3 additionally records the test mask-exposure disparity, which is only a *proxy* for group-wise explanation parity.

- High F1/AUC is good; low demographic parity and equal opportunity gaps are good *if utility is preserved*.
- Group prediction gaps and group mask-exposure gaps answer different questions.
- Stage 2 does **not** validate ground-truth explanation correctness, adversarial robustness, slow drift or geometric distillation.
- The robustness / counterfactual metrics in old baseline tables cannot be compared unless the same evaluation protocols are implemented.
- Always inspect per-group positive prediction rate, per-group TPR, and F1 for possible collapsed predictions.
- Summaries use sample SD (`ddof=1`) over five seeds.

## Next stage

After A3 succeeds, define a valid counterfactual or geometric sensitive-drift subspace and add **selective** mask regularization. Measure correctness on known-motif graphs separately. Then test structural attacks and slow drift. Do not claim robustness, explanation correctness or novelty from Stage 2 performance alone.
