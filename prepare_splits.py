"""Create a fixed, stratified 60/20/20 split for Bail, Credit or German.

Input data/<dataset>_raw.npz needs x, edge_index, y, sensitive.
Do not use this to replace existing NIFTY/FairGNN/EDITS splits for direct comparisons.
"""
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


def make_split(y, sensitive, seed=42):
    y = np.asarray(y).reshape(-1)
    sensitive = np.asarray(sensitive).reshape(-1)
    ids = np.arange(len(y))
    # Prefer balancing both class and sensitive group; fall back to class if rare strata cannot be split.
    joint = 2 * y.astype(int) + sensitive.astype(int)
    def stratify(subset):
        vals, counts = np.unique(joint[subset], return_counts=True)
        if len(vals) > 1 and counts.min() >= 5:
            return joint[subset]
        vals, counts = np.unique(y[subset], return_counts=True)
        if len(vals) > 1 and counts.min() >= 5:
            return y[subset]
        raise ValueError('Not enough examples per class to construct a reliable stratified split')
    train, rest = train_test_split(ids, train_size=.6, random_state=seed, stratify=stratify(ids))
    valid, test = train_test_split(rest, test_size=.5, random_state=seed, stratify=stratify(rest))
    return {'train_idx': np.sort(train), 'val_idx': np.sort(valid), 'test_idx': np.sort(test)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['bail', 'credit', 'german'], required=True)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--split-seed', type=int, default=42)
    args = p.parse_args()
    root = Path(args.data_dir)
    src = root / f'{args.dataset}_raw.npz'
    dst = root / f'{args.dataset}.npz'
    if dst.exists():
        raise FileExistsError(f'{dst} already exists; refusing to overwrite splits')
    with np.load(src, allow_pickle=False) as file:
        arrays = {key: file[key] for key in file.files}
    required = {'x', 'edge_index', 'y', 'sensitive'}
    if not required.issubset(arrays):
        raise ValueError(f'Missing fields: {sorted(required - arrays.keys())}')
    if not np.all(np.isin(arrays['y'], [0, 1])) or not np.all(np.isin(arrays['sensitive'], [0, 1])):
        raise ValueError('Expected binary y and sensitive values (0 or 1)')
    arrays.update(make_split(arrays['y'], arrays['sensitive'], args.split_seed))
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **arrays)
    for key in ('train_idx', 'val_idx', 'test_idx'):
        ix = arrays[key]
        yy = arrays['y'][ix]
        ss = arrays['sensitive'][ix]
        print(f'{key}: n={len(ix)}, positives={int(yy.sum())}, sensitive_1={int(ss.sum())}')
    print(f'Saved {dst} (split seed={args.split_seed})')


if __name__ == '__main__':
    main()
