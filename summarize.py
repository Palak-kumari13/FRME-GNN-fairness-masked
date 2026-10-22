"""Summarize completed Stage 2 runs as mean ± sample SD, with run counts."""
import argparse
import csv
from collections import defaultdict
import numpy as np

METRICS = ['f1', 'auc', 'parity', 'equality', 'mask_mean', 'mask_frac_gt_05', 'mask_exposure_gap']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv_path')
    p.add_argument('--output', default='summary.csv')
    args = p.parse_args()
    groups = defaultdict(list)
    with open(args.csv_path, newline='') as f:
        for row in csv.DictReader(f):
            groups[(row['dataset'], row['model'])].append(row)
    with open(args.output, 'w', newline='') as f:
        columns = ['dataset', 'model', 'n'] + [f'{k}_{s}' for k in METRICS for s in ('mean', 'sd')]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for (dataset, model), rows in sorted(groups.items()):
            if len(set(r['seed'] for r in rows)) != len(rows):
                raise ValueError(f'Duplicate seeds for {dataset} {model}; use a fresh results directory')
            out = {'dataset': dataset, 'model': model, 'n': len(rows)}
            for k in METRICS:
                values = np.asarray([float(r[k]) for r in rows], dtype=float)
                values = values[np.isfinite(values)]
                out[f'{k}_mean'] = round(float(values.mean()), 6) if len(values) else ''
                out[f'{k}_sd'] = round(float(values.std(ddof=1)), 6) if len(values) > 1 else ''
            writer.writerow(out)
            print(dataset, model, 'n=', len(rows), 'F1=', out['f1_mean'], 'AUC=', out['auc_mean'])
    print('Saved', args.output)


if __name__ == '__main__':
    main()
