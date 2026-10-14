"""Audit local CV-job CSVs without printing or redistributing their contents.

Usage: python audit_dataset.py /path/train.csv /path/test.csv --output audit.json
No external packages, training, downloads, or network requests are required.
Exact-text counts are reproducible. Normalized counts are a supplementary check.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ('resume_text', 'job_description_text', 'label')
LABELS = {'No Fit', 'Potential Fit', 'Good Fit'}

def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing required columns: {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty dataset file')
    if any(row.get(key) is None for row in rows for key in FIELDS):
        raise ValueError('Malformed CSV rows or missing field values')
    invalid = set(row['label'] for row in rows) - LABELS
    if invalid:
        raise ValueError(f'Unexpected label values: {sorted(invalid)}')
    return rows

def norm(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()

def audit(train_path, test_path):
    parts = {'train': read_csv(train_path), 'test': read_csv(test_path)}
    rows = parts['train'] + parts['test']
    pairs = defaultdict(list)
    for row in rows:
        pairs[(row[FIELDS[0]], row[FIELDS[1]])].append(row['label'])
    conflicts = {key: labels for key, labels in pairs.items() if len(set(labels)) > 1}
    normalized_pairs = defaultdict(set)
    for row in rows:
        normalized_pairs[(norm(row[FIELDS[0]]), norm(row[FIELDS[1]]))].add(row['label'])
    out = {
        'rows': len(rows),
        'split_rows': {name: len(part) for name, part in parts.items()},
        'classes': dict(Counter(row['label'] for row in rows)),
        'unique_resumes': len({row[FIELDS[0]] for row in rows}),
        'unique_jobs': len({row[FIELDS[1]] for row in rows}),
        'unique_pairs': len(pairs),
        'duplicate_pair_excess': len(rows) - len(pairs),
        'conflicting_pairs': len(conflicts),
        'conflicting_rows': sum(map(len, conflicts.values())),
        'clean_unique_pairs_before_normalization': len(pairs) - len(conflicts),
        'empty_or_whitespace': sum(any(not row[key].strip() for key in FIELDS) for row in rows),
        'normalized_unique_pairs': len(normalized_pairs),
        'normalized_conflicting_pairs': sum(len(v) > 1 for v in normalized_pairs.values()),
        'sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                   for name, path in [('train', train_path), ('test', test_path)]},
    }
    for label, key in [('resume', FIELDS[0]), ('job', FIELDS[1])]:
        out[label + '_overlap'] = len({row[key] for row in parts['train']} & {row[key] for row in parts['test']})
        out['normalized_' + label + '_overlap'] = len({norm(row[key]) for row in parts['train']} & {norm(row[key]) for row in parts['test']})
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_csv', type=Path)
    parser.add_argument('test_csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path('audit.json'))
    args = parser.parse_args()
    result = audit(args.train_csv, args.test_csv)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
