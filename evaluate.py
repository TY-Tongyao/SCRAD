import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import numpy as np
    from sklearn.metrics import f1_score
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    with np.load(args.labels, allow_pickle=False) as data:
        ids, labels = (data['node_ids'], data['y'])
    if len(rows) != len(ids):
        raise ValueError('Incomplete test predictions')
    np.testing.assert_array_equal([row['id'] for row in rows], ids)
    votes = np.asarray([row['sequence_labels'] for row in rows])
    if votes.shape != (len(ids), 9) or not np.isin(votes, [0, 1]).all():
        raise ValueError('Invalid binary decisions')
    scores = votes.mean(axis=1)
    np.testing.assert_array_equal([row['score'] for row in rows], scores)
    predictions = scores >= 0.5
    np.testing.assert_array_equal([row['predicted_label'] for row in rows], predictions)
    value = float(f1_score(labels, predictions, average='macro', labels=[0, 1], zero_division=0))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'macro_f1': value}) + '\n')
    print(f'Macro-F1: {value * 100:.4f}%')
if __name__ == '__main__':
    main()
