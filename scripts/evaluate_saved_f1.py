import argparse
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['reddit', 'questions'], required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    predictions = args.predictions or root / 'reference' / args.dataset / 'predictions.jsonl'
    labels = args.labels or root / 'artifacts' / args.dataset / 'evaluation.npz'
    for path in [predictions, labels]:
        if not path.is_file():
            parser.error(f'Missing external asset: {path}. Use --predictions and --labels to specify prepared files.')
    subprocess.run([sys.executable, str(root / 'evaluate.py'), '--predictions', str(predictions), '--labels', str(labels)], check=True)

if __name__ == '__main__':
    main()
