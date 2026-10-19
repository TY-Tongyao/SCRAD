import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import os
import time
import requests
ROOT = Path(__file__).resolve().parents[1]
REDDIT_STORAGE_ORIGIN = 'https://s3.us-west-2.amazonaws.com/dgl-data/dataset/reddit.zip'
FILES = {'reddit': ('reddit_graphsage/reddit.zip', 'https://data.dgl.ai/dataset/reddit.zip', '9a16353c28f8ddd07148fc5ac9b57b818d7911ea0fbe9052d66d49fc32b372bf'), 'questions': ('questions_original.npz', 'https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/questions.npz', '757ebd772bab1475c4dd951ca9e364400c6db161656cff9d21780ee874cf3074')}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def download(url, target, workers=16):
    response = requests.head(url, allow_redirects=True, timeout=(20, 90))
    response.raise_for_status()
    size = int(response.headers['Content-Length'])
    parts = target.with_name(target.name + '.parts')
    parts.mkdir(exist_ok=True)
    prefix = target.with_name(target.name + '.download')
    start = 0
    if prefix.exists():
        start = prefix.stat().st_size
        if start > size:
            raise RuntimeError('Partial download exceeds source size')
    chunk_size = 8 * 1024 * 1024
    ranges = [(offset, min(offset + chunk_size, size) - 1) for offset in range(start, size, chunk_size)]

    def fetch(bounds):
        first, last = bounds
        path = parts / f'{first:012d}-{last:012d}.part'
        expected = last - first + 1
        for attempt in range(6):
            have = path.stat().st_size if path.exists() else 0
            if have == expected:
                return expected
            if have > expected:
                raise RuntimeError(f'Oversized range: {path}')
            try:
                with requests.get(url, headers={'Range': f'bytes={first + have}-{last}'}, stream=True, timeout=(20, 120)) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {first + have}-{last}/{size}':
                        raise RuntimeError('Source did not honor requested byte range')
                    with path.open('ab') as out:
                        for block in r.iter_content(256 * 1024):
                            out.write(block)
                if path.stat().st_size != expected:
                    raise RuntimeError('Incomplete range')
                return expected
            except (requests.RequestException, RuntimeError):
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 15))
    completed = start
    print(json.dumps({'stage': 'download', 'url': url, 'bytes': size, 'workers': workers}), flush=True)
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, bounds) for bounds in ranges]):
            completed += future.result()
            if time.monotonic() - last_report >= 15 or completed == size:
                print(json.dumps({'downloaded_bytes': completed, 'total_bytes': size, 'percent': round(100 * completed / size, 2)}), flush=True)
                last_report = time.monotonic()
    assembled = target.with_name(target.name + '.assembled')
    with assembled.open('wb') as out:
        sources = ([prefix] if start else []) + [parts / f'{first:012d}-{last:012d}.part' for first, last in ranges]
        for source in sources:
            with source.open('rb') as inp:
                for block in iter(lambda: inp.read(8 * 1024 * 1024), b''):
                    out.write(block)
    if assembled.stat().st_size != size:
        raise RuntimeError('Assembled archive has incorrect size')
    os.replace(assembled, target)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['reddit', 'questions', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--reddit-source', choices=['s3', 'dgl'], default='s3', help="Use DGL's public storage origin or CDN for the identical archive")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    for name in FILES if args.dataset == 'all' else [args.dataset]:
        filename, url, expected = FILES[name]
        target = ROOT / 'data/raw' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        download_url = REDDIT_STORAGE_ORIGIN if name == 'reddit' and args.reddit_source == 's3' else url
        if not target.exists():
            download(download_url, target, args.workers)
        actual = sha256(target)
        if expected is not None and actual != expected:
            raise RuntimeError(f'{filename} differs from the recorded canonical source; preserve and investigate')
        parts = target.with_name(target.name + '.parts')
        if parts.exists():
            for part in parts.glob('*.part'):
                part.unlink()
            if not any(parts.iterdir()):
                parts.rmdir()
        target.with_name(target.name + '.download').unlink(missing_ok=True)
        manifest = {'dataset_id': 'graphsage_reddit_232965' if name == 'reddit' else 'questions_48921', 'url': url, 'download_url': download_url, 'bytes': target.stat().st_size, 'sha256': actual}
        manifest_path = target.parent / 'download_manifest.json' if name == 'reddit' else target.with_suffix('.download.json')
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(manifest), flush=True)
if __name__ == '__main__':
    main()
