from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlparse, urljoin
import requests
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models/DeepSeek-R1-Distill-Qwen-14B-AWQ'
REPO = 'casperhansen/deepseek-r1-distill-qwen-14b-awq'
REVISION = 'bc43ec1bbf08de53452630806d5989208b4186db'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def download(meta):
    name = meta['path']
    if name.startswith('.'):
        return
    target = DEST / name
    part = target.with_suffix(target.suffix + '.partial')
    expected = meta.get('lfs', {}).get('oid')
    if target.exists() and target.stat().st_size == meta['size'] and (not expected or sha256(target) == expected):
        print('verified', name, flush=True)
        return
    for attempt in range(8):
        try:
            offset = part.stat().st_size if part.exists() else 0
            host = 'https://huggingface.co' if attempt % 2 == 0 else 'https://hf-mirror.com'
            url = f'{host}/{REPO}/resolve/{REVISION}/{name}?download=true'
            session = requests.Session()
            for _ in range(5):
                host_name = urlparse(url).hostname or ''
                session.trust_env = host_name in ('huggingface.co', 'hf-mirror.com')
                response = session.get(url, headers={'Range': f'bytes={offset}-'} if offset else {}, allow_redirects=False, stream=True, timeout=(20, 90))
                if response.is_redirect:
                    url = urljoin(url, response.headers['Location'])
                    response.close()
                else:
                    break
            with response:
                response.raise_for_status()
                if response.status_code != 206:
                    offset = 0
                written = offset
                last = written
                with part.open('ab' if offset else 'wb') as f:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        f.write(chunk)
                        written += len(chunk)
                        if written - last > 256 * 1024 * 1024:
                            print(name, round(written / meta['size'] * 100, 1), '%', flush=True)
                            last = written
            if part.stat().st_size != meta['size']:
                raise ValueError('size mismatch')
            if expected and sha256(part) != expected:
                raise ValueError('sha256 mismatch')
            part.replace(target)
            print('verified', name, flush=True)
            return
        except Exception as e:
            print(name, 'attempt', attempt + 1, type(e).__name__, flush=True)
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f'download failed: {name}')
if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'results/deepseek_files.json').read_text())
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(download, manifest))
    (DEST / 'download_verified.json').write_text(json.dumps({'repo': REPO, 'revision': REVISION, 'files': manifest}, indent=2))
