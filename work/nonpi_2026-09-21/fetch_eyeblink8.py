"""Download the public author's archive, refusing HTML/security responses."""
from pathlib import Path
import hashlib, json, requests, zipfile

OUT = Path(__file__).resolve().parent
DEST = OUT.parents[1] / 'data/external/eyeblink8_official.zip'
URL = 'https://www.blinkingmatters.com/files/upload/research/eyeblink8.zip'

def main():
    meta = {'url': URL, 'destination': str(DEST)}
    try:
        with requests.get(URL, stream=True, timeout=(30, 60)) as response:
            meta.update(status=response.status_code, content_type=response.headers.get('Content-Type'),
                        final_url=response.url)
            response.raise_for_status()
            chunks = response.iter_content(1024 * 1024)
            first = next(chunks)
            if not first.startswith(b'PK'):
                meta.update(state='blocked_non_zip', response_excerpt=first[:500].decode('utf8', errors='replace'))
            else:
                DEST.parent.mkdir(parents=True, exist_ok=True)
                partial = DEST.with_suffix('.download')
                h = hashlib.sha256(); size = 0
                with partial.open('wb') as f:
                    for chunk in (first,):
                        f.write(chunk); h.update(chunk); size += len(chunk)
                    for chunk in chunks:
                        f.write(chunk); h.update(chunk); size += len(chunk)
                        if size > 3_000_000_000:
                            raise RuntimeError('Archive exceeds 3 GB budget; partial kept for inspection')
                with zipfile.ZipFile(partial) as z:
                    meta['members'] = len(z.infolist())
                    meta['videos'] = [i.filename for i in z.infolist() if i.filename.lower().endswith(('.avi','.mp4'))]
                partial.replace(DEST)
                meta.update(state='downloaded', sha256=h.hexdigest(), bytes=size)
    except Exception as exc:
        meta.update(state='failed', error=repr(exc))
    (OUT/'external_download.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    print(json.dumps(meta, indent=2), flush=True)

if __name__ == '__main__': main()
