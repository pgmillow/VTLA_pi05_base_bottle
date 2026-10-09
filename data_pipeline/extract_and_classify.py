"""Extract the verified bottle dataset, then apply the existing /data classifier."""
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path('/data/open_bottle_cap_vtla_20261006')
ARCHIVE = Path('/data/拧瓶盖/open_bottle_cap.zip')


def main():
    raw = ROOT / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        entries = archive.infolist()
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != 'open_bottle_cap':
                raise ValueError(f'Unexpected archive path: {entry.filename}')
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f'Unexpected symlink: {entry.filename}')
            if not entry.is_dir() and path.suffix != '.mcap':
                raise ValueError(f'Unexpected file type: {entry.filename}')
        for number, entry in enumerate(entries, 1):
            target = raw / entry.filename
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not target.exists():
                temporary = target.with_suffix('.mcap.part')
                with archive.open(entry) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output, length=8*1024*1024)
                os.replace(temporary, target)
            if target.stat().st_size != entry.file_size:
                raise ValueError(f'Extracted size mismatch: {target}')
            manifest.append({'file': entry.filename, 'bytes': entry.file_size, 'zip_crc32': entry.CRC})
            print(f'Extracted {number}/{len(entries)}: {entry.filename}', flush=True)
    (ROOT / 'archive_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    subprocess.run([sys.executable, str(PROJECT/'data_pipeline/canonical/classify_mcap_quality.py'),
                    str(raw/'open_bottle_cap'), '--out-dir', str(ROOT/'audit/classification'),
                    '--converter', str(PROJECT/'data_pipeline/canonical/mcap_to_lerobot_v21.py')], check=True)


if __name__ == '__main__':
    main()
