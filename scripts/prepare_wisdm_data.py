"""Verify official WISDM files and report raw-data quality without preprocessing."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(manifest, download=False):
    directory = ROOT / manifest['local_directory']
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / manifest['archive_name']
    if not archive_path.exists():
        if not download:
            raise ValueError(f"압축 파일이 없습니다: {archive_path}. --download 또는 공식 브라우저 다운로드를 사용하세요.")
        temporary = archive_path.with_suffix('.partial')
        try:
            with urllib.request.urlopen(manifest['download_url'], timeout=60) as response:
                with temporary.open('wb') as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if digest(temporary) != manifest['archive_sha256']:
                raise ValueError('다운로드 해시가 다릅니다. 배포 파일 변경 여부를 확인하세요.')
            temporary.replace(archive_path)
        finally:
            temporary.unlink(missing_ok=True)
    if digest(archive_path) != manifest['archive_sha256']:
        raise ValueError('압축 파일 해시가 다릅니다. 기존 파일은 변경하지 않았습니다.')
    # Read only the exact expected regular files; never extract archive paths blindly.
    with tarfile.open(archive_path, 'r:gz') as archive:
        for name, expected_hash in manifest['files'].items():
            member = archive.getmember(name)
            if not member.isfile() or '..' in Path(name).parts or Path(name).is_absolute():
                raise ValueError(f'허용하지 않는 압축 항목: {name}')
            with archive.extractfile(member) as stream:
                content = stream.read()
            if hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError(f'내부 파일 해시가 다릅니다: {name}')
            target = directory / name
            if target.exists():
                if digest(target) != expected_hash:
                    raise ValueError(f'기존 파일이 원본과 다릅니다. 덮어쓰지 않습니다: {name}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    return directory / manifest['raw_file']


def summarize(path, manifest):
    counts = Counter()
    activities = Counter()
    users = set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            counts['physical_lines'] += 1
            value = line.strip()
            if not value:
                counts['blank_lines'] += 1
                continue
            parts = value.removesuffix(';').split(',')
            if len(parts) != 6:
                counts['wrong_field_count'] += 1
                continue
            user, activity, timestamp, x, y, z = parts
            try:
                numbers = [float(item) for item in (timestamp, x, y, z)]
            except ValueError:
                counts['invalid_numeric'] += 1
                continue
            if not all(math.isfinite(item) for item in numbers):
                counts['nonfinite_numeric'] += 1
                continue
            if numbers[0] <= 0:
                counts['nonpositive_timestamp'] += 1
                continue
            if activity not in manifest['label_order']:
                counts['unknown_activity'] += 1
                continue
            counts['valid_lines'] += 1
            activities[activity] += 1
            users.add(user)
    keys = ('physical_lines', 'blank_lines', 'wrong_field_count', 'invalid_numeric',
            'nonfinite_numeric', 'nonpositive_timestamp', 'unknown_activity', 'valid_lines')
    return {
        'dataset': manifest['dataset'],
        'source_page': manifest['source_page'],
        'raw_file': str(path.relative_to(ROOT)),
        'raw_sha256': digest(path),
        'counts': {key: counts[key] for key in keys},
        'valid_user_count': len(users),
        'valid_activity_counts': {key: activities[key] for key in manifest['label_order']},
        'published_record_count': manifest['published_record_count'],
        'parsing_rule': 'one physical line per record; trailing semicolon removed; six fields; finite numeric timestamp/x/y/z; positive timestamp; known activity',
        'raw_modified': False,
        'split_created': False,
        'scaler_fitted': False,
        'prediction_executed': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='원본 압축 파일이 없으면 공식 서버에서 다운로드')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/wisdm-data.json').read_text())
    report = summarize(prepare(manifest, args.download), manifest)
    output = ROOT / 'results/data/wisdm-raw-summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'검증 기록: {output.relative_to(ROOT)}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        print(f'준비 실패: {error}', file=sys.stderr)
        raise SystemExit(1)
