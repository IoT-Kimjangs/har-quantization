"""Verify the official PAMAP2 archive and extract the nested dataset safely."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/pamap2'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def safe_extract(archive, destination):
    root=Path(destination).resolve()
    for item in archive.infolist():
        name=PurePosixPath(item.filename)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError(f'Unsafe archive member path: {item.filename}')
        target=(root/Path(*name.parts)).resolve()
        if target!=root and not target.is_relative_to(root):
            raise ValueError(f'Archive member escapes destination: {item.filename}')
    archive.extractall(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--archive',type=Path,default=Path('data/raw/pamap2/pamap2_official.zip')); args=parser.parse_args()
    path=args.archive if args.archive.is_absolute() else ROOT/args.archive
    config=json.loads((ROOT/'configs/pamap2-data.json').read_text(encoding='utf-8'))
    if not path.is_file():raise FileNotFoundError(f'Download the official archive first: {path}')
    actual=sha(path)
    if path.stat().st_size!=config['archive_bytes'] or actual!=config['archive_sha256']:
        raise ValueError(f'Official PAMAP2 archive does not match pinned size/hash: {path.stat().st_size}, {actual}')
    RAW.mkdir(parents=True,exist_ok=True)
    dataset=RAW/'PAMAP2_Dataset'
    outer=zipfile.ZipFile(path)
    if set(outer.namelist())!={'PAMAP2_Dataset.zip','readme.pdf'}:
        raise ValueError('Unexpected official outer ZIP member list')
    inner_path=RAW/'PAMAP2_Dataset.zip'
    for name in outer.namelist():
        target=RAW/name
        if target.exists():
            if name=='PAMAP2_Dataset.zip' and sha(target)==sha_from_zip(outer,name):continue
            if name=='readme.pdf':continue
            raise FileExistsError(f'Refusing to overwrite {target}')
        with outer.open(name) as src,target.open('wb') as dst:
            while chunk:=src.read(1024*1024):dst.write(chunk)
    if not dataset.exists():
        with zipfile.ZipFile(inner_path) as inner:safe_extract(inner,RAW)
    files=sorted(dataset.rglob('*.dat'))
    if len(files)!=14:raise ValueError(f'Expected 14 author-notebook .dat files; found {len(files)}')
    summary={'dataset':config['dataset'],'archive_bytes':path.stat().st_size,'archive_sha256':actual,
             'extracted_dat_files':len(files),'protocol_files':sum('Protocol' in f.parts for f in files),
             'optional_files':sum('Optional' in f.parts for f in files),
             'raw_root':dataset.relative_to(ROOT).as_posix(),
             'license_source':'UCI dataset page lists CC BY 4.0; retain the bundled readme.pdf.'}
    output=ROOT/'results/data/pamap2-raw-summary.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def sha_from_zip(archive,name):
    h=hashlib.sha256()
    with archive.open(name) as stream:
        while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()


if __name__=='__main__':main()
