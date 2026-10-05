"""Safely import the best/ folder downloaded from the user's own Colab run."""
import argparse
import shutil
import zipfile
from pathlib import Path

from backend.inference import validate_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
parser.add_argument('--destination', type=Path, required=True)
args = parser.parse_args()
allowed = {'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt'}
if args.destination.exists():
    raise SystemExit('The destination already exists. Use a new folder; no checkpoint will be overwritten.')
if args.source.is_dir():
    validate_checkpoint(args.source)
    args.destination.mkdir(parents=True)
    for name in allowed:
        src = args.source / name
        if src.is_file() and not src.is_symlink():
            shutil.copy2(src, args.destination / name)
else:
    with zipfile.ZipFile(args.source) as archive:
        files = [x for x in archive.infolist() if not x.is_dir() and Path(x.filename).name in allowed]
        parents = {str(Path(x.filename).parent) for x in files}
        if len(parents) != 1 or len({Path(x.filename).name for x in files}) != len(files):
            raise SystemExit('The archive must contain exactly one complete best/ folder.')
        if sum(x.file_size for x in files) > 2_000_000_000:
            raise SystemExit('Unexpected archive size; inspect before import.')
        if any('..' in Path(x.filename).parts or Path(x.filename).is_absolute() or ((x.external_attr >> 16) & 0o170000) == 0o120000 for x in files):
            raise SystemExit('Unsafe archive path or symbolic link.')
        args.destination.mkdir(parents=True)
        for entry in files:
            with archive.open(entry) as src, (args.destination / Path(entry.filename).name).open('wb') as dst:
                shutil.copyfileobj(src, dst)
validate_checkpoint(args.destination)
print('Checkpoint imported with the exact 77-label mapping. Weight loading and prediction parity still require verification.')
