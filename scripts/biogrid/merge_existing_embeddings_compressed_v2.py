#!/usr/bin/env python3
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from Bio import SeqIO
from tqdm import tqdm


def save_npz_batch(batch_dict, out_file: Path, compress: bool = True, compresslevel: int = 1):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(out_file, mode="w", compression=compression, compresslevel=compresslevel if compress else None) as zf:
        for seq_id, arr in batch_dict.items():
            bio = io.BytesIO()
            np.save(bio, arr, allow_pickle=False)
            zf.writestr(f"{seq_id}.npy", bio.getvalue())


def load_done_ids(outdir: Path):
    manifest = outdir / "done_ids.txt"
    done = set()
    if manifest.exists():
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(line)
        return done

    for f in sorted(outdir.glob("batch_*.npz")):
        try:
            data = np.load(f, allow_pickle=False)
            done.update([k[:-4] if k.endswith('.npy') else k for k in data.files])
        except Exception:
            pass
    return done


def append_done_ids(outdir: Path, ids):
    manifest = outdir / "done_ids.txt"
    with open(manifest, "a", encoding="utf-8") as f:
        for seq_id in ids:
            f.write(seq_id + "\n")


def save_batch(batch_items, outdir: Path, batch_idx: int, compress: bool, compresslevel: int):
    if not batch_items:
        return
    batch_dict = dict(batch_items)
    out_file = outdir / f"batch_{batch_idx:06d}.npz"
    save_npz_batch(batch_dict, out_file, compress=compress, compresslevel=compresslevel)
    append_done_ids(outdir, batch_dict.keys())



def merge_existing_embeddings(input_faa: str, embeddings_dir: str, outdir: str, batch_size: int = 200, dtype: str = "float16", compress: bool = True, compresslevel: int = 1, delete_source: bool = False):
    outdir = Path(outdir)
    embdir = Path(embeddings_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    done = load_done_ids(outdir)
    seq_ids = [r.id for r in SeqIO.parse(input_faa, 'fasta')]
    existing_batches = sorted(outdir.glob('batch_*.npz'))
    batch_idx = len(existing_batches)

    target_dtype = np.float16 if dtype == 'float16' else np.float32
    batch = []
    source_files = []
    total = 0
    skipped = 0

    for seq_id in tqdm(seq_ids, desc='Merging existing embeddings', unit='seq'):
        if seq_id in done:
            skipped += 1
            continue
        f = embdir / f'{seq_id}.residue_emb.npy'
        if not f.exists():
            continue
        try:
            emb = np.load(f, allow_pickle=False)
            if emb.ndim != 2:
                continue
            emb = emb.astype(target_dtype, copy=False)
            batch.append((seq_id, emb))
            source_files.append(f)
            total += 1

            if len(batch) >= batch_size:
                save_batch(batch, outdir, batch_idx, compress, compresslevel)
                if delete_source:
                    for sf in source_files:
                        try:
                            sf.unlink()
                        except Exception:
                            pass
                batch_idx += 1
                batch = []
                source_files = []
        except Exception as e:
            print(f'Error reading {seq_id}: {e}')

    if batch:
        save_batch(batch, outdir, batch_idx, compress, compresslevel)
        if delete_source:
            for sf in source_files:
                try:
                    sf.unlink()
                except Exception:
                    pass

    meta = {
        'input_faa': input_faa,
        'embeddings_dir': embeddings_dir,
        'outdir': outdir.as_posix(),
        'batch_size': batch_size,
        'dtype': dtype,
        'compress': compress,
        'compresslevel': compresslevel,
        'total_new_merged': total,
        'skipped_existing': skipped,
    }
    with open(outdir / 'merge_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'Merged {total} new embeddings into {outdir}')
    print(f'Skipped {skipped} embeddings already recorded in done_ids.txt or existing batches')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Merge existing single-sequence residue embeddings into batched NPZ files with resume support')
    p.add_argument('--input', required=True, help='Input FASTA file')
    p.add_argument('--embeddings_dir', required=True, help='Directory with single-sequence .npy files')
    p.add_argument('--outdir', required=True, help='Output directory for batched .npz files')
    p.add_argument('--batch_size', type=int, default=200, help='Sequences per batch; smaller is faster and safer for compression')
    p.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    p.add_argument('--no_compress', action='store_true', help='Store NPZ without zip compression for maximum speed')
    p.add_argument('--compresslevel', type=int, default=1, help='Zip compression level, 1=fastest, 9=smallest')
    p.add_argument('--delete_source', action='store_true', help='Delete source single-sequence files after successful batch save')
    args = p.parse_args()

    merge_existing_embeddings(
        input_faa=args.input,
        embeddings_dir=args.embeddings_dir,
        outdir=args.outdir,
        batch_size=args.batch_size,
        dtype=args.dtype,
        compress=not args.no_compress,
        compresslevel=args.compresslevel,
        delete_source=args.delete_source,
    )
