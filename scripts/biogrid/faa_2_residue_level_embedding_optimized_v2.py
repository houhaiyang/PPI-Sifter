#!/usr/bin/env python3
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
from esm.tokenization import get_esmc_model_tokenizers

MODEL_PATH = "/home/share/huadjyin/home/houhaiyang/HF_HOME/transformers/EvolutionaryScale/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth"


def load_esmc_model(model_path: str, device: str = 'cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = ESMC(d_model=1152, n_heads=18, n_layers=36, tokenizer=get_esmc_model_tokenizers()).eval().to(device)
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    if device.type != 'cpu':
        model = model.to(torch.bfloat16)
    return model, device


def get_residue_embedding(model, device, sequence: str):
    with torch.no_grad():
        protein = ESMProtein(sequence=sequence)
        input_ids = model._tokenize([protein.sequence]).to(device)
        output = model(input_ids)
        emb = output.embeddings.float().cpu().numpy()[0]
        seq_len = len(sequence)
        if emb.shape[0] == seq_len:
            return emb
        if emb.shape[0] == seq_len + 2:
            return emb[1:-1]
        if emb.shape[0] == seq_len + 1:
            return emb[1:]
        return emb[:seq_len]


def save_npz_batch(batch_dict, out_file: Path, compress: bool = True, compresslevel: int = 1):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(out_file, mode='w', compression=compression, compresslevel=compresslevel if compress else None) as zf:
        for seq_id, arr in batch_dict.items():
            bio = io.BytesIO()
            np.save(bio, arr, allow_pickle=False)
            zf.writestr(f'{seq_id}.npy', bio.getvalue())


def load_done_ids(outdir: Path):
    manifest = outdir / 'done_ids.txt'
    done = set()
    if manifest.exists():
        with open(manifest, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(line)
        return done

    for f in sorted(outdir.glob('batch_*.npz')):
        try:
            data = np.load(f, allow_pickle=False)
            done.update([k[:-4] if k.endswith('.npy') else k for k in data.files])
        except Exception:
            pass
    return done


def append_done_ids(outdir: Path, ids):
    manifest = outdir / 'done_ids.txt'
    with open(manifest, 'a', encoding='utf-8') as f:
        for seq_id in ids:
            f.write(seq_id + '\n')


def process_fasta(input_faa: str, outdir: str, model, device, batch_size: int = 200, dtype: str = 'float16', compress: bool = True, compresslevel: int = 1):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    done = load_done_ids(outdir)
    batch_idx = len(list(outdir.glob('batch_*.npz')))
    target_dtype = np.float16 if dtype == 'float16' else np.float32

    batch = []
    total_new = 0
    total_skipped = 0

    for record in SeqIO.parse(input_faa, 'fasta'):
        seq_id = record.id
        if seq_id in done:
            total_skipped += 1
            continue
        try:
            emb = get_residue_embedding(model, device, str(record.seq)).astype(target_dtype, copy=False)
            batch.append((seq_id, emb))
            total_new += 1
            if len(batch) >= batch_size:
                batch_dict = dict(batch)
                save_npz_batch(batch_dict, outdir / f'batch_{batch_idx:06d}.npz', compress=compress, compresslevel=compresslevel)
                append_done_ids(outdir, batch_dict.keys())
                batch_idx += 1
                batch = []
        except Exception as e:
            print(f'Error processing {seq_id}: {e}')

    if batch:
        batch_dict = dict(batch)
        save_npz_batch(batch_dict, outdir / f'batch_{batch_idx:06d}.npz', compress=compress, compresslevel=compresslevel)
        append_done_ids(outdir, batch_dict.keys())

    meta = {
        'input_faa': input_faa,
        'outdir': outdir.as_posix(),
        'batch_size': batch_size,
        'dtype': dtype,
        'compress': compress,
        'compresslevel': compresslevel,
        'total_new_generated': total_new,
        'total_skipped_existing': total_skipped,
    }
    with open(outdir / 'generate_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'Generated {total_new} new embeddings and skipped {total_skipped} existing ones')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate residue embeddings in batched NPZ files with resume support')
    p.add_argument('--input', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch_size', type=int, default=200, help='Sequences per batch; smaller is faster and safer for compression')
    p.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    p.add_argument('--no_compress', action='store_true')
    p.add_argument('--compresslevel', type=int, default=1)
    args = p.parse_args()

    model, device = load_esmc_model(MODEL_PATH, args.device)
    process_fasta(
        input_faa=args.input,
        outdir=args.outdir,
        model=model,
        device=device,
        batch_size=args.batch_size,
        dtype=args.dtype,
        compress=not args.no_compress,
        compresslevel=args.compresslevel,
    )
