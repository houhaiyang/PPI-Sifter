#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
from esm.tokenization import get_esmc_model_tokenizers

MODEL_PATH = "/home/share/huadjyin/home/houhaiyang/HF_HOME/transformers/EvolutionaryScale/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth"

def load_esmc_model(model_path: str, device: str = "cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ESMC(
        d_model=1152,
        n_heads=18,
        n_layers=36,
        tokenizer=get_esmc_model_tokenizers(),
    ).eval().to(device)

    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)

    if device.type != "cpu":
        model = model.to(torch.bfloat16)

    return model, device

def get_residue_embedding(model, device, sequence: str):
    with torch.no_grad():
        protein = ESMProtein(sequence=sequence)
        input_ids = model._tokenize([protein.sequence]).to(device)
        output = model(input_ids)
        emb = output.embeddings

        emb = emb.float().cpu().numpy()[0]

        seq_len = len(sequence)
        total_len = emb.shape[0]

        if total_len == seq_len:
            residue_emb = emb
        elif total_len == seq_len + 2:
            residue_emb = emb[1:-1]
        elif total_len == seq_len + 1:
            residue_emb = emb[1:]
        else:
            residue_emb = emb[:seq_len]

    return residue_emb

def process_fasta(input_faa: str, outdir: str, model, device):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    for record in SeqIO.parse(input_faa, "fasta"):
        seq_id = record.id
        sequence = str(record.seq)

        npy_file = os.path.join(outdir, f"{seq_id}.residue_emb.npy")
        csv_file = os.path.join(outdir, f"{seq_id}.residue_emb.csv.gz")

        if os.path.exists(npy_file):
            continue

        try:
            residue_emb = get_residue_embedding(model, device, sequence)

            np.save(npy_file, residue_emb)

            df = pd.DataFrame(
                residue_emb,
                index=[f"res_{i+1}" for i in range(residue_emb.shape[0])]
            )
            df.to_csv(csv_file, index=True, header=False, compression="gzip")

        except Exception as e:
            print(f"Error processing {seq_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate residue embeddings for protein sequences in a FASTA file.")
    parser.add_argument("--input", required=True, help="Input FASTA file (.faa)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")

    args = parser.parse_args()

    print("Loading ESMC model...")
    model, device = load_esmc_model(MODEL_PATH, args.device)
    print(f"Model loaded on {device}")

    print(f"Processing sequences from {args.input}...")
    process_fasta(args.input, args.outdir, model, device)
    print("Done.")

if __name__ == "__main__":
    main()