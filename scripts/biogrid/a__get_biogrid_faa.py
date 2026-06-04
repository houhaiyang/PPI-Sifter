
import os
import json
import requests
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from io import StringIO

file_path = 'data/BioGRID/BIOGRID-ALL-4.4.240.csv.gz'
out_dir = 'data/BioGRID/faa_chunks'
prefix = 'BIOGRID-ALL-4.4.240.uniprot'
out_meta = os.path.join(out_dir, f'{prefix}.uniprot_ids.csv')
out_failed = os.path.join(out_dir, f'{prefix}.failed_ids.csv')
out_long = os.path.join(out_dir, f'{prefix}.longer_than_esmc_limit.csv')
checkpoint_file = os.path.join(out_dir, f'{prefix}.checkpoint.json')

DEBUG_MODE = False
DEBUG_N = 100
ESMC_MAX_LEN = 1022
CHUNK_SIZE = 2000   # 每下载成功累计到 CHUNK_SIZE 条，就写一个 part0001.faa / part0002.faa ...

os.makedirs(out_dir, exist_ok=True)

df_biogrid = pd.read_csv(file_path, compression='gzip', low_memory=False)
print(df_biogrid.columns.tolist())

cols = [
    'SWISS-PROT Accessions Interactor A',
    'TREMBL Accessions Interactor A',
    'SWISS-PROT Accessions Interactor B',
    'TREMBL Accessions Interactor B',
]

id_series = []
for c in cols:
    if c in df_biogrid.columns:
        s = (
            df_biogrid[c]
            .dropna()
            .astype(str)
            .str.split(r'[;|, ]+')
            .explode()
            .dropna()
            .str.strip()
        )
        s = s[(s != '') & (s.str.lower() != 'nan') & (s != '-')]
        id_series.append(s)

all_ids = pd.Index([])
for s in id_series:
    all_ids = all_ids.union(pd.Index(s.unique()))

all_ids = all_ids.sort_values()
if DEBUG_MODE:
    all_ids = all_ids[:DEBUG_N]

pd.DataFrame({'uniprot_id': all_ids}).to_csv(out_meta, index=False, encoding='utf-8-sig')
print('Unique IDs to download:', len(all_ids))

session = requests.Session()

def fetch_uniprot_fasta(uniprot_id):
    url = f'https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta'
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return None
    text = r.text.strip()
    return text if text.startswith('>') else None

def parse_fasta_text(fasta_text):
    rec = next(SeqIO.parse(StringIO(fasta_text), 'fasta'))
    return rec.id, rec.description, str(rec.seq).upper()

def write_faa(records, path):
    with open(path, 'w', encoding='utf-8') as f:
        for rec_id, desc, seq in records:
            f.write(f'>{rec_id} {desc}\n')
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')

def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'next_idx': 0, 'chunk_idx': 0}

def save_checkpoint(next_idx, chunk_idx):
    tmp = {
        'next_idx': next_idx,
        'chunk_idx': chunk_idx,
        'total_ids': len(all_ids)
    }
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(tmp, f, ensure_ascii=False, indent=2)

def append_csv(path, rows, header):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=header)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode='a', index=False, header=write_header, encoding='utf-8-sig')

ckpt = load_checkpoint()
start_idx = ckpt['next_idx']
chunk_idx = ckpt['chunk_idx']

failed_rows = []
long_rows = []
records = []

for i in tqdm(range(start_idx, len(all_ids)), desc='Downloading UniProt FASTA'):
    uid = all_ids[i]
    try:
        fasta_text = fetch_uniprot_fasta(uid)
        if fasta_text is None:
            failed_rows.append([uid])
        else:
            rec_id, desc, seq = parse_fasta_text(fasta_text)
            records.append((rec_id, desc, seq))
            if len(seq) > ESMC_MAX_LEN:
                long_rows.append([rec_id, len(seq)])
    except Exception:
        failed_rows.append([uid])

    if len(records) >= CHUNK_SIZE:
        chunk_idx += 1
        out_faa = os.path.join(out_dir, f'{prefix}.part{chunk_idx:04d}.faa')
        write_faa(records, out_faa)
        append_csv(out_failed, failed_rows, ['failed_uniprot_id'])
        append_csv(out_long, long_rows, ['uniprot_id', 'length'])
        records = []
        failed_rows = []
        long_rows = []
        save_checkpoint(i + 1, chunk_idx)

if records:
    chunk_idx += 1
    out_faa = os.path.join(out_dir, f'{prefix}.part{chunk_idx:04d}.faa')
    write_faa(records, out_faa)

append_csv(out_failed, failed_rows, ['failed_uniprot_id'])
append_csv(out_long, long_rows, ['uniprot_id', 'length'])
save_checkpoint(len(all_ids), chunk_idx)

print('Done.')
print('Chunks written:', chunk_idx)
print('Checkpoint saved to:', checkpoint_file)
