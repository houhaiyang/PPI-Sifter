'''
输入一个目录下的所有 .faa；

或者你手动传入多个 .faa；

每个输入文件对应输出一个 clean fasta 和 metadata。

'''


from Bio import SeqIO
import re
import csv
import os
from glob import glob

input_dir = 'data/BioGRID/faa_chunks'
output_dir = 'data/BioGRID/clean_chunks'
pattern = os.path.join(input_dir, '*.faa')

os.makedirs(output_dir, exist_ok=True)

def parse_uniprot_header(raw_header: str):
    raw_header = str(raw_header).strip()

    m = re.match(r'^(?:sp|tr)\|([A-Z0-9]+(?:-\d+)?)\|([^\s]+)\s*(.*)$', raw_header)
    if m:
        accession = m.group(1)
        entry = m.group(2)
        rest = m.group(3).strip()
    else:
        accession = 'UNKNOWN'
        entry = ''
        rest = raw_header

    species = ''
    m_os = re.search(r'\bOS=(.*?)(?:\sOX=|\sGN=|\sPE=|\sSV=|$)', rest)
    if m_os:
        species = m_os.group(1).strip()

    original_uniprot_id = ''
    m_id = re.match(r'^(?:sp|tr)\|([A-Z0-9]+(?:-\d+)?)\|([^\s]+)', raw_header)
    if m_id:
        original_uniprot_id = f'{m_id.group(1)}|{m_id.group(2)}'

    if accession == 'UNKNOWN':
        m_fallback = re.search(r'\b([A-Z0-9]{6,10}(?:-\d+)?)\b', raw_header)
        if m_fallback:
            accession = m_fallback.group(1)

    return accession, species, original_uniprot_id, raw_header

def process_one_faa(input_faa, output_faa, metadata_csv):
    with open(output_faa, 'w', encoding='utf-8') as fout, open(metadata_csv, 'w', newline='', encoding='utf-8-sig') as fmeta:
        writer = csv.writer(fmeta)
        writer.writerow(['accession', 'original_header', 'species', 'original_uniprot_id'])

        seen = set()
        for record in SeqIO.parse(input_faa, 'fasta'):
            raw_header = record.description if record.description else record.id
            accession, species, original_uniprot_id, original_header = parse_uniprot_header(raw_header)

            if accession in seen:
                continue
            seen.add(accession)

            record.id = accession
            record.name = accession
            record.description = ''
            SeqIO.write(record, fout, 'fasta')

            writer.writerow([accession, original_header, species, original_uniprot_id])

    print(f'Saved cleaned FASTA to: {output_faa}')
    print(f'Saved metadata to: {metadata_csv}')

fasta_files = sorted(glob(pattern))
print('Found FASTA files:', len(fasta_files))

for input_faa in fasta_files:
    base = os.path.basename(input_faa).replace('.faa', '')
    output_faa = os.path.join(output_dir, f'{base}.clean.faa')
    metadata_csv = os.path.join(output_dir, f'{base}.metadata.csv')
    process_one_faa(input_faa, output_faa, metadata_csv)



# 合并 output_dir 下所有 clean fasta
merged_faa = os.path.join(output_dir, '../all.clean.faa')
clean_faa_files = sorted(glob(os.path.join(output_dir, '*.clean.faa')))

with open(merged_faa, 'w', encoding='utf-8') as fout:
    for faa_file in clean_faa_files:
        with open(faa_file, 'r', encoding='utf-8') as fin:
            content = fin.read().strip()
            if content:
                fout.write(content + '\n')

print(f'Merged clean FASTA saved to: {merged_faa}')


# 合并 output_dir 下所有 metadata csv
merged_metadata_csv = os.path.join(output_dir, '../all.metadata.csv')
metadata_files = sorted(glob(os.path.join(output_dir, '*.metadata.csv')))

header_written = False
with open(merged_metadata_csv, 'w', newline='', encoding='utf-8-sig') as fout:
    writer = csv.writer(fout)

    for meta_file in metadata_files:
        with open(meta_file, 'r', encoding='utf-8-sig', newline='') as fin:
            reader = csv.reader(fin)
            rows = list(reader)
            if not rows:
                continue

            if not header_written:
                writer.writerow(rows[0])
                header_written = True

            for row in rows[1:]:
                writer.writerow(row)

print(f'Merged metadata CSV saved to: {merged_metadata_csv}')

