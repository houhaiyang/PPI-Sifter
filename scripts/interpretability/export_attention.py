import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ppisifter.config import load_config
from ppisifter.io import load_embeddings_pt
from ppisifter.data import PairDataset, collate_fn
from ppisifter.model import PPISifter
from ppisifter.interpret import export_pair_interpretation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--topk', type=int, default=20)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.get('device', 'cpu') if torch.cuda.is_available() else 'cpu')
    embeddings = load_embeddings_pt(cfg['paths']['embeddings_pt'])
    ds = PairDataset(args.pairs, embeddings)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model = PPISifter(**cfg['model']).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state['model_state'])
    model.eval()
    outdir = Path(args.output_dir)
    with torch.no_grad():
        for batch in loader:
            out = model(batch.a.to(device), batch.b.to(device), batch.mask_a.to(device), batch.mask_b.to(device))
            meta = dict(batch.meta[0])
            meta['interaction_prob'] = float(out['prob'][0].cpu())
            export_pair_interpretation(meta, out['attn_map'][0].cpu(), out['valid_mask'][0].cpu(), outdir, topk=args.topk)

if __name__ == '__main__':
    main()
