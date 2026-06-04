import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ppisifter.config import load_config
from ppisifter.io import load_embeddings_pt
from ppisifter.data import PairDataset, collate_fn
from ppisifter.model import PPISifter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.get('device', 'cpu') if torch.cuda.is_available() else 'cpu')
    embeddings = load_embeddings_pt(cfg['paths']['embeddings_pt'])
    ds = PairDataset(args.pairs, embeddings)
    loader = DataLoader(ds, batch_size=cfg['infer']['batch_size'], shuffle=False, collate_fn=collate_fn)
    model = PPISifter(**cfg['model']).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state['model_state'])
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch.a.to(device), batch.b.to(device), batch.mask_a.to(device), batch.mask_b.to(device))
            for meta, prob in zip(batch.meta, out['prob'].cpu().tolist()):
                rows.append({**meta, 'interaction_prob': prob})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
