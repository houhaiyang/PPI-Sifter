import argparse
import torch
from torch.utils.data import DataLoader
from ppisifter.config import load_config
from ppisifter.io import load_embeddings_pt
from ppisifter.data import PairDataset, collate_fn
from ppisifter.model import PPISifter
from ppisifter.train_utils import run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--pairs', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.get('device', 'cpu') if torch.cuda.is_available() else 'cpu')
    embeddings = load_embeddings_pt(cfg['paths']['embeddings_pt'])
    pairs_csv = args.pairs or cfg['paths']['test_csv']
    ds = PairDataset(pairs_csv, embeddings)
    loader = DataLoader(ds, batch_size=cfg['infer']['batch_size'], shuffle=False, collate_fn=collate_fn)
    model = PPISifter(**cfg['model']).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model_state'])
    metrics = run_epoch(model, loader, None, cfg, device, train=False)
    print(metrics)

if __name__ == '__main__':
    main()
