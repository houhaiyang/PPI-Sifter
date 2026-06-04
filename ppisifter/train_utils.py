from pathlib import Path
import json
import torch
from tqdm import tqdm
from .losses import compute_loss
from .metrics import classification_metrics


def run_epoch(model, loader, optimizer, cfg, device, train=True):
    model.train(train)
    all_prob, all_true = [], []
    loss_records = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader):
            a = batch.a.to(device)
            b = batch.b.to(device)
            ma = batch.mask_a.to(device)
            mb = batch.mask_b.to(device)
            labels = batch.labels.to(device) if batch.labels is not None else None
            outputs = model(a, b, ma, mb)
            if labels is None:
                continue
            loss, loss_dict = compute_loss(outputs, labels, cfg)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
                optimizer.step()
            loss_records.append(loss_dict)
            all_prob.extend(outputs['prob'].detach().cpu().tolist())
            all_true.extend(labels.detach().cpu().tolist())
    metrics = classification_metrics(all_true, all_prob, threshold=0.5) if all_true else {}
    if loss_records:
        metrics['loss'] = sum(x['total'] for x in loss_records) / len(loss_records)
    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'metrics': metrics,
    }, path)


def save_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
