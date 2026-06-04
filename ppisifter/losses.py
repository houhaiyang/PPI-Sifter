import torch
import torch.nn.functional as F


def focal_bce_with_logits(logits, targets, alpha=0.9, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    return (alpha_t * (1 - pt).pow(gamma) * bce).mean()


def sparse_loss(attn_map, valid_mask):
    vals = attn_map[valid_mask]
    if vals.numel() == 0:
        return attn_map.new_tensor(0.0)
    return vals.mean()


def symmetry_loss(attn_map):
    at = attn_map.transpose(1, 2)
    h = min(attn_map.shape[1], attn_map.shape[2])
    return F.l1_loss(attn_map[:, :h, :h], at[:, :h, :h])


def compute_loss(outputs, labels, cfg):
    pos_weight = torch.tensor([cfg['train']['pos_weight']], device=labels.device)
    wbce = F.binary_cross_entropy_with_logits(outputs['logit'], labels, pos_weight=pos_weight)
    focal = focal_bce_with_logits(outputs['logit'], labels, cfg['train']['focal_alpha'], cfg['train']['focal_gamma'])
    sparse = sparse_loss(outputs['attn_map'], outputs['valid_mask'])
    sym = symmetry_loss(outputs['attn_map'])
    total = wbce + cfg['train']['lambda_focal'] * focal + cfg['train']['lambda_sparse'] * sparse + cfg['train']['lambda_sym'] * sym
    return total, {
        'wbce': float(wbce.detach().cpu()),
        'focal': float(focal.detach().cpu()),
        'sparse': float(sparse.detach().cpu()),
        'sym': float(sym.detach().cpu()),
        'total': float(total.detach().cpu()),
    }
