import torch
from ppisifter.model import PPISifter


def test_shapes():
    model = PPISifter(input_dim=16, proj_dim=8, num_heads=2, ffn_hidden=32, cls_hidden=(8, 4))
    a = torch.randn(2, 5, 16)
    b = torch.randn(2, 7, 16)
    ma = torch.tensor([[1,1,1,1,0],[1,1,1,0,0]], dtype=torch.bool)
    mb = torch.tensor([[1,1,1,1,1,0,0],[1,1,1,1,0,0,0]], dtype=torch.bool)
    out = model(a, b, ma, mb)
    assert out['prob'].shape == (2,)
    assert out['attn_map'].shape == (2, 5, 7)
