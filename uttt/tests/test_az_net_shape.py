import torch
from uttt.agents.az.net import AlphaZeroNetUTTT, AZNetConfig

cfg = AZNetConfig(in_planes=7, channels=96, blocks=6, policy_reduce=32, value_hidden=256)
net = AlphaZeroNetUTTT(cfg)
x = torch.randn(2, 7, 9, 9)

policy_logits, value = net(x)
assert policy_logits.shape == (2, 81)
assert value.shape == (2,)
print("OK:", policy_logits.shape, value.shape)
