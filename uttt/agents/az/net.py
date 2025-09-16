# uttt/agents/az/net.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    This file implements the neural network architecture used in the AlphaZero agent for Ultimate Tic-Tac-Toe.
    The network consists of a residual tower followed by separate policy and value heads.
'''
# ---- Utilities ----------------------------------------------------------------

def _init_weights_kaiming(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply softmax over 'dim' after masking illegal positions to -inf.
    logits:     [B, A] (A = 81)
    legal_mask: [B, A] bool
    """
    neg_inf = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~legal_mask, neg_inf)
    return F.softmax(masked, dim=dim)


# ---- Residual Tower (no norms) ------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Simple two-layer residual block with no normalization.
    Conv3x3 -> ReLU -> Conv3x3 -> residual add -> ReLU
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        out = F.relu(out + x, inplace=True)
        return out


# ---- Config -------------------------------------------------------------------

@dataclass
class AZNetConfig:
    in_planes: int = 7        # your state planes (7×9×9)
    channels: int = 96        # trunk width (try 64–128)
    blocks: int = 6           # number of residual blocks
    board_n: int = 9          # spatial size (9x9)
    policy_reduce: int = 32   # internal feature width in policy head (NOT #actions)
    value_hidden: int = 256   # MLP hidden for value head


# ---- AlphaZero Network (no norms) ---------------------------------------------

class AlphaZeroNetUTTT(nn.Module):
    """
    AlphaZero-style dual-head net for Ultimate Tic-Tac-Toe (no norm layers).

    Input:  x ∈ R^{B×P×9×9}  (P = in_planes)
    Outputs:
      - policy_logits ∈ R^{B×81}  (one logit per cell; mask illegals before softmax)
      - value ∈ [-1, 1]^{B}       (expected outcome from current player's perspective)
    """
    def __init__(self, cfg: AZNetConfig = AZNetConfig()):
        super().__init__()
        self.cfg = cfg
        C = cfg.channels
        H = cfg.board_n

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_planes, C, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.tower = nn.Sequential(*[ResidualBlock(C) for _ in range(cfg.blocks)])

        # Policy head:
        # 1×1 conv (C -> policy_reduce) -> ReLU -> Flatten -> Linear(... -> 81)
        self.policy_conv = nn.Conv2d(C, cfg.policy_reduce, kernel_size=1, bias=True)
        self.policy_fc   = nn.Linear(cfg.policy_reduce * H * H, H * H, bias=True)

        # Value head:
        # 1×1 conv (C -> 32) -> ReLU -> Flatten -> Linear -> ReLU -> Linear -> Tanh
        self.value_conv  = nn.Conv2d(C, 32, kernel_size=1, bias=True)
        self.value_fc1   = nn.Linear(32 * H * H, cfg.value_hidden, bias=True)
        self.value_fc2   = nn.Linear(cfg.value_hidden, 1, bias=True)

        self.apply(_init_weights_kaiming)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, P, 9, 9]
        Returns:
            policy_logits: [B, 81]
            value:         [B]
        """
        B = x.size(0)
        H = self.cfg.board_n
        assert x.dim() == 4 and x.size(-1) == H and x.size(-2) == H, f"Expected [B,P,{H},{H}], got {tuple(x.shape)}"

        h = self.stem(x)
        h = self.tower(h)

        # Policy
        p = F.relu(self.policy_conv(h), inplace=True)
        p = p.view(B, -1)                    # [B, policy_reduce*H*H]
        policy_logits = self.policy_fc(p)    # [B, H*H] = [B, 81]

        # Value
        v = F.relu(self.value_conv(h), inplace=True)
        v = v.view(B, -1)                    # [B, 32*H*H]
        v = F.relu(self.value_fc1(v), inplace=True)
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)  # [B]

        return policy_logits, value

    @torch.no_grad()
    def predict_priors_value(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience inference:
          returns priors (softmax over legal) and value
        Args:
          x:          [B, P, 9, 9]
          legal_mask: [B, 81] bool
        """
        logits, value = self.forward(x)
        priors = F.softmax(logits, dim=-1) if legal_mask is None else masked_softmax(logits, legal_mask, dim=-1)
        return priors, value


# ---- Standalone inference function for MCTS compatibility --------------------

def infer(net: AlphaZeroNetUTTT, obs_batch: torch.Tensor, legal_mask_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inference function for MCTS compatibility.
    
    Args:
        net: The neural network
        obs_batch: [B, 7, 9, 9] observations
        legal_mask_batch: [B, 81] legal action mask
    
    Returns:
        priors: [B, 81] action probabilities
        values: [B] value estimates
    """
    return net.predict_priors_value(obs_batch, legal_mask_batch)
