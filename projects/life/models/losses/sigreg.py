# Adapted from LeJEPA by Randall Balestriero and Yann LeCun.
# Source: https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
# License: CC BY-NC 4.0 (see LICENSE).
# Modified for LIFE.

import torch
import torch.nn as nn

from mmdet3d.models.builder import LOSSES


class SIGRegBNMLPProjector(nn.Module):
    """BN-MLP projection head used before the SIGReg statistic."""

    def __init__(
        self,
        input_dim,
        hidden_dim=2048,
        output_dim=None,
        num_layers=2,
        bn_affine=True,
        bias=True,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError('SIGReg BN-MLP projector needs at least 2 layers')
        output_dim = int(output_dim or input_dim)
        hidden_dim = int(hidden_dim)
        dims = [int(input_dim)]
        dims.extend([hidden_dim] * (int(num_layers) - 1))
        dims.append(output_dim)

        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1], bias=bias))
            if idx < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[idx + 1], affine=bn_affine))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@LOSSES.register_module()
class SIGRegLoss(nn.Module):
    """Gaussian characteristic-function regularization of fused BEV tokens.

    Tokens are sampled before the BN-MLP projector. The sampled projections and
    warmup advance once per training forward; the counter is not checkpointed.
    """

    def __init__(
        self,
        input_dim=256,
        projector=None,
        loss_weight=0.01,
        num_projections=1024,
        projection_chunk=64,
        knots=17,
        max_tokens=8192,
        sample_seed=20260708,
        projection_seed=3407,
        warmup_iters=200,
    ):
        super().__init__()
        if num_projections < 1 or projection_chunk < 1 or max_tokens < 2:
            raise ValueError(
                'SIGReg requires positive projection counts and at least two tokens.'
            )
        if knots < 2 or warmup_iters < 0 or loss_weight < 0:
            raise ValueError(
                'SIGReg requires knots >= 2 and non-negative warmup/loss weight.'
            )
        self.loss_weight = float(loss_weight)
        self.num_projections = int(num_projections)
        self.projection_chunk = int(projection_chunk)
        self.max_tokens = int(max_tokens)
        self.sample_seed = int(sample_seed)
        self.projection_seed = int(projection_seed)
        self.warmup_iters = int(warmup_iters)
        self.projector = SIGRegBNMLPProjector(
            input_dim=input_dim, **(projector or {})
        )

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer('_t', t, persistent=False)
        self.register_buffer('_phi', window, persistent=False)
        self.register_buffer('_weights', weights * window, persistent=False)
        self.register_buffer(
            '_step', torch.zeros((), dtype=torch.long), persistent=False
        )

    def _sample_tokens(self, features):
        if features.ndim != 4:
            raise ValueError(
                'SIGReg expects fused BEV features with shape [B, C, H, W].'
            )
        tokens = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        if tokens.shape[0] < 2:
            raise ValueError('SIGReg requires at least two BEV tokens.')
        if tokens.shape[0] <= self.max_tokens:
            return tokens
        generator = torch.Generator(device=tokens.device)
        generator.manual_seed(self.sample_seed + int(self._step.item()))
        indices = torch.randint(
            tokens.shape[0],
            (self.max_tokens,),
            device=tokens.device,
            generator=generator,
        )
        return tokens.index_select(0, indices)

    def _statistic(self, x):
        x = x.float()
        n, dim = x.shape
        stats = []
        generator = torch.Generator(device=x.device)
        generator.manual_seed(self.projection_seed + int(self._step.item()))
        remaining = self.num_projections
        while remaining > 0:
            chunk = min(self.projection_chunk, remaining)
            projection = torch.randn(
                dim, chunk, device=x.device, generator=generator
            )
            projection = projection / projection.norm(
                p=2, dim=0, keepdim=True
            ).clamp_min(1e-12)
            projected = x @ projection
            xt = projected.unsqueeze(-1) * self._t.to(x.device)
            error = (
                xt.cos().mean(dim=0) - self._phi.to(x.device)
            ).square() + xt.sin().mean(dim=0).square()
            stats.append((error @ self._weights.to(x.device)) * n)
            remaining -= chunk
        return torch.cat(stats).mean()

    def forward(self, features):
        tokens = self._sample_tokens(features).contiguous()
        if (
            self.training
            and torch.is_grad_enabled()
            and not tokens.requires_grad
        ):
            raise RuntimeError(
                'SIGReg requires an upstream gradient from the BEV encoder.'
            )
        if tokens.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                embeddings = self.projector(tokens.float())
        else:
            embeddings = self.projector(tokens.float())
        raw = self._statistic(embeddings)
        if not torch.isfinite(raw):
            raise FloatingPointError(f'Non-finite SIGReg loss: {raw}')
        warmup = (
            min(1.0, (float(self._step.item()) + 1.0) / self.warmup_iters)
            if self.warmup_iters > 0
            else 1.0
        )
        weighted = raw * self.loss_weight * warmup
        self._step.add_(1)
        return weighted, raw.detach()
