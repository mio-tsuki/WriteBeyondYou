import torch
from torch import nn


class StyleLuongAttention(nn.Module):
    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        lengths: torch.Tensor,
        keys: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, steps, hidden = memory.shape

        values = memory.reshape(batch, count * steps, hidden)
        if keys is None:
            keys = self.key(values)

        scores = torch.bmm(
            keys, query.unsqueeze(-1)
        ).squeeze(-1)

        positions = torch.arange(steps, device=memory.device)
        mask = (positions < lengths.unsqueeze(-1))
        mask = mask.reshape(batch, count * steps)

        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1)
        weights = weights.to(values.dtype)

        context = torch.bmm(
            weights.unsqueeze(1), values
        ).squeeze(1)

        return context, weights.reshape(batch, count, steps)


class ContentMonotonicAttention(nn.Module):
    def __init__(self, hidden_size: int = 512):
        super().__init__()

        self.key = nn.Linear(
            hidden_size, hidden_size, bias=False
        )
        self.score_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        lengths: torch.Tensor,
        previous: torch.Tensor,
        keys: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keys is None:
            keys = self.key(memory)

        scores = torch.bmm(
            keys, query.unsqueeze(-1)
        ).squeeze(-1)

        positions = torch.arange(
            memory.size(1), device=memory.device
        )
        mask = positions.unsqueeze(0) < lengths.unsqueeze(1)

        p = torch.sigmoid(
            scores.float() + self.score_bias.float()
        )
        p = p.masked_fill(~mask, 0.0)

        weights = monotonic_alignment(
            p, previous.float()
        )

        context = torch.bmm(
            weights.to(memory.dtype).unsqueeze(1),
            memory,
        ).squeeze(1)

        return context, weights

def monotonic_alignment(
    p: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    shifted = torch.cat(
        [torch.zeros_like(p[:, :1]), p[:, :-1]],
        dim=1,
    )

    # Each recurrence step is an affine map f_i(c) = b_i + a_i * c,
    # where b_i is the previous alignment and a_i is 1 - shifted_i.
    # Affine-map composition is associative, so a doubling prefix computes
    # every carry in O(log T) tensor rounds without divisions (including when
    # a_i is exactly zero).
    prefix_a = 1 - shifted
    prefix_b = previous
    offset = 1
    steps = p.size(1)

    while offset < steps:
        prior_a = prefix_a
        prior_b = prefix_b
        prefix_a = torch.cat(
            (
                prior_a[:, :offset],
                prior_a[:, offset:] * prior_a[:, :-offset],
            ),
            dim=1,
        )
        prefix_b = torch.cat(
            (
                prior_b[:, :offset],
                prior_b[:, offset:]
                + prior_a[:, offset:] * prior_b[:, :-offset],
            ),
            dim=1,
        )
        offset *= 2

    return p * prefix_b
