import torch
from torch import nn
from torch._higher_order_ops import scan


class StyleLuongAttention(nn.Module):
    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, steps, hidden = memory.shape

        values = memory.reshape(batch, count * steps, hidden)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def step(carry, inputs):
        p_i, previous_i, shifted_i = inputs

        next_carry = previous_i + (1 - shifted_i) * carry
        alignment = p_i * next_carry

        return next_carry, alignment

    _, alignments = scan(
        step,
        p.new_zeros(p.size(0)),
        (p.T, previous.T, shifted.T),
    )

    return alignments.T
