import torch
from torch import nn

from .attention import (
    ContentMonotonicAttention,
    StyleLuongAttention,
)


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size: int = 512):
        super().__init__()

        self.initial = nn.Linear(512, hidden_size * 2)

        self.lstm = nn.LSTMCell(
            input_size=5 + hidden_size * 2,
            hidden_size=hidden_size,
        )

        nn.init.normal_(self.initial.weight, std=0.001)
        nn.init.zeros_(self.initial.bias)

    def initial_state(
        self,
        content_last: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.tanh(self.initial(content_last))
        c, h = state.chunk(2, dim=-1)

        return h, c

    def forward(
        self,
        stroke: torch.Tensor,
        attention: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((stroke, attention), dim=-1)
        return self.lstm(x, state)

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size: int = 512):
        super().__init__()

        self.core = LSTMDecoder(hidden_size)

        self.content_attention = ContentMonotonicAttention(hidden_size)
        self.style_attention = StyleLuongAttention(hidden_size)

        self.content_layer = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.style_layer = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(
        self,
        stroke: torch.Tensor,
        previous_attention: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        content_memory: torch.Tensor,
        content_lengths: torch.Tensor,
        style_memory: torch.Tensor,
        style_lengths: torch.Tensor,
        previous_alignment: torch.Tensor,
    ):
        h, c = self.core(stroke, previous_attention, state)

        content_context, content_alignment = self.content_attention(
            h, content_memory, content_lengths, previous_alignment
        )

        style_context, style_weights = self.style_attention(
            h, style_memory, style_lengths
        )

        content_output = self.content_layer(
            torch.cat((h, content_context), dim=-1)
        )

        style_output = self.style_layer(
            torch.cat((h, style_context), dim=-1)
        )

        attention = torch.cat(
            (content_output, style_output),
            dim=-1,
        )

        return attention, (h, c), content_alignment, style_weights
