import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )


def forward(
    self,
    x: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = pack_padded_sequence(
        x,
        lengths.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    packed_output, (h, _) = self.lstm(packed)

    all_h, _ = pad_packed_sequence(
        packed_output,
        batch_first=True,
        total_length=x.size(1),
    )

    last_h = torch.cat((h[-2], h[-1]), dim=-1)

    return all_h, last_h
