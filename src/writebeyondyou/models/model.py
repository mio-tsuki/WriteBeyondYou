import torch
from torch import nn

from ..data.targets import prepare_targets
from .decoder import AttentionDecoder
from .encoder import BiLSTMEncoder
from .gmm import GMMHead, handwriting_loss


class HandwritingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.content_encoder = BiLSTMEncoder(num_layers=1)
        self.style_encoder = BiLSTMEncoder(num_layers=4)
        self.decoder = AttentionDecoder()
        self.head = GMMHead()

    def forward(self, batch: dict[str, torch.Tensor]):
        # Normalize coordinates without modifying the original batch.
        content = batch["content"].clone()
        style = batch["style"].clone()

        content[..., :2] /= 200.0
        style[..., :2] /= 200.0

        # Content encoder.
        content_memory, content_last = self.content_encoder(
            content,
            batch["content_lengths"],
        )

        # Style encoder: [B, N, T, 3] -> [B*N, T, 3].
        B, N, T, _ = style.shape

        style_memory, _ = self.style_encoder(
            style.flatten(0, 1),
            batch["style_lengths"].flatten(),
        )

        style_memory = style_memory.reshape(B, N, T, -1)

        # Teacher-forcing inputs and targets.
        inputs, xy, pen, xy_mask, pen_mask = prepare_targets(
            batch["target"],
            batch["target_lengths"],
        )

        # Decoder initial state.
        state = self.decoder.core.initial_state(content_last)

        attention = content_memory.new_zeros(B, 1024)

        alignment = content_memory.new_zeros(
            B, content_memory.size(1)
        )
        alignment[:, 0] = 1.0

        # Autoregressive decoder with teacher forcing.
        outputs = []

        for t in range(inputs.size(1)):
            attention, state, alignment, _ = self.decoder(
                inputs[:, t],
                attention,
                state,
                content_memory,
                batch["content_lengths"],
                style_memory,
                batch["style_lengths"],
                alignment,
            )

            outputs.append(attention)

        # [B, T+1, 1024] -> GMM parameters.
        features = torch.stack(outputs, dim=1)
        params = self.head(features)

        return handwriting_loss(
            params, xy, pen, xy_mask, pen_mask
        )
