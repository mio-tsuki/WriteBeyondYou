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

    @torch.inference_mode()
    def generate(
        self,
        batch: dict[str, torch.Tensor],
        max_steps: int = 768,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> torch.Tensor:
        if not 0 < temperature < float("inf"):
            raise ValueError("temperature must be positive and finite")

        if batch["content"].size(0) != 1:
            raise ValueError("Inference currently supports batch_size=1")

        device = next(self.parameters()).device

        content = batch["content"].to(device).clone()
        style = batch["style"].to(device).clone()
        content_lengths = batch["content_lengths"].to(device)
        style_lengths = batch["style_lengths"].to(device)

        content[..., :2] /= 200.0
        style[..., :2] /= 200.0

        content_memory, content_last = self.content_encoder(
            content, content_lengths
        )

        B, N, T, _ = style.shape
        style_memory, _ = self.style_encoder(
            style.flatten(0, 1),
            style_lengths.flatten(),
        )
        style_memory = style_memory.reshape(B, N, T, -1)

        state = self.decoder.core.initial_state(content_last)

        stroke = content.new_zeros(B, 5)
        stroke[:, 2] = 1.0

        attention = content_memory.new_zeros(B, 1024)
        alignment = content_memory.new_zeros(
            B, content_memory.size(1)
        )
        alignment[:, 0] = 1.0

        points = []

        for _ in range(max_steps):
            attention, state, alignment, _ = self.decoder(
                stroke,
                attention,
                state,
                content_memory,
                content_lengths,
                style_memory,
                style_lengths,
                alignment,
            )

            params = self.head(attention)

            # Select a Gaussian component and pen state.
            if greedy:
                mixture = params["pi_logits"].argmax(dim=-1)
                pen = params["pen_logits"].argmax(dim=-1)
            else:
                mixture = torch.distributions.Categorical(
                    logits=params["pi_logits"].float() / temperature
                ).sample()

                pen = torch.distributions.Categorical(
                    logits=params["pen_logits"].float() / temperature
                ).sample()

            if pen.item() == 2:
                break

            index = mixture.unsqueeze(-1)

            mu_x = params["mu_x"].float().gather(1, index)
            mu_y = params["mu_y"].float().gather(1, index)

            if greedy:
                dx, dy = mu_x, mu_y
            else:
                sigma_x = (
                    params["log_sigma_x"]
                    .float()
                    .clamp(-10, 6)
                    .gather(1, index)
                    .exp()
                )
                sigma_y = (
                    params["log_sigma_y"]
                    .float()
                    .clamp(-10, 6)
                    .gather(1, index)
                    .exp()
                )
                rho = (
                    params["rho_logits"]
                    .float()
                    .gather(1, index)
                    .tanh()
                    .clamp(-0.999, 0.999)
                )

                # Scale covariance by temperature.
                scale = temperature ** 0.5
                z1 = torch.randn_like(mu_x)
                z2 = torch.randn_like(mu_y)

                dx = mu_x + sigma_x * scale * z1
                dy = mu_y + sigma_y * scale * (
                    rho * z1
                    + (1 - rho.square()).sqrt() * z2
                )

            xy = torch.cat((dx, dy), dim=-1)

            pen_up = (pen == 1).to(xy.dtype).unsqueeze(-1)
            points.append(torch.cat((xy, pen_up), dim=-1))

            stroke = torch.cat(
                (
                    xy,
                    torch.nn.functional.one_hot(
                        pen, num_classes=3
                    ).to(xy.dtype),
                ),
                dim=-1,
            )

        if not points:
            return content.new_zeros(0, 3)

        result = torch.cat(points, dim=0)
        result[:, :2] *= 200.0

        return result
