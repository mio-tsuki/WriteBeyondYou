import math

import torch
from torch import nn
from torch.nn import functional as F


class GMMHead(nn.Module):
    def __init__(
        self,
        input_size: int = 1024,
        num_mixtures: int = 20,
    ) -> None:
        super().__init__()

        self.projection = nn.Linear(
            input_size,
            3 + 6 * num_mixtures,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.projection(x)

        pen_logits = output[..., :3]

        pi, mu_x, mu_y, sigma_x, sigma_y, rho = (
            output[..., 3:].chunk(6, dim=-1)
        )

        return {
            "pen_logits": pen_logits,
            "pi_logits": pi,
            "mu_x": mu_x,
            "mu_y": mu_y,
            "log_sigma_x": sigma_x,
            "log_sigma_y": sigma_y,
            "rho_logits": rho,
        }

def gmm_nll(
    params: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    log_pi = params["pi_logits"].float().log_softmax(dim=-1)

    mu_x = params["mu_x"].float()
    mu_y = params["mu_y"].float()

    log_sigma_x = params["log_sigma_x"].float().clamp(-10, 6)
    log_sigma_y = params["log_sigma_y"].float().clamp(-10, 6)

    sigma_x = log_sigma_x.exp()
    sigma_y = log_sigma_y.exp()

    rho = params["rho_logits"].float().tanh()
    rho = rho.clamp(-0.999, 0.999)

    x = target[..., 0].float().unsqueeze(-1)
    y = target[..., 1].float().unsqueeze(-1)

    dx = (x - mu_x) / sigma_x
    dy = (y - mu_y) / sigma_y

    one_minus_rho2 = 1 - rho.square()

    z = dx.square() + dy.square() - 2 * rho * dx * dy

    log_pdf = (
        -math.log(2 * math.pi)
        - log_sigma_x
        - log_sigma_y
        - 0.5 * one_minus_rho2.log()
        - z / (2 * one_minus_rho2)
    )

    return -torch.logsumexp(log_pi + log_pdf, dim=-1)

def handwriting_loss(
    params: dict[str, torch.Tensor],
    xy: torch.Tensor,
    pen: torch.Tensor,
    xy_mask: torch.Tensor,
    pen_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xy_nll = gmm_nll(params, xy)

    pen_ce = F.cross_entropy(
        params["pen_logits"].float().transpose(1, 2),
        pen,
        reduction="none",
    )

    xy_loss = (
        (xy_nll * xy_mask).sum()
        / xy_mask.sum().clamp_min(1)
    )

    pen_loss = (
        (pen_ce * pen_mask).sum()
        / pen_mask.sum().clamp_min(1)
    )

    total = xy_loss + 2.0 * pen_loss

    return total, xy_loss, pen_loss
