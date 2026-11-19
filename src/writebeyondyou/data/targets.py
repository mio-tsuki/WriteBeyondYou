import torch


def prepare_targets(
    target: torch.Tensor,
    lengths: torch.Tensor,
    scale: float = 200.0,
):
    batch, steps, _ = target.shape
    device = target.device

    positions = torch.arange(steps + 1, device=device)
    xy_mask = positions[None, :] < lengths[:, None]
    pen_mask = positions[None, :] <= lengths[:, None]

    # Format-3 → Format-5, with an initial S0.
    inputs = target.new_zeros(batch, steps + 1, 5)
    inputs[:, 0, 2] = 1.0

    inputs[:, 1:, :2] = target[:, :, :2] / scale
    inputs[:, 1:, 2] = (
        (1 - target[:, :, 2]) * xy_mask[:, :steps]
    )
    inputs[:, 1:, 3] = (
        target[:, :, 2] * xy_mask[:, :steps]
    )

    # Coordinate targets, including an unused EOS position.
    xy = target.new_zeros(batch, steps + 1, 2)
    xy[:, :steps] = target[:, :, :2] / scale

    # 0 = continue, 1 = pen up, 2 = EOS
    pen = torch.full(
        (batch, steps + 1),
        2,
        dtype=torch.long,
        device=device,
    )
    pen[:, :steps] = torch.where(
        xy_mask[:, :steps],
        target[:, :, 2].long(),
        2,
    )

    return inputs, xy, pen, xy_mask, pen_mask
