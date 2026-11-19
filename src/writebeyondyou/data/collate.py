import torch

from .bucket import BUCKETS, collate_strokes


def collate_handwriting(samples: list[dict], *, pack_style: bool = False) -> dict:
    content, content_lengths = collate_strokes(
        [sample["content"] for sample in samples]
    )

    target, target_lengths = collate_strokes(
        [sample["target"] for sample in samples]
    )

    styles = [
        style
        for sample in samples
        for style in sample["style"]
    ]

    style, style_lengths = collate_strokes(styles)

    batch_size = len(samples)
    style_count = len(samples[0]["style"])

    style = style.reshape(batch_size, style_count, -1, 3)
    style_lengths = style_lengths.reshape(batch_size, style_count)

    batch = {
        "content": content,
        "content_lengths": content_lengths,
        "style": style,
        "style_lengths": style_lengths,
        "target": target,
        "target_lengths": target_lengths,
    }
    if pack_style:
        style_steps = style.size(2)
        flat_style = style.flatten(0, 1)
        flat_lengths = style_lengths.flatten()
        starts = torch.empty_like(flat_lengths)
        offset = 0
        previous_steps = -1
        for steps in BUCKETS:
            rows = torch.nonzero((flat_lengths > previous_steps) & (flat_lengths <= steps)).flatten()
            previous_steps = steps
            if rows.numel() == 0:
                continue
            # Round the number of rows for more reusable TPU kernel shapes.
            capacity = ((rows.numel() + 63) // 64) * 64
            values = flat_style.new_zeros(capacity, steps, 3)
            group_lengths = flat_lengths.new_zeros(capacity)
            values[:rows.numel()] = flat_style[rows, :steps]
            group_lengths[:rows.numel()] = flat_lengths[rows]
            starts[rows] = offset + torch.arange(rows.numel()) * steps
            batch[f"style_bucket_{steps}"] = values
            batch[f"style_bucket_{steps}_lengths"] = group_lengths
            offset += capacity * steps

        positions = torch.arange(style_count * style_steps)
        valid = positions.remainder(style_steps) < style_lengths.repeat_interleave(
            style_steps, dim=1
        )
        lengths = style_lengths.sum(dim=1)
        # Round the packed width to limit recompiles between successive batches.
        width = max(128, ((int(lengths.max()) + 127) // 128) * 128)
        ordered = (positions + (~valid) * positions.numel()).sort(dim=1).values
        ordered = ordered[:, :width]
        tokens = ordered.remainder(positions.numel())
        starts = starts.reshape(batch_size, style_count)
        packed = starts.gather(1, tokens.div(style_steps, rounding_mode="floor"))
        packed = packed + tokens.remainder(style_steps)
        batch["style_attention_indices"] = torch.where(ordered < positions.numel(), packed, 0)
        batch["style_attention_lengths"] = lengths.unsqueeze(1)
    return batch
