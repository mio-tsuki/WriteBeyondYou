from .bucket import collate_strokes


def collate_handwriting(samples: list[dict]) -> dict:
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

    return {
        "content": content,
        "content_lengths": content_lengths,
        "style": style,
        "style_lengths": style_lengths,
        "target": target,
        "target_lengths": target_lengths,
    }
