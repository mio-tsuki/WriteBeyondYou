import torch

from writebeyondyou.data.collate import collate_handwriting


def test_packed_style_indices_keep_each_example_valid_points_in_order() -> None:
    sequence_a = torch.arange(90, dtype=torch.float32).reshape(30, 3)
    sequence_b = torch.arange(300, 459, dtype=torch.float32).reshape(53, 3)
    sequence_c = torch.arange(600, 798, dtype=torch.float32).reshape(66, 3)
    empty = torch.empty(0, 3)
    styles = [[sequence_a, sequence_b, sequence_a], [empty, sequence_c, sequence_b]]
    samples = [
        {"content": sequence_a, "target": sequence_a, "style": style} for style in styles
    ]
    batch = collate_handwriting(samples, pack_style=True)
    indices = batch["style_attention_indices"]
    lengths = batch["style_attention_lengths"]
    assert indices.shape == (2, 128)
    assert lengths.tolist() == [[113], [119]]
    bank = torch.cat(
        [batch[f"style_bucket_{steps}"].flatten(0, 1) for steps in (64, 128)]
    )
    packed = bank.index_select(0, indices.flatten()).reshape(2, 128, 3)
    for row, length, reference in zip(packed, lengths.flatten(), styles, strict=True):
        torch.testing.assert_close(row[:length], torch.cat(reference))
