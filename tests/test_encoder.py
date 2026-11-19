import pytest
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from writebeyondyou.models.encoder import BiLSTMEncoder


def _packed_reference(
    encoder: BiLSTMEncoder, x: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = pack_padded_sequence(
        x, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    packed_output, (hidden, _) = encoder.lstm(packed)
    output, _ = pad_packed_sequence(
        packed_output, batch_first=True, total_length=x.shape[1]
    )
    final = torch.cat((hidden[-2], hidden[-1]), dim=-1)
    return output, final


@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize(
    ("lengths", "steps"),
    [
        ([7, 4, 1, 6], 7),  # unsorted, full length, and one-step sequences
        ([3, 5, 2], 8),  # every sequence has trailing padding
    ],
)
def test_forward_matches_packed_lstm_for_valid_lengths(
    num_layers: int, lengths: list[int], steps: int
) -> None:
    torch.manual_seed(101)
    encoder = BiLSTMEncoder(input_size=3, hidden_size=5, num_layers=num_layers)
    x = torch.randn(len(lengths), steps, 3)
    lengths_tensor = torch.tensor(lengths)

    actual_memory, actual_last = encoder(x, lengths_tensor)
    expected_memory, expected_last = _packed_reference(encoder, x, lengths_tensor)

    torch.testing.assert_close(actual_memory, expected_memory, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_last, expected_last, rtol=1e-5, atol=1e-6)
    padding = torch.arange(steps)[None, :] >= lengths_tensor[:, None]
    assert torch.count_nonzero(actual_memory[padding]) == 0


def test_forward_and_parameter_gradients_match_packed_lstm() -> None:
    torch.manual_seed(202)
    encoder = BiLSTMEncoder(input_size=3, hidden_size=4, num_layers=2)
    x = torch.randn(3, 6, 3, requires_grad=True)
    lengths = torch.tensor([6, 2, 4])

    actual_memory, actual_last = encoder(x, lengths)
    actual_loss = actual_memory.square().sum() + actual_last.square().sum()
    actual_loss.backward()
    actual_input_grad = x.grad.detach().clone()
    actual_parameter_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in encoder.named_parameters()
    }

    encoder.zero_grad(set_to_none=True)
    reference_x = x.detach().clone().requires_grad_()
    expected_memory, expected_last = _packed_reference(encoder, reference_x, lengths)
    (expected_memory.square().sum() + expected_last.square().sum()).backward()

    torch.testing.assert_close(actual_input_grad, reference_x.grad, rtol=1e-5, atol=1e-6)
    for name, parameter in encoder.named_parameters():
        torch.testing.assert_close(
            actual_parameter_grads[name], parameter.grad, rtol=1e-5, atol=1e-6
        )


def test_state_dict_keeps_standard_bidirectional_lstm_keys() -> None:
    encoder = BiLSTMEncoder(input_size=3, hidden_size=4, num_layers=2)
    expected = {
        f"lstm.{parameter}{layer}{direction}"
        for layer in range(2)
        for direction in ("", "_reverse")
        for parameter in ("weight_ih_l", "weight_hh_l", "bias_ih_l", "bias_hh_l")
    }

    assert set(encoder.state_dict()) == expected
