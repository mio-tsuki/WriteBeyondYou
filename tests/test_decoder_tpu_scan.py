"""Numerical contract tests for the pure JAX decoder scan."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="JAX is optional",
)


def test_teacher_forcing_scan_matches_torch_outputs_and_gradients() -> None:
    import jax
    import jax.numpy as jnp

    from writebeyondyou.models._tpu_decoder import teacher_forcing_scan_jax
    from writebeyondyou.models.decoder import AttentionDecoder

    torch.manual_seed(17)
    hidden = 2
    batch = 2
    steps = 3
    content_steps = 4
    style_count = 2
    style_steps = 3

    def tensor(shape: tuple[int, ...], *, requires_grad: bool = True) -> torch.Tensor:
        value = torch.randn(shape, dtype=torch.float32) * 0.15
        return value.requires_grad_(requires_grad)

    inputs = tensor((steps, batch, 5))
    initial_attention = tensor((batch, hidden * 2))
    initial_h = tensor((batch, hidden))
    initial_c = tensor((batch, hidden))
    content_memory = tensor((batch, content_steps, hidden))
    content_keys = tensor((batch, content_steps, hidden))
    style_memory = tensor((batch, style_count, style_steps, hidden))
    style_keys = tensor((batch, style_count * style_steps, hidden))
    initial_alignment = torch.zeros((batch, content_steps), dtype=torch.float32)
    initial_alignment[:, 0] = 1.0
    initial_alignment.requires_grad_()
    content_lengths = torch.tensor([4, 3], dtype=torch.int64)
    style_lengths = torch.tensor([[3, 3], [2, 2]], dtype=torch.int64)

    decoder = AttentionDecoder(hidden_size=hidden)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.12)

    # The helper takes parameter tensors explicitly; preserve the exact module
    # parameters as gradient targets in the PyTorch recurrence.
    module_parameters = (
        decoder.core.lstm.weight_ih,
        decoder.core.lstm.weight_hh,
        decoder.core.lstm.bias_ih,
        decoder.core.lstm.bias_hh,
        decoder.content_attention.score_bias,
        decoder.content_layer.weight,
        decoder.style_layer.weight,
    )

    torch_outputs = []
    state = (initial_h, initial_c)
    attention = initial_attention
    alignment = initial_alignment
    for stroke in inputs.unbind(0):
        attention, state, alignment, _ = decoder(
            stroke,
            attention,
            state,
            content_memory,
            content_lengths,
            style_memory,
            style_lengths,
            alignment,
            content_keys=content_keys,
            style_keys=style_keys,
        )
        torch_outputs.append(attention)
    expected = torch.stack(torch_outputs)

    jax_arguments = (
        inputs,
        initial_attention,
        initial_h,
        initial_c,
        content_memory,
        content_keys,
        content_lengths,
        style_memory.reshape(batch, style_count * style_steps, hidden),
        style_keys,
        style_lengths,
        initial_alignment,
        *module_parameters,
    )

    def scan_loss(*arguments):
        outputs = teacher_forcing_scan_jax(*arguments)
        cotangent = jnp.linspace(-0.7, 0.9, outputs.size, dtype=outputs.dtype).reshape(
            outputs.shape
        )
        return jnp.sum(outputs * cotangent)

    jax_values = tuple(jnp.asarray(value.detach().numpy()) for value in jax_arguments)
    actual = teacher_forcing_scan_jax(*jax_values)
    cotangent = torch.linspace(-0.7, 0.9, expected.numel()).reshape(expected.shape)
    torch_loss = torch.sum(expected * cotangent)
    torch_loss.backward()
    differentiable_indices = (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17)
    selected_gradients = jax.grad(scan_loss, argnums=differentiable_indices)(*jax_values)
    jax_gradients = dict(zip(differentiable_indices, selected_gradients, strict=True))

    torch.testing.assert_close(
        torch.from_numpy(np.array(actual)), expected.detach(), rtol=2e-5, atol=2e-6
    )

    differentiable_arguments = (
        inputs,
        initial_attention,
        initial_h,
        initial_c,
        content_memory,
        content_keys,
        style_memory,
        style_keys,
        initial_alignment,
        *module_parameters,
    )
    for torch_value, argument_index in zip(
        differentiable_arguments, differentiable_indices, strict=True
    ):
        jax_gradient = torch.from_numpy(np.array(jax_gradients[argument_index]))
        if argument_index == 7:
            jax_gradient = jax_gradient.reshape_as(style_memory)
        torch.testing.assert_close(
            jax_gradient,
            torch_value.grad,
            rtol=5e-4,
            atol=2e-5,
        )
