"""Compiled JAX teacher-forcing decoder path for TorchTPU.

The pure JAX function is kept separate from the TorchTPU registration so it can
also serve as a small CPU reference when JAX is installed. No JAX import occurs
unless the TPU path is selected.
"""

from typing import Any


def teacher_forcing_scan_jax(
    inputs: Any,
    initial_attention: Any,
    initial_h: Any,
    initial_c: Any,
    content_memory: Any,
    content_keys: Any,
    content_lengths: Any,
    style_memory: Any,
    style_keys: Any,
    style_lengths: Any,
    initial_alignment: Any,
    lstm_weight_ih: Any,
    lstm_weight_hh: Any,
    lstm_bias_ih: Any,
    lstm_bias_hh: Any,
    content_score_bias: Any,
    content_layer_weight: Any,
    style_layer_weight: Any,
) -> Any:
    """Run the existing decoder recurrence with one compiled ``lax.scan``.

    ``inputs`` is time-major. Memories and parameters follow the existing
    PyTorch decoder layout. The returned tensor has shape ``[T, B, 2H]``.
    """
    import jax
    import jax.numpy as jnp

    content_positions = jnp.arange(content_memory.shape[1])
    style_count = style_lengths.shape[1]
    style_steps = style_memory.shape[1] // style_count
    style_positions = jnp.arange(style_steps)
    flat_style_mask = (style_positions[None, :] < style_lengths[:, :, None]).reshape(
        style_lengths.shape[0], style_count * style_steps
    )
    content_mask = content_positions[None, :] < content_lengths[:, None]

    def step(carry: tuple[Any, Any, Any, Any], stroke: Any):
        attention, h, c, previous_alignment = carry
        x = jnp.concatenate((stroke, attention), axis=-1)
        gates = x @ lstm_weight_ih.T + lstm_bias_ih
        gates = gates + h @ lstm_weight_hh.T + lstm_bias_hh
        input_gate, forget_gate, candidate, output_gate = jnp.split(gates, 4, axis=-1)
        input_gate = jax.nn.sigmoid(input_gate)
        forget_gate = jax.nn.sigmoid(forget_gate)
        candidate = jnp.tanh(candidate)
        output_gate = jax.nn.sigmoid(output_gate)
        c = forget_gate * c + input_gate * candidate
        h = output_gate * jnp.tanh(c)

        content_scores = jnp.einsum("bsh,bh->bs", content_keys, h)
        p = jax.nn.sigmoid(
            content_scores.astype(jnp.float32) + content_score_bias.astype(jnp.float32)
        )
        p = jnp.where(content_mask, p, 0.0)
        shifted = jnp.concatenate((jnp.zeros_like(p[:, :1]), p[:, :-1]), axis=1)

        def compose(left: tuple[Any, Any], right: tuple[Any, Any]):
            left_a, left_b = left
            right_a, right_b = right
            return right_a * left_a, right_b + right_a * left_b

        _, prefix_b = jax.lax.associative_scan(
            compose,
            (1.0 - shifted.T, previous_alignment.T),
        )
        alignment = (p.T * prefix_b).T
        content_context = jnp.einsum(
            "bs,bsh->bh", alignment.astype(content_memory.dtype), content_memory
        )

        style_scores = jnp.einsum("bsh,bh->bs", style_keys, h)
        style_scores = jnp.where(flat_style_mask, style_scores, -jnp.inf)
        style_weights = jax.nn.softmax(style_scores.astype(jnp.float32), axis=-1).astype(
            style_memory.dtype
        )
        style_context = jnp.einsum("bs,bsh->bh", style_weights, style_memory)

        content_output = jnp.concatenate((h, content_context), axis=-1) @ content_layer_weight.T
        style_output = jnp.concatenate((h, style_context), axis=-1) @ style_layer_weight.T
        attention = jnp.concatenate((content_output, style_output), axis=-1)
        return (attention, h, c, alignment), attention

    (_, _, _, _), outputs = jax.lax.scan(
        step,
        (initial_attention, initial_h, initial_c, initial_alignment),
        inputs,
    )
    return outputs


_torch_decoder_op: Any | None = None


def _get_torch_decoder_op() -> Any:
    """Register the custom forward/backward pair once on a TorchTPU runtime."""
    global _torch_decoder_op
    if _torch_decoder_op is not None:
        return _torch_decoder_op

    from jax import Array, vjp
    from torch.library import register_autograd
    from torch.tpu.pallas import jax_op

    def forward(
        inputs: Array, initial_attention: Array, initial_h: Array, initial_c: Array,
        content_memory: Array, content_keys: Array, content_lengths: Array,
        style_memory: Array, style_keys: Array, style_lengths: Array,
        initial_alignment: Array, lstm_weight_ih: Array, lstm_weight_hh: Array,
        lstm_bias_ih: Array, lstm_bias_hh: Array, content_score_bias: Array,
        content_layer_weight: Array, style_layer_weight: Array,
    ) -> Array:
        return teacher_forcing_scan_jax(
            inputs, initial_attention, initial_h, initial_c, content_memory,
            content_keys, content_lengths, style_memory, style_keys, style_lengths,
            initial_alignment, lstm_weight_ih, lstm_weight_hh, lstm_bias_ih,
            lstm_bias_hh, content_score_bias, content_layer_weight, style_layer_weight,
        )

    def backward(
        inputs: Array, initial_attention: Array, initial_h: Array, initial_c: Array,
        content_memory: Array, content_keys: Array, content_lengths: Array,
        style_memory: Array, style_keys: Array, style_lengths: Array,
        initial_alignment: Array, lstm_weight_ih: Array, lstm_weight_hh: Array,
        lstm_bias_ih: Array, lstm_bias_hh: Array, content_score_bias: Array,
        content_layer_weight: Array, style_layer_weight: Array, output_gradient: Array,
    ) -> tuple[
        Array, Array, Array, Array, Array, Array, Array, Array,
        Array, Array, Array, Array, Array, Array, Array, Array,
    ]:
        floating = (
            inputs, initial_attention, initial_h, initial_c, content_memory,
            content_keys, style_memory, style_keys, initial_alignment,
            lstm_weight_ih, lstm_weight_hh, lstm_bias_ih, lstm_bias_hh,
            content_score_bias, content_layer_weight, style_layer_weight,
        )

        def floating_compute(
            v_inputs: Array, v_initial_attention: Array, v_initial_h: Array,
            v_initial_c: Array, v_content_memory: Array, v_content_keys: Array,
            v_style_memory: Array, v_style_keys: Array, v_initial_alignment: Array,
            v_lstm_weight_ih: Array, v_lstm_weight_hh: Array, v_lstm_bias_ih: Array,
            v_lstm_bias_hh: Array, v_content_score_bias: Array,
            v_content_layer_weight: Array, v_style_layer_weight: Array,
        ) -> Array:
            return teacher_forcing_scan_jax(
                v_inputs, v_initial_attention, v_initial_h, v_initial_c,
                v_content_memory, v_content_keys, content_lengths, v_style_memory,
                v_style_keys, style_lengths, v_initial_alignment, v_lstm_weight_ih,
                v_lstm_weight_hh, v_lstm_bias_ih, v_lstm_bias_hh,
                v_content_score_bias, v_content_layer_weight, v_style_layer_weight,
            )

        _, pullback = vjp(floating_compute, *floating)
        return pullback(output_gradient)

    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        ctx.save_for_backward(*inputs)

    _forward_op = jax_op("writebeyondyou::teacher_forcing_decoder", forward)
    _backward_op = jax_op("writebeyondyou::teacher_forcing_decoder_backward", backward)

    def torch_backward(ctx: Any, output_gradient: Any) -> tuple[Any, ...]:
        gradients = iter(_backward_op(*ctx.saved_tensors, output_gradient))
        return tuple(None if index in (6, 9) else next(gradients) for index in range(18))

    register_autograd(
        "writebeyondyou::teacher_forcing_decoder",
        torch_backward,
        setup_context=setup_context,
    )
    _torch_decoder_op = _forward_op
    return _torch_decoder_op


def run_tpu_teacher_forcing(*inputs: Any) -> Any:
    """Invoke the registered TPU op, initializing registration lazily."""
    return _get_torch_decoder_op()(*inputs)
