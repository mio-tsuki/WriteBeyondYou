"""TorchTPU recurrent kernels implemented with Pallas JAX ops."""

from functools import lru_cache
from typing import Any

import torch


@lru_cache(maxsize=2)
def _direction_op(reverse: bool) -> Any:
    import jax
    import jax.numpy as jnp
    from torch.tpu.pallas import jax_op

    direction = "reverse" if reverse else "forward"
    op_name = f"wby::encoder_{direction}"

    def scan(
        x: jax.Array,
        lengths: jax.Array,
        weight_ih: jax.Array,
        weight_hh: jax.Array,
        bias_ih: jax.Array,
        bias_hh: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        steps = x.shape[0]
        batch = x.shape[1]
        positions = jnp.arange(steps)

        input_gates = jnp.matmul(x, weight_ih.T) + bias_ih
        hidden_size = weight_hh.shape[1]
        zero_state = jnp.zeros((batch, hidden_size), dtype=x.dtype)

        def step(state, values):
            h, c = state
            gates, position = values
            gates = gates + jnp.matmul(h, weight_hh.T) + bias_hh
            i, f, g, o = jnp.split(gates, 4, axis=-1)
            candidate_c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
            candidate_h = jax.nn.sigmoid(o) * jnp.tanh(candidate_c)
            active = position < lengths
            h = jnp.where(active[:, None], candidate_h, h)
            c = jnp.where(active[:, None], candidate_c, c)
            output = jnp.where(active[:, None], h, 0)
            return (h, c), output

        final_state, output = jax.lax.scan(
            step,
            (zero_state, zero_state),
            (input_gates, positions),
            reverse=reverse,
        )
        return output, final_state[0]

    def scan_backward(
        x: jax.Array,
        lengths: jax.Array,
        weight_ih: jax.Array,
        weight_hh: jax.Array,
        bias_ih: jax.Array,
        bias_hh: jax.Array,
        grad_output: jax.Array,
        grad_hidden: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        def differentiable(xx, w_ih, w_hh, b_ih, b_hh):
            return scan(xx, lengths, w_ih, w_hh, b_ih, b_hh)

        _, pullback = jax.vjp(
            differentiable, x, weight_ih, weight_hh, bias_ih, bias_hh
        )
        return pullback((grad_output, grad_hidden))

    forward_op = jax_op(op_name, scan)
    backward_op = jax_op(f"{op_name}_backward", scan_backward)

    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    def backward(ctx, grad_output, grad_hidden):
        x, lengths, weight_ih, weight_hh, bias_ih, bias_hh = ctx.saved_tensors
        grad_x, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh = backward_op(
            x,
            lengths,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            grad_output,
            grad_hidden,
        )
        return grad_x, None, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh

    torch.library.register_autograd(op_name, backward, setup_context=setup_context)
    return forward_op


def run_direction(
    x: torch.Tensor,
    lengths: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    *,
    reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _direction_op(reverse)(x, lengths, weight_ih, weight_hh, bias_ih, bias_hh)
