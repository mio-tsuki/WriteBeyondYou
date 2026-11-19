import torch
from torch import nn


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # PackedSequence requires copying lengths to the host and is not
        # supported by accelerator graph compilers. Keep a dense, static
        # sequence and hold recurrent state outside each example's true length.
        batch, steps, _ = x.shape
        if x.device.type == "tpu":
            lengths = lengths.to(device=x.device, dtype=torch.int32)
        positions = torch.arange(steps, device=x.device)
        valid = positions[:, None] < lengths[None, :]

        layer_input = x.transpose(0, 1)
        final_states = []

        for layer in range(self.lstm.num_layers):
            directions = []
            layer_states = []
            for reverse in (False, True):
                suffix = "_reverse" if reverse else ""
                weight_ih = getattr(self.lstm, f"weight_ih_l{layer}{suffix}")
                weight_hh = getattr(self.lstm, f"weight_hh_l{layer}{suffix}")
                bias_ih = getattr(self.lstm, f"bias_ih_l{layer}{suffix}")
                bias_hh = getattr(self.lstm, f"bias_hh_l{layer}{suffix}")
                if x.device.type == "tpu":
                    from ._tpu_encoder import run_direction

                    output, last_h = run_direction(
                        layer_input,
                        lengths,
                        weight_ih,
                        weight_hh,
                        bias_ih,
                        bias_hh,
                        reverse=reverse,
                    )
                    directions.append(output)
                    layer_states.append(last_h)
                    continue

                initial = x.new_zeros(batch, self.lstm.hidden_size)
                input_gates = torch.nn.functional.linear(
                    layer_input, weight_ih, bias_ih
                )
                input_gate_steps = input_gates.unbind(dim=0)

                def cell(h, c, gates):
                    gates = gates + torch.nn.functional.linear(
                        h, weight_hh, bias_hh
                    )
                    i, f, g, o = gates.chunk(4, dim=-1)
                    next_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                    next_h = torch.sigmoid(o) * torch.tanh(next_c)
                    return next_h, next_c

                h = c = initial
                output_steps = [None] * steps
                indices = range(steps - 1, -1, -1) if reverse else range(steps)
                for t in indices:
                    h_candidate, c_candidate = cell(h, c, input_gate_steps[t])
                    mask = valid[t, :, None]
                    h = torch.where(mask, h_candidate, h)
                    c = torch.where(mask, c_candidate, c)
                    output_steps[t] = torch.where(mask, h, torch.zeros_like(h))
                output = torch.stack(output_steps, dim=0)
                last_h = h
                directions.append(output)
                layer_states.append(last_h)

            layer_input = torch.cat(directions, dim=-1)
            final_states = layer_states

        all_h = layer_input.transpose(0, 1)
        last_h = torch.cat(final_states, dim=-1)
        return all_h, last_h
