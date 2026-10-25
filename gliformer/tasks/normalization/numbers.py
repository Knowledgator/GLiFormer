"""Number heads of the normalization head.

Each head reads the span vectors and has three methods:

* ``forward(vectors)``: outputs for every span (FP32);
* ``loss(outputs, values, mask)``: loss on spans with a number target
  (``values`` is float64), or ``None`` when no span has one;
* ``decode(outputs)``: float64 values and a confidence per span.

The kinds (``normalization_config.number.heads``):

* ``regression``: one regressed value, ``asinh(y / s)`` (any sign) or a
  sign/zero class plus ``log10(|y| / s)``;
* ``bins``: one class over fixed bins of ``asinh(y / s)`` or ``log10``
  (with a zero class and a sign), decoded as the distribution's median;
* ``sci_digits``: ``y = sign x d1.d2d3d4d5 x c x 10^e``: a sign/zero class,
  an exponent class, the significant digits as 10-way classes (each
  conditioned on the exponent and the digits before it), and a regressed
  coefficient ``c`` that scales the digits.

Heads keep their parameters and arithmetic in FP32 whatever the rest of the
model runs in: rounding a prediction to bf16 afterwards cannot recover the
digits it lost.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_EVEN, Decimal

import torch
import torch.nn.functional as F
from torch import nn

from ...layers.mlp import create_mlp
from .bins import ZERO_EPSILON, BinSupport

SIGN_NEGATIVE = 0
SIGN_ZERO = 1
SIGN_POSITIVE = 2


def sign_states(values: torch.Tensor) -> torch.Tensor:
    """Negative / zero / positive class of float64 values."""

    states = torch.full(values.shape, SIGN_POSITIVE, dtype=torch.long, device=values.device)
    states = torch.where(values < 0, torch.full_like(states, SIGN_NEGATIVE), states)
    return torch.where(values == 0, torch.full_like(states, SIGN_ZERO), states)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    if not mask.any():
        return None
    return values[mask].mean()


class _FP32Module(nn.Module):
    """Keeps parameters in FP32 through ``.half()``/``.to(bfloat16)``."""

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        for parameter in self.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
        return result

    @staticmethod
    def _fp32(fn, *args):
        device_type = args[0].device.type if args[0].device.type in {"cuda", "cpu"} else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            return fn(*(
                arg.float() if isinstance(arg, torch.Tensor) and torch.is_floating_point(arg) else arg
                for arg in args
            ))


def _trunk(hidden_size: int, dropout: float) -> nn.Module:
    return create_mlp(
        input_dim=hidden_size,
        intermediate_dims=[hidden_size],
        output_dim=hidden_size,
        dropout=dropout,
        activation="gelu",
    )


# -- regression ---------------------------------------------------------------


class RegressionNumberHead(_FP32Module):
    """One regressed value in asinh or log10 space."""

    kind = "regression"

    def __init__(self, hidden_size: int, dropout: float, cfg):
        super().__init__()
        self.transform = cfg.transform
        self.scale = float(cfg.scale)
        self.loss_kind = cfg.loss
        self.huber_delta = float(cfg.huber_delta)
        self.trunk = _trunk(hidden_size, dropout)
        self.value = nn.Linear(hidden_size, 1)
        self.state = nn.Linear(hidden_size, 3) if self.transform == "log10" else None

    def forward(self, vectors, **_):
        return self._fp32(self._forward, vectors)

    def _forward(self, vectors):
        features = self.trunk(vectors)
        outputs = {"z": self.value(features).squeeze(-1)}
        if self.state is not None:
            outputs["state_logits"] = self.state(features)
        return outputs

    def _z_targets(self, values):
        if self.transform == "asinh":
            return torch.asinh(values / self.scale), torch.ones_like(values, dtype=torch.bool)
        magnitude = values.abs() / self.scale
        nonzero = magnitude >= ZERO_EPSILON
        return torch.log10(magnitude.clamp(min=ZERO_EPSILON)), nonzero

    def loss(self, outputs, values, mask):
        target, has_z = self._z_targets(values)
        use = mask & has_z
        terms = []
        if use.any():
            prediction = outputs["z"].double()
            if self.loss_kind == "huber":
                errors = F.huber_loss(prediction, target, delta=self.huber_delta, reduction="none")
            else:
                errors = (prediction - target).abs()
            terms.append(_masked_mean(errors.float(), use))
        if self.state is not None and mask.any():
            states = sign_states(values)
            terms.append(F.cross_entropy(outputs["state_logits"][mask], states[mask]))
        terms = [term for term in terms if term is not None]
        return (torch.stack(terms).sum() if terms else None), {}

    def decode(self, outputs):
        z = outputs["z"].double()
        if self.transform == "asinh":
            return self.scale * torch.sinh(z.clamp(-700, 700)), torch.ones_like(z)
        probs = torch.softmax(outputs["state_logits"].float(), dim=-1)
        confidence, state = probs.max(dim=-1)
        magnitude = self.scale * torch.pow(10.0, z.clamp(max=300))
        values = torch.where(state == SIGN_NEGATIVE, -magnitude, magnitude)
        values = torch.where(state == SIGN_ZERO, torch.zeros_like(values), values)
        return values, confidence.double()


# -- bins ---------------------------------------------------------------------


class BinsNumberHead(_FP32Module):
    """Bin logits (and a sign logit for log10)."""

    kind = "bins"

    def __init__(self, hidden_size: int, dropout: float, cfg):
        super().__init__()
        self.support = BinSupport.from_config(cfg)
        self.num_classes = self.support.num_classes
        self.trunk = _trunk(hidden_size, dropout)
        self.proj = nn.Linear(hidden_size, self.num_classes)
        self.sign = nn.Linear(hidden_size, 1) if self.support.has_zero else None
        self.register_buffer(
            "edges", torch.tensor(self.support.edges, dtype=torch.float64), persistent=True,
        )

    # The edges stay float64 whatever dtype the model is cast to.
    def _apply(self, fn, recurse=True):
        edges = self.edges.detach().clone()
        result = super()._apply(fn, recurse)
        self.edges = edges.to(device=self.edges.device, dtype=torch.float64)
        return result

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "edges"
        if key in state_dict:
            loaded = state_dict[key].detach().cpu()
            expected = torch.tensor(self.support.edges, dtype=torch.float64)
            same = loaded.shape == expected.shape and (
                torch.allclose(loaded.to(torch.float64), expected, rtol=1e-6, atol=1e-6)
                or torch.equal(expected.to(loaded.dtype), loaded)
            )
            if not same:
                raise ValueError(
                    "Checkpoint bin edges differ from normalization_config.number.bins "
                    f"({tuple(loaded.shape)} vs {tuple(expected.shape)} edges)"
                )
            state_dict[key] = expected
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, vectors, **_):
        return self._fp32(self._forward, vectors)

    def _forward(self, vectors):
        features = self.trunk(vectors)
        outputs = {"logits": self.proj(features)}
        if self.sign is not None:
            outputs["sign_logit"] = self.sign(features).squeeze(-1)
        return outputs

    def target_classes(self, values):
        """Class of each value and whether it has one (finite support)."""

        support = self.support
        edges = self.edges.to(values.device)
        offset = int(support.has_zero)
        if support.transform == "asinh":
            z = torch.asinh(values / support.scale)
            zero = torch.zeros_like(values, dtype=torch.bool)
        else:
            magnitude = values.abs() / support.scale
            zero = magnitude < ZERO_EPSILON
            z = torch.log10(magnitude.clamp(min=ZERO_EPSILON))
        position = torch.bucketize(z.contiguous(), edges, right=True)
        if support.has_tails:
            target = position + offset
            valid = torch.ones_like(zero)
        else:
            last = edges.numel()
            closed_end = (position == last) & (z == edges[-1])
            valid = ((position > 0) & (position < last)) | closed_end
            target = torch.where(closed_end, position - 1, position) - 1 + offset
        if support.has_zero:
            target = torch.where(zero, torch.zeros_like(target), target)
            valid = valid | zero
        return target.long(), valid

    def loss(self, outputs, values, mask):
        target, valid = self.target_classes(values)
        use = mask & valid
        stats = {"out_of_range": (mask & ~valid).sum().detach()}
        if not use.any():
            return None, stats
        loss = F.cross_entropy(
            outputs["logits"][use], target[use].clamp(0, self.num_classes - 1),
        )
        if self.sign is not None:
            nonzero = use & target.ne(0)
            if nonzero.any():
                loss = loss + F.binary_cross_entropy_with_logits(
                    outputs["sign_logit"][nonzero], values[nonzero].lt(0).float(),
                )
        return loss, stats

    def decode(self, outputs):
        probs = torch.softmax(outputs["logits"].float(), dim=-1).cpu()
        negative = (
            torch.sigmoid(outputs["sign_logit"].float()).cpu()
            if "sign_logit" in outputs else torch.zeros(probs.shape[:-1])
        )
        flat_probs = probs.reshape(-1, probs.shape[-1]).tolist()
        flat_negative = negative.reshape(-1).tolist()
        medians = [
            self.support.quantile(row, 0.5, negative_prob=neg)
            for row, neg in zip(flat_probs, flat_negative)
        ]
        values = torch.tensor(medians, dtype=torch.float64).reshape(probs.shape[:-1])
        confidence = probs.max(dim=-1).values.double()
        device = outputs["logits"].device
        return values.to(device), confidence.to(device)


# -- sign x digits x coefficient x 10^exponent ---------------------------------


def scientific_targets(values: torch.Tensor, mask: torch.Tensor, exp_range, num_digits: int):
    """Exponent class and significant digits of each value (exact decimals).

    Values are rounded half-even to ``num_digits`` significant digits first,
    carrying into the exponent (9.99996 -> 1.0000 x 10^(e+1)). Returns
    ``exp_class`` (``-100`` for zeros, values outside the range, no target)
    and ``digits`` ``(..., num_digits)`` (``-100`` where ``exp_class`` is).
    """

    low, high = int(exp_range[0]), int(exp_range[1])
    exp_class = torch.full(values.shape, -100, dtype=torch.long)
    digits = torch.full((*values.shape, num_digits), -100, dtype=torch.long)
    quantum = Decimal(1).scaleb(-(num_digits - 1))
    flat_values = values.detach().cpu().reshape(-1).tolist()
    flat_mask = mask.detach().cpu().reshape(-1).tolist()
    flat_exp = exp_class.view(-1)
    flat_digits = digits.view(-1, num_digits)
    for index, (value, keep) in enumerate(zip(flat_values, flat_mask)):
        if not keep or value == 0 or not math.isfinite(value):
            continue
        magnitude = Decimal(repr(abs(float(value))))
        exponent = magnitude.adjusted()
        scaled = magnitude.scaleb(-exponent).quantize(quantum, rounding=ROUND_HALF_EVEN)
        if scaled >= 10:
            exponent += 1
            scaled = magnitude.scaleb(-exponent).quantize(quantum, rounding=ROUND_HALF_EVEN)
        if not low <= exponent <= high:
            continue
        written = [int(char) for char in format(scaled, "f").replace(".", "")][:num_digits]
        written += [0] * (num_digits - len(written))
        flat_exp[index] = exponent - low
        flat_digits[index] = torch.tensor(written)
    return exp_class.to(values.device), digits.to(values.device)


class SciDigitsNumberHead(_FP32Module):
    """Sign, exponent, digit classes and a coefficient on the digits."""

    kind = "sci_digits"

    def __init__(self, hidden_size: int, dropout: float, cfg):
        super().__init__()
        self.exp_low, self.exp_high = (int(value) for value in cfg.exp_range)
        self.num_exponents = self.exp_high - self.exp_low + 1
        self.num_digits = int(cfg.digits)
        self.use_coefficient = bool(cfg.coefficient)
        self.apply_coefficient = bool(cfg.apply_coefficient)
        self.coefficient_range = float(cfg.coefficient_log10_range)
        self.coefficient_loss_coef = float(cfg.coefficient_loss_coef)
        width = int(cfg.hidden)
        self.trunk = _trunk(hidden_size, dropout)
        self.state = nn.Linear(hidden_size, 3)
        self.exponent = nn.Linear(hidden_size, self.num_exponents)
        self.digit_in = nn.Linear(hidden_size, width)
        self.exponent_embedding = nn.Embedding(self.num_exponents, width)
        self.digit_position = nn.Embedding(self.num_digits, width)
        self.digit_previous = nn.Embedding(self.num_digits * 10, width)
        self.digit_out = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 10),
        )
        if self.use_coefficient:
            self.coefficient_out = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, 1),
            )
            # c = 1 at the start.
            with torch.no_grad():
                self.coefficient_out[-1].weight.mul_(0.01)
                self.coefficient_out[-1].bias.zero_()

    def forward(self, vectors, digit_vectors=None):
        """``digit_vectors`` ``(..., digits, D)``: one vector per digit (token queries)."""

        return self._fp32(self._forward, vectors, digit_vectors)

    def _forward(self, vectors, digit_vectors=None):
        features = self.trunk(vectors)
        outputs = {
            "state_logits": self.state(features),
            "exp_logits": self.exponent(features),
            "features": features,
        }
        if digit_vectors is not None:
            outputs["digit_features"] = digit_vectors
        return outputs

    @staticmethod
    def _digit_inputs(outputs, select=None):
        """The (features, digit_features) the digit layers read, optionally masked."""

        features = outputs["features"]
        digit_features = outputs.get("digit_features")
        if select is not None:
            features = features[select]
            digit_features = None if digit_features is None else digit_features[select]
        return features, digit_features

    # -- digits ---------------------------------------------------------------

    def _base(self, features, exp_index, digit_features=None):
        """``(..., num_digits, width)``: the input of every digit position."""

        index = exp_index.clamp(0, self.num_exponents - 1)
        exponent = self.exponent_embedding(index).unsqueeze(-2)
        if digit_features is not None:
            return self.digit_in(digit_features) + exponent
        base = self.digit_in(features).unsqueeze(-2) + exponent
        return base.expand(*base.shape[:-2], self.num_digits, base.shape[-1])

    def _teacher_digit_logits(self, features, exp_index, digits, digit_features=None):
        """``(..., num_digits, 10)`` logits with the gold digits fed in."""

        base = self._base(features, exp_index, digit_features)
        positions = torch.arange(self.num_digits, device=features.device)
        previous = self.digit_previous(positions * 10 + digits.clamp(0, 9))
        prefix = torch.cumsum(previous, dim=-2) - previous  # digits before each position
        logits = self.digit_out(base + self.digit_position.weight + prefix)
        logits = logits.clone()
        logits[..., 0, 0] = -1e4  # the first significant digit is never 0
        return logits

    def _greedy_digits(self, features, exp_index, digit_features=None):
        base = self._base(features, exp_index, digit_features)
        prefix = torch.zeros_like(base[..., 0, :])
        digits, probs = [], []
        for position in range(self.num_digits):
            logits = self.digit_out(base[..., position, :] + self.digit_position.weight[position] + prefix)
            if position == 0:
                logits = logits.clone()
                logits[..., 0] = -1e4
            prob, digit = torch.softmax(logits, dim=-1).max(dim=-1)
            digits.append(digit)
            probs.append(prob)
            prefix = prefix + self.digit_previous(position * 10 + digit)
        return torch.stack(digits, dim=-1), torch.stack(probs, dim=-1)

    def _coefficient(self, features, exp_index, digits, digit_features=None):
        """Multiplier in ``[10^-r, 10^r]`` for the given digits."""

        base = self._base(features, exp_index, digit_features).mean(-2)
        positions = torch.arange(self.num_digits, device=features.device)
        hidden = base + self.digit_previous(positions * 10 + digits.clamp(0, 9)).sum(-2)
        raw = self.coefficient_out(hidden).squeeze(-1)
        return torch.pow(10.0, self.coefficient_range * torch.tanh(raw))

    def _digits_value(self, digits):
        scale = 10.0 ** -torch.arange(self.num_digits, device=digits.device, dtype=torch.float64)
        return (digits.double() * scale).sum(-1)

    # -- loss -----------------------------------------------------------------

    def loss(self, outputs, values, mask):
        if not mask.any():
            return None, {}
        states = sign_states(values)
        terms = [F.cross_entropy(outputs["state_logits"][mask], states[mask])]
        exp_class, digits = scientific_targets(values, mask, (self.exp_low, self.exp_high), self.num_digits)
        stats = {"out_of_range": (mask & (states != SIGN_ZERO) & (exp_class < 0)).sum().detach()}
        use = exp_class >= 0
        if use.any():
            features, digit_features = self._digit_inputs(outputs, use)
            gold_exp = exp_class[use]
            terms.append(F.cross_entropy(outputs["exp_logits"][use], gold_exp))
            digit_logits = self._fp32(
                self._teacher_digit_logits, features, gold_exp, digits[use], digit_features,
            )
            terms.append(F.cross_entropy(digit_logits.reshape(-1, 10), digits[use].reshape(-1)))
            if self.use_coefficient and self.coefficient_loss_coef > 0:
                # Trained on the head's own greedy digits: c fixes what the
                # digits it would decode got wrong.
                with torch.no_grad():
                    greedy, _ = self._fp32(
                        self._greedy_digits, features.detach(), gold_exp,
                        None if digit_features is None else digit_features.detach(),
                    )
                coefficient = self._fp32(self._coefficient, features, gold_exp, greedy, digit_features)
                significand = values[use].abs() / torch.pow(10.0, (gold_exp + self.exp_low).double())
                estimate = self._digits_value(greedy) * coefficient.double()
                relative = ((estimate - significand).abs() / significand).float()
                stats["coefficient_loss"] = relative.mean().detach()
                terms.append(self.coefficient_loss_coef * relative.mean())
        return torch.stack(terms).sum(), stats

    # -- decode ---------------------------------------------------------------

    def decode(self, outputs):
        state_probs = torch.softmax(outputs["state_logits"].float(), dim=-1)
        exp_probs = torch.softmax(outputs["exp_logits"].float(), dim=-1)
        state_conf, state = state_probs.max(dim=-1)
        exp_conf, exp_index = exp_probs.max(dim=-1)
        features, digit_features = self._digit_inputs(outputs)
        digits, digit_probs = self._fp32(self._greedy_digits, features, exp_index, digit_features)
        significand = self._digits_value(digits)
        if self.use_coefficient and self.apply_coefficient:
            significand = significand * self._fp32(
                self._coefficient, features, exp_index, digits, digit_features,
            ).double()
        magnitude = significand * torch.pow(10.0, (exp_index + self.exp_low).double())
        values = torch.where(state == SIGN_NEGATIVE, -magnitude, magnitude)
        values = torch.where(state == SIGN_ZERO, torch.zeros_like(values), values)
        confidence = state_conf.double() * torch.where(
            state == SIGN_ZERO, torch.ones_like(exp_conf), exp_conf * digit_probs.prod(-1),
        ).double()
        return values, confidence


NUMBER_HEADS = {
    "regression": RegressionNumberHead,
    "bins": BinsNumberHead,
    "sci_digits": SciDigitsNumberHead,
}


def build_number_heads(hidden_size: int, dropout: float, number_cfg) -> nn.ModuleDict:
    return nn.ModuleDict({
        kind: NUMBER_HEADS[kind](hidden_size, dropout, getattr(number_cfg, kind))
        for kind in number_cfg.heads
    })


__all__ = [
    "BinsNumberHead",
    "NUMBER_HEADS",
    "RegressionNumberHead",
    "SciDigitsNumberHead",
    "build_number_heads",
    "scientific_targets",
    "sign_states",
]
