"""Loss primitives shared by task heads and model orchestration."""

import inspect
from collections.abc import Callable

import torch
import torch.nn.functional as F


def binary_focal_or_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    focal_loss_alpha: float = 0.25,
    focal_loss_gamma: float = 2.0,
    focal_loss_prob_margin: float = 0.0,
    reduction: str = "none",
    label_smoothing: float = 0.0,
    normalize_prob: bool = True,
    ignore_index: int = -100,
    eps: float = 1e-6,
    **_: object,
) -> torch.Tensor:
    """Return elementwise stable focal loss, or BCE when focal is disabled.

    The focal base term is evaluated in logits space whenever ``logits`` are
    logits.  In particular, hard positive examples retain a useful gradient
    even after their logit falls below the range where ``sigmoid`` followed by
    a clamped logarithm can represent it.  This matters for independent-label
    set prediction, where an early common negative offset must remain
    recoverable.

    A non-positive ``focal_loss_alpha`` disables only alpha balancing. Focal
    modulation remains active while ``focal_loss_gamma`` is positive; plain
    BCE is selected only when both controls are non-positive.
    """

    if reduction not in {"none", "sum", "mean"}:
        raise ValueError(f"Unsupported reduction: {reduction!r}")

    valid_mask = targets != ignore_index
    safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets)).to(
        dtype=logits.dtype
    )
    if label_smoothing:
        safe_targets = safe_targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if normalize_prob:
        probabilities = torch.sigmoid(logits)
        if focal_loss_prob_margin == 0.0:
            # This is algebraically identical to the usual positive/negative
            # log-probability terms, but remains finite with non-zero gradient
            # for logits such as +/-200.
            losses = F.binary_cross_entropy_with_logits(
                logits,
                safe_targets,
                reduction="none",
            )
            negative_probabilities = 1.0 - probabilities
        else:
            margin_probabilities = torch.clamp(
                probabilities - focal_loss_prob_margin,
                min=0.0,
                max=1.0,
            )
            negative_probabilities = 1.0 - margin_probabilities
            positive_term = safe_targets * F.softplus(-logits)
            negative_term = -(1.0 - safe_targets) * torch.log(
                negative_probabilities.clamp(min=eps)
            )
            losses = positive_term + negative_term
    else:
        probabilities = logits
        margin_probabilities = torch.clamp(
            probabilities - focal_loss_prob_margin,
            min=0.0,
            max=1.0,
        )
        negative_probabilities = 1.0 - margin_probabilities
        losses = (
            -safe_targets * torch.log(probabilities.clamp(min=eps))
            -(1.0 - safe_targets)
            * torch.log(negative_probabilities.clamp(min=eps))
        )

    if focal_loss_gamma > 0.0:
        target_probabilities = (
            probabilities * safe_targets
            + negative_probabilities * (1.0 - safe_targets)
        )
        losses = losses * (1.0 - target_probabilities).pow(focal_loss_gamma)

    # Alpha values <= 0 mean "no class balancing". This also avoids the
    # surprising alpha=0 behaviour that erases every positive target.
    if focal_loss_alpha > 0.0:
        alpha_weights = (
            focal_loss_alpha * safe_targets
            + (1.0 - focal_loss_alpha) * (1.0 - safe_targets)
        )
        losses = losses * alpha_weights

    losses = losses * valid_mask.to(dtype=losses.dtype)

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.sum() / valid_mask.sum().clamp(min=1)
    raise AssertionError("unreachable")


def configured_binary_loss(
    config,
    logits: torch.Tensor,
    targets: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Resolve per-task focal controls and apply the shared binary loss policy."""

    loss_kwargs = dict(kwargs)
    for name, default in (
        ("focal_loss_alpha", 0.25),
        ("focal_loss_gamma", 2.0),
        ("focal_loss_prob_margin", 0.0),
    ):
        value = getattr(config, name, None)
        loss_kwargs.setdefault(name, default if value is None else value)
    return binary_focal_or_bce(logits, targets, **loss_kwargs)


def binary_loss_with_focal_overrides(
    loss_fn: Callable[..., torch.Tensor],
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    focal_loss_alpha: float | None = None,
    focal_loss_gamma: float | None = None,
    focal_loss_prob_margin: float | None = None,
    **loss_kwargs,
) -> torch.Tensor:
    """Call an elementwise binary loss with optional focal overrides.

    Project loss callables use the descriptive ``focal_loss_*`` names, while
    GLiNER's public focal helper uses ``alpha``, ``gamma``, and
    ``prob_margin``.  Supporting both signatures keeps task-level component
    controls usable with the model's configured loss closure and with heads
    invoked directly in downstream code.

    Callables that expose neither interface are still valid custom losses;
    they are called without focal keywords.
    """

    overrides = {
        "focal_loss_alpha": focal_loss_alpha,
        "focal_loss_gamma": focal_loss_gamma,
        "focal_loss_prob_margin": focal_loss_prob_margin,
    }
    overrides = {
        name: value for name, value in overrides.items() if value is not None
    }
    if not overrides:
        return loss_fn(logits, targets, **loss_kwargs)

    try:
        parameters = inspect.signature(loss_fn).parameters
    except (TypeError, ValueError):
        parameters = {}

    accepts_arbitrary_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    aliases = {
        "focal_loss_alpha": "alpha",
        "focal_loss_gamma": "gamma",
        "focal_loss_prob_margin": "prob_margin",
    }
    accepted = dict(loss_kwargs)
    for name, value in overrides.items():
        if accepts_arbitrary_keywords or name in parameters:
            accepted[name] = value
        elif aliases[name] in parameters:
            accepted[aliases[name]] = value
    return loss_fn(logits, targets, **accepted)


__all__ = [
    "binary_focal_or_bce",
    "binary_loss_with_focal_overrides",
    "configured_binary_loss",
]
