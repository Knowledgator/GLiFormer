"""Head-only experiment: can the detection head localize small LVIS boxes?

Capture a frozen `flat_inputs` + GT batch from the trained checkpoint, then
train a freshly-initialized ObjectDetectionHead on those frozen features and
measure how well the matched predicted boxes fit the GT boxes. Compare:
  A) plain IoU loss   + L1-only matcher           (old behavior)
  B) GIoU loss        + L1+GIoU matcher            (new behavior)
"""
import copy
import json
import sys

import torch

from glinext import GLiNExT
from glinext.tasks.box_ops import aligned_box_iou, aligned_generalized_box_iou
from glinext.tasks.vision.model import ObjectDetectionHead

CKPT = sys.argv[1] if len(sys.argv) > 1 else "./logs/vision_only/checkpoint-88000"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 6
STEPS = 400


def _detection_loss_with_overlap(output, bbox_labels, coefficient, overlap_fn):
    """Replace the head's configured GIoU term with an experiment metric.

    The production detector intentionally uses GIoU. This experiment also
    needs the old plain-IoU objective, so adjust only that term explicitly
    while retaining the head's class, L1, objectness, and matching losses.
    """
    if coefficient == 0 or overlap_fn is aligned_generalized_box_iou:
        return output.loss

    predicted = []
    targets = []
    boxes = output.extra["bbox_preds"]
    for batch_idx, matches in (output.extra.get("matches") or {}).items():
        for prediction_idx, target_idx in matches:
            predicted.append(boxes[batch_idx, prediction_idx])
            targets.append(bbox_labels[batch_idx, target_idx])
    if not predicted:
        return output.loss

    predicted = torch.stack(predicted)
    targets = torch.stack(targets).to(device=predicted.device, dtype=predicted.dtype)
    denominator = predicted.shape[0]
    production_loss = (
        1.0 - aligned_generalized_box_iou(predicted, targets)
    ).sum() / denominator
    experiment_loss = (1.0 - overlap_fn(predicted, targets)).sum() / denominator
    return output.loss + coefficient * (experiment_loss - production_loss)


def main():
    torch.manual_seed(0)
    model = GLiNExT.from_pretrained(CKPT, load_tokenizer=True).to(DEVICE, dtype=torch.float32)
    model.eval()
    config = model.config

    # Build a labeled batch through the training collator.
    rows = json.load(open("data/lvis_validation.json"))[:N_SAMPLES]
    collator = model._create_data_collator()
    batch = collator(rows)
    batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}

    # Capture the head's call args (frozen features + GT) via a pre-hook.
    captured = {}

    def pre_hook(module, args, kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise StopIteration  # short-circuit: we only need the inputs

    h = model.model.heads["object_detection"].register_forward_pre_hook(pre_hook, with_kwargs=True)
    with torch.no_grad():
        try:
            model.model(**batch)
        except StopIteration:
            pass
    h.remove()

    shared = captured["args"][0]
    kwargs = captured["kwargs"]
    flat_inputs = kwargs["flat_inputs"]

    def detach(x):
        return x.detach().clone() if torch.is_tensor(x) else x

    # Freeze everything the head consumes.
    for fld in flat_inputs.__dataclass_fields__:
        setattr(flat_inputs, fld, detach(getattr(flat_inputs, fld)))
    frozen = {k: detach(v) for k, v in kwargs.items() if k != "dependency_outputs"}
    frozen["dependency_outputs"] = {}
    frozen.pop("base_loss_fn", None)

    n_gt = int((frozen["object_detection_object_mask"] > 0).sum().item())
    print(f"captured batch: {N_SAMPLES} imgs, {n_gt} GT objects, "
          f"feat {flat_inputs.words_embedding.shape}")

    def run(use_giou, bbox_coef=None, giou_coef=None, steps=STEPS):
        torch.manual_seed(42)
        experiment_config = copy.deepcopy(config)
        detection_config = experiment_config.object_detection_config
        detection_config.matcher_giou_cost = 2.0 if use_giou else 0.0
        if bbox_coef is not None:
            detection_config.bbox_loss_coef = bbox_coef
        if giou_coef is not None:
            detection_config.iou_loss_coef = giou_coef
        head = ObjectDetectionHead.from_config(experiment_config).to(
            DEVICE,
            dtype=torch.float32,
        )
        overlap_fn = aligned_generalized_box_iou if use_giou else aligned_box_iou
        opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
        loss = None
        for step in range(steps):
            head.train()
            out = head(shared, **frozen)
            loss = _detection_loss_with_overlap(
                out,
                frozen["object_detection_bbox_labels"],
                float(detection_config.iou_loss_coef),
                overlap_fn,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
            opt.step()
        # Evaluate matched-box fit.
        head.eval()
        with torch.no_grad():
            predictions = head._compute_detection(flat_inputs)
            ious, l1s = [], []
            for b in range(predictions.class_logits.shape[0]):
                matches = head._match_single(
                    predictions.class_logits[b], predictions.boxes_xyxy[b],
                    frozen["object_detection_class_labels"][b],
                    frozen["object_detection_bbox_labels"][b],
                    frozen["object_detection_object_mask"][b],
                    predictions.anchor_mask[b],
                )
                for a, o in matches:
                    pred = predictions.boxes_xyxy[b, a]
                    tgt = frozen["object_detection_bbox_labels"][b, o].to(pred)
                    ious.append(aligned_box_iou(pred[None], tgt[None]).item())
                    l1s.append((pred - tgt).abs().sum().item())
        miou = sum(ious) / max(len(ious), 1)
        ml1 = sum(l1s) / max(len(l1s), 1)
        return float(loss.detach()), miou, ml1, len(ious)

    # Oracle ceiling: best IoU each GT box can reach against a same-sized box
    # centered at the nearest 14x14 patch center (the feature-grid resolution).
    import itertools
    side = 14
    centers = [((c + 0.5) / side, (r + 0.5) / side) for r, c in itertools.product(range(side), range(side))]
    centers_t = torch.tensor(centers)
    gt_all = []
    om = frozen["object_detection_object_mask"]
    bl = frozen["object_detection_bbox_labels"]
    for b in range(bl.shape[0]):
        for o in range(bl.shape[1]):
            if om[b, o] > 0:
                gt_all.append(bl[b, o].cpu())
    gt_all = torch.stack(gt_all)
    gw = (gt_all[:, 2] - gt_all[:, 0]).clamp(min=1e-3)
    gh = (gt_all[:, 3] - gt_all[:, 1]).clamp(min=1e-3)
    best = []
    for i in range(gt_all.shape[0]):
        cx, cy = centers_t[:, 0], centers_t[:, 1]
        cand = torch.stack([cx - gw[i] / 2, cy - gh[i] / 2, cx + gw[i] / 2, cy + gh[i] / 2], dim=-1)
        best.append(
            aligned_box_iou(cand, gt_all[i][None].expand_as(cand)).max().item()
        )
    print(f"ORACLE (correct size, nearest 14x14 patch center): mean_IoU {sum(best)/len(best):.3f}")

    configs = [
        ("A IoU  L1=10 iou=2  400st", False, 10.0, 2.0, 400),
        ("B GIoU L1=10 giou=2 400st", True, 10.0, 2.0, 400),
        ("A IoU  L1=10 iou=2  1500st", False, 10.0, 2.0, 1500),
        ("B GIoU L1=10 giou=2 1500st", True, 10.0, 2.0, 1500),
    ]
    for label, flag, bc, gc, st in configs:
        loss, miou, ml1, nm = run(flag, bc, gc, st)
        print(f"{label:30s}  loss {loss:7.3f}  matched {nm:3d}  "
              f"mean_IoU {miou:.3f}  mean_L1 {ml1:.3f}")


if __name__ == "__main__":
    main()
