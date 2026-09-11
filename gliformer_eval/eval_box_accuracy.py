"""Measure object-detection box accuracy of a checkpoint on real images.

Runs the full model forward, matches each image's predicted boxes to its GT
boxes with the Hungarian matcher, and reports mean matched IoU + recall@0.5.
A direct, checkpoint-comparable number instead of eyeballing the demo.

    python glinex_eval/eval_box_accuracy.py <abs_checkpoint_path> [n_images]
"""
import json
import sys

import torch

from gliformer import GLiFormer
from gliformer.tasks.box_ops import aligned_box_iou, pairwise_box_iou

CKPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    model = GLiFormer.from_pretrained(CKPT, load_tokenizer=True).to(DEVICE, dtype=torch.float32)
    model.eval()
    head = model.model.heads["object_detection"]

    rows = json.load(open("data/lvis_validation.json"))[:N]
    collator = model._create_data_collator()
    batch = collator(rows)
    batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}

    captured = {}

    def pre_hook(module, args, kwargs):
        captured["args"], captured["kwargs"] = args, kwargs
        raise StopIteration

    h = head.register_forward_pre_hook(pre_hook, with_kwargs=True)
    with torch.no_grad():
        try:
            model.model(**batch)
        except StopIteration:
            pass
    h.remove()

    kw = captured["kwargs"]
    flat_inputs = kw["flat_inputs"]
    cls_labels = kw["object_detection_class_labels"]
    bbox_labels = kw["object_detection_bbox_labels"]
    obj_mask = kw["object_detection_object_mask"]

    with torch.no_grad():
        predictions = head._compute_detection(flat_inputs)

    matched_ious, oracle_ious, n_gt, n_rec = [], [], 0, 0
    for b in range(predictions.class_logits.shape[0]):
        gt_n = int((obj_mask[b] > 0).sum().item())
        n_gt += gt_n
        matches = head._match_single(
            predictions.class_logits[b],
            predictions.boxes_xyxy[b],
            cls_labels[b],
            bbox_labels[b],
            obj_mask[b],
            predictions.anchor_mask[b],
        )
        for a, o in matches:
            predicted_box = predictions.boxes_xyxy[b, a][None]
            target_box = bbox_labels[b, o][None].to(predicted_box)
            iou = aligned_box_iou(predicted_box, target_box).item()
            matched_ious.append(iou)
            n_rec += int(iou >= 0.5)
        # Oracle: for each GT, the best IoU over ALL predicted boxes (allow reuse).
        # Upper bound on box quality, independent of the model's class/objectness
        # driven assignment. If this is also low, the boxes themselves are bad.
        gt_idx = (obj_mask[b] > 0).nonzero(as_tuple=True)[0]
        for o in gt_idx.tolist():
            gt = bbox_labels[b, o][None].to(predictions.boxes_xyxy)
            best = pairwise_box_iou(predictions.boxes_xyxy[b], gt).max().item()
            oracle_ious.append(best)

    miou = sum(matched_ious) / max(len(matched_ious), 1)
    p25 = sum(i >= 0.25 for i in matched_ious) / max(len(matched_ious), 1)
    p50 = sum(i >= 0.5 for i in matched_ious) / max(len(matched_ious), 1)
    p75 = sum(i >= 0.75 for i in matched_ious) / max(len(matched_ious), 1)
    obj_prob = torch.sigmoid(predictions.objectness_logits.float())
    print(f"checkpoint: {CKPT}")
    print(
        f"images {predictions.class_logits.shape[0]}  GT objects {n_gt}  "
        f"matched {len(matched_ious)}"
    )
    oracle_miou = sum(oracle_ious) / max(len(oracle_ious), 1)
    oracle_p50 = sum(i >= 0.5 for i in oracle_ious) / max(len(oracle_ious), 1)
    print(f"mean matched IoU      {miou:.3f}")
    print(f"ORACLE best-box IoU   {oracle_miou:.3f}  (>=0.5: {oracle_p50:.1%})  "
          f"-- upper bound, any slot may claim any GT")
    print(f"matched IoU>=0.25     {p25:.1%}")
    print(f"matched IoU>=0.50     {p50:.1%}  (recall over matched)")
    print(f"matched IoU>=0.75     {p75:.1%}")
    print(f"recall@0.5 over GT    {n_rec / max(n_gt, 1):.1%}")
    print(f"objectness  mean {obj_prob.mean():.3f}  max {obj_prob.max():.3f}")


if __name__ == "__main__":
    main()
