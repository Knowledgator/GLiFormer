"""A small stand-in for GLiFormer that mimics ``inference_requests`` output."""

from __future__ import annotations

import re
from types import SimpleNamespace

import torch

from gliformer.serve import GLiFormerServeConfig, GLiFormerServer

NAME_RE = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b")


class FakeGLiFormer:
    """Deterministic fake heads.

    * classification: the first label scores ``class_scores[0]``, the rest
      follow ``class_scores`` (default 0.9, 0.2, 0.1, ...);
    * NER: every capitalized phrase is tagged with the group's first label;
    * joint relations: the first two names are linked by the first relation;
    * structuring: field values come from ``struct_values`` by field name.
    """

    def __init__(self, *, typed: bool = False, relation_head: str = "joint_relex", embedding: bool = True):
        structuring = SimpleNamespace(typed_fields=typed)
        self.config = SimpleNamespace(
            model_variant="text",
            ner_config={},
            classification_config={},
            joint_relex_config={} if relation_head == "joint_relex" else None,
            open_relex_config={} if relation_head == "open_relex" else None,
            structuring_config=structuring,
            embedding_config={} if embedding else None,
        )
        self.model = SimpleNamespace(
            token_rep_layer=SimpleNamespace(bert_layer=SimpleNamespace(model=SimpleNamespace()))
        )
        self.calls: list[tuple[list[dict], dict]] = []
        self.class_scores = [0.9, 0.2, 0.1, 0.05]
        self.struct_values: dict[str, object] = {}
        self.ner_score = 0.9
        self.fail_with: Exception | None = None

    def eval(self):
        return self

    def compile(self):
        self.compiled = True

    def create_inference_collator(self, return_tokens=True):
        return {"return_tokens": return_tokens}

    # -- heads -----------------------------------------------------------

    def inference_requests(self, requests, **kwargs):
        self.calls.append((list(requests), kwargs))
        if self.fail_with is not None:
            raise self.fail_with
        return [self._row(request) for request in requests]

    @staticmethod
    def _groups(value):
        if isinstance(value, list):
            return {None: value}
        return value or {}

    def _row(self, request):
        text = request["text"]
        request = {
            **request,
            "classes": self._groups(request.get("classes")),
            "entities": self._groups(request.get("entities")),
        }
        output: dict = {}
        if request.get("classes"):
            output["classification"] = [
                [
                    {"class_name": label, "score": self.class_scores[min(index, len(self.class_scores) - 1)]}
                    for index, label in enumerate(labels)
                ]
                for labels in request["classes"].values()
            ]
        entities = request.get("entities") or {}
        joint = request.get("joint_relations") or {}
        names = [(match.start(), match.end(), match.group(0)) for match in NAME_RE.finditer(text)]
        if entities:
            ner = []
            for group_index, labels in enumerate(entities.values()):
                for start, end, name in names:
                    ner.append({
                        "start": start, "end": end, "text": name,
                        "label": labels[0], "score": self.ner_score,
                        "group_index": group_index,
                    })
            output["ner"] = ner
        if joint:
            relations = []
            for offset, config in enumerate(joint.values()):
                if len(names) >= 2 and config["relations"]:
                    (hs, he, head), (ts, te, tail) = names[0], names[1]
                    relations.append({
                        "head": {"start": hs, "end": he, "text": head, "type": config["entities"][0]},
                        "tail": {"start": ts, "end": te, "text": tail, "type": config["entities"][-1]},
                        "relation": config["relations"][0],
                        "score": 0.8,
                        "group_index": len(entities) + offset,
                    })
            output["joint_relex"] = relations
        structures = request.get("structures")
        if isinstance(structures, dict) and "$root" in structures:
            output["structuring"] = self._structure(structures["$root"])
        return output

    def _structure(self, template):
        if isinstance(template, dict):
            return {
                key.lstrip("!"): self._structure_value(key.lstrip("!"), value)
                for key, value in template.items()
            }
        return self._structure_value("", template)

    def _structure_value(self, name, template):
        if isinstance(template, list) and template and isinstance(template[0], dict):
            return self.struct_values.get(name, [])
        if isinstance(template, dict) and "$type" not in template:
            return self._structure(template)
        return self.struct_values.get(name)

    def embed_text(self, texts, **kwargs):
        self.calls.append(([{"text": text} for text in texts], {"embed_text": True, **kwargs}))
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([
                float("apple" in lowered) + 0.1,
                float("phone" in lowered or "smartphone" in lowered) + 0.1,
                0.1,
            ])
        return torch.tensor(vectors)


def make_server(model: FakeGLiFormer | None = None, **overrides):
    model = model or FakeGLiFormer()
    settings = {
        "model": "test/gliformer",
        "device": "cpu",
        "dtype": "float32",
        "enable_compilation": False,
        "enable_memory_calibration": False,
    }
    settings.update(overrides)
    config = GLiFormerServeConfig(**settings)
    return GLiFormerServer(config, model=model), model


def body(input, tasks, **options):
    request = {"model": "test/gliformer", "input": input, "tasks": tasks}
    if options:
        request["options"] = options
    return request
