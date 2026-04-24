"""GLiNExT Gradio Demo — Multi-task Information Extraction"""

import json
from typing import Dict, List, Optional, Union

import gradio as gr
import torch

from glinext import GLiNExT, FieldType, StructuringOutputFormatter

# ── Model loading ────────────────────────────────────────────────────────────

MODEL_ID = "./logs/multitask_shared/checkpoint-110000"
model: Optional[GLiNExT] = None


def get_model():
    global model
    if model is None:
        model = GLiNExT.from_pretrained(MODEL_ID, load_tokenizer=True)
        model.eval()
    return model


# ── Shared helpers ───────────────────────────────────────────────────────────

def parse_label_groups(groups_json: str) -> Optional[Union[List[str], Dict[str, List[str]]]]:
    """Parse label groups JSON into the format expected by GLiNExT.

    Accepts:
      [{"parent": "name", "labels": ["a", "b"]}, ...]
    Returns:
      - None if empty
      - List[str] if single unnamed group
      - Dict[str, List[str]] if named groups
    """
    if not groups_json or groups_json.strip() in ("", "[]"):
        return None
    try:
        groups = json.loads(groups_json)
    except json.JSONDecodeError:
        return None
    if not groups:
        return None
    # Filter out groups with no labels
    groups = [g for g in groups if g.get("labels")]
    if not groups:
        return None
    if len(groups) == 1:
        return groups[0]["labels"]
    return {g["parent"]: g["labels"] for g in groups}


def _parse_field_type(type_spec: str) -> Union[str, FieldType]:
    """Parse a field type like `str` or `list[int]` into a formatter spec."""
    normalized = type_spec.strip().replace(" ", "")
    if not normalized:
        return "str"
    if normalized == "string":
        return "str"

    if normalized.startswith("list[") and normalized.endswith("]"):
        item_type = normalized[5:-1].strip() or "str"
        if item_type == "string":
            item_type = "str"
        return FieldType("list", list_item_type=item_type)

    return normalized


def _parse_structure_field(label: str) -> tuple[str, Optional[Union[str, FieldType]]]:
    """Parse a structuring field label, optionally with `field:type` syntax."""
    field_name, sep, type_spec = label.rpartition(":")
    if not sep:
        return label.strip(), None
    return field_name.strip(), _parse_field_type(type_spec)


def parse_structure_groups(groups_json: str) -> Optional[Dict[str, Union[List[str], dict]]]:
    """Parse structuring groups, preserving optional per-field type hints."""
    if not groups_json or groups_json.strip() in ("", "[]"):
        return None
    try:
        groups = json.loads(groups_json)
    except json.JSONDecodeError:
        return None
    if not groups:
        return None
    groups = [g for g in groups if g.get("labels")]
    if not groups:
        return None

    parsed_groups = {}
    for group in groups:
        field_names = []
        field_types = {}
        for raw_label in group["labels"]:
            field_name, field_type = _parse_structure_field(raw_label)
            if not field_name:
                continue
            field_names.append(field_name)
            if field_type is not None:
                field_types[field_name] = field_type

        if not field_names:
            continue

        parsed_groups[group["parent"]] = {
            "fields": field_names,
            "field_types": field_types or None,
        }

    return parsed_groups or None


def build_structuring_formatter(
    structures: Optional[Dict[str, Union[List[str], dict]]]
) -> Optional[StructuringOutputFormatter]:
    """Build a formatter when any structuring field carries type metadata."""
    if not structures:
        return None

    schema_types = {}
    for schema_name, schema_spec in structures.items():
        if not isinstance(schema_spec, dict):
            continue
        field_types = schema_spec.get("field_types")
        if field_types:
            schema_types[schema_name] = field_types

    if not schema_types:
        return None

    return StructuringOutputFormatter(schema_types)


def parse_joint_relex_groups(groups_json: str) -> Optional[Dict[str, dict]]:
    """Parse joint relex groups.

    Accepts:
      [{"parent": "name", "entities": ["a"], "relations": ["r"]}, ...]
    """
    if not groups_json or groups_json.strip() in ("", "[]"):
        return None
    try:
        groups = json.loads(groups_json)
    except json.JSONDecodeError:
        return None
    if not groups:
        return None
    groups = [g for g in groups if g.get("entities") and g.get("relations")]
    if not groups:
        return None
    return {
        g["parent"]: {"entities": g["entities"], "relations": g["relations"]}
        for g in groups
    }


def create_label_group_manager(
    task_name: str,
    default_parent: str = "default",
    default_labels: str = "",
    extra_fields: bool = False,
):
    """Create a dynamic label group manager with add/remove buttons.

    Returns (state, render_fn) where state is a gr.State holding the groups JSON.
    """
    groups_state = gr.State(value="[]")

    with gr.Column():
        gr.Markdown(f"### Label Groups")

        # Container for rendered groups
        groups_display = gr.JSON(
            label="Current groups",
            visible=False,
        )

        if not extra_fields:
            with gr.Row():
                parent_input = gr.Textbox(
                    label="Group name",
                    placeholder=default_parent,
                    scale=1,
                )
                labels_input = gr.Textbox(
                    label="Labels (comma-separated)",
                    placeholder=default_labels,
                    scale=3,
                )
                add_btn = gr.Button("+", scale=0, min_width=50, variant="primary")
        else:
            # Joint relex: entities + relations
            with gr.Row():
                parent_input = gr.Textbox(
                    label="Group name",
                    placeholder=default_parent,
                    scale=1,
                )
                labels_input = gr.Textbox(
                    label="Entity types (comma-separated)",
                    placeholder="person, organization, location",
                    scale=2,
                )
                relations_input = gr.Textbox(
                    label="Relation types (comma-separated)",
                    placeholder="works_at, born_in, located_in",
                    scale=2,
                )
                add_btn = gr.Button("+", scale=0, min_width=50, variant="primary")

        # Display current groups as a dataframe
        groups_table = gr.Dataframe(
            headers=["#", "Group", "Labels"] if not extra_fields else ["#", "Group", "Entities", "Relations"],
            datatype=["number", "str", "str"] if not extra_fields else ["number", "str", "str", "str"],
            label="Groups",
            interactive=False,
            col_count=(3 if not extra_fields else 4, "fixed"),
        )

        with gr.Row():
            remove_idx = gr.Number(label="Remove group #", precision=0, minimum=1, scale=1)
            remove_btn = gr.Button("Remove", scale=0, min_width=80, variant="stop")
            clear_btn = gr.Button("Clear all", scale=0, min_width=80)

    def add_group(current_json, parent, labels, *args):
        try:
            groups = json.loads(current_json) if current_json else []
        except json.JSONDecodeError:
            groups = []

        parent = parent.strip() if parent.strip() else f"group_{len(groups) + 1}"
        label_list = [l.strip() for l in labels.split(",") if l.strip()]
        if not label_list:
            return current_json, _groups_to_table(groups, extra_fields)

        if extra_fields:
            rel_list = [r.strip() for r in args[0].split(",") if r.strip()] if args else []
            groups.append({"parent": parent, "entities": label_list, "relations": rel_list})
        else:
            groups.append({"parent": parent, "labels": label_list})

        new_json = json.dumps(groups)
        return new_json, _groups_to_table(groups, extra_fields)

    def remove_group(current_json, idx):
        try:
            groups = json.loads(current_json) if current_json else []
        except json.JSONDecodeError:
            groups = []
        idx = int(idx) - 1
        if 0 <= idx < len(groups):
            groups.pop(idx)
        new_json = json.dumps(groups)
        return new_json, _groups_to_table(groups, extra_fields)

    def clear_groups():
        return "[]", _groups_to_table([], extra_fields)

    if not extra_fields:
        add_btn.click(
            fn=add_group,
            inputs=[groups_state, parent_input, labels_input],
            outputs=[groups_state, groups_table],
        )
    else:
        add_btn.click(
            fn=add_group,
            inputs=[groups_state, parent_input, labels_input, relations_input],
            outputs=[groups_state, groups_table],
        )

    remove_btn.click(
        fn=remove_group,
        inputs=[groups_state, remove_idx],
        outputs=[groups_state, groups_table],
    )

    clear_btn.click(
        fn=clear_groups,
        inputs=[],
        outputs=[groups_state, groups_table],
    )

    return groups_state


def _groups_to_table(groups: list, extra_fields: bool = False) -> list:
    """Convert groups list to a table for display."""
    if not groups:
        return []
    rows = []
    for i, g in enumerate(groups):
        if extra_fields:
            rows.append([
                i + 1,
                g.get("parent", ""),
                ", ".join(g.get("entities", [])),
                ", ".join(g.get("relations", [])),
            ])
        else:
            rows.append([
                i + 1,
                g.get("parent", ""),
                ", ".join(g.get("labels", [])),
            ])
    return rows


# ── Task inference functions ─────────────────────────────────────────────────

def run_ner(text: str, groups_json: str, threshold: float, flat_ner: bool, multi_label: bool):
    if not text.strip():
        return "Please enter some text."
    entities = parse_label_groups(groups_json)
    if entities is None:
        return "Please add at least one label group."
    m = get_model()
    results = m.inference(text, entities=entities, threshold=threshold, flat_ner=flat_ner, multi_label=multi_label)
    ner_results = results.get("ner", [[]])[0]
    return json.dumps(ner_results, indent=2, default=str)


def run_classification(text: str, groups_json: str, threshold: float, multi_label: bool):
    if not text.strip():
        return "Please enter some text."
    classes = parse_label_groups(groups_json)
    if classes is None:
        return "Please add at least one label group."
    m = get_model()
    results = m.inference(text, classes=classes, threshold=threshold, multi_label=multi_label)
    cls_results = results.get("classification", [[]])[0]
    return json.dumps(cls_results, indent=2, default=str)


def run_open_relex(text: str, groups_json: str, threshold: float, flat_ner: bool):
    if not text.strip():
        return "Please enter some text."
    relations = parse_label_groups(groups_json)
    if relations is None:
        return "Please add at least one label group."
    m = get_model()
    results = m.inference(text, relations=relations, threshold=threshold, flat_ner=flat_ner)
    relex_results = results.get("open_relex", [[]])[0]
    return json.dumps(relex_results, indent=2, default=str)


def run_joint_relex(text: str, groups_json: str, threshold: float, flat_ner: bool):
    if not text.strip():
        return "Please enter some text."
    joint_relations = parse_joint_relex_groups(groups_json)
    if joint_relations is None:
        return "Please add at least one group with both entities and relations."
    m = get_model()
    results = m.inference(text, joint_relations=joint_relations, threshold=threshold, flat_ner=flat_ner)
    output = {}
    if "ner" in results:
        output["ner"] = results["ner"][0]
    if "joint_relex" in results:
        output["joint_relex"] = results["joint_relex"][0]
    return json.dumps(output, indent=2, default=str)


def run_structuring(text: str, groups_json: str, threshold: float, flat_ner: bool):
    if not text.strip():
        return "Please enter some text."
    structures = parse_structure_groups(groups_json)
    if structures is None:
        return "Please add at least one schema group."
    m = get_model()
    schema = m.create_schema()
    for schema_name, schema_spec in structures.items():
        if isinstance(schema_spec, dict) and schema_spec.get("field_types"):
            schema.add_structure(schema_name, schema_spec["field_types"])
        elif isinstance(schema_spec, dict):
            schema.add_structure(schema_name, schema_spec.get("fields", []))
        else:
            schema.add_structure(schema_name, schema_spec)
    results = m.inference_from_schema(schema=schema, texts=text, threshold=threshold, flat_ner=flat_ner)
    struct_results = results.get("structuring", [{}])[0]
    return json.dumps(struct_results, indent=2, default=str)


def run_embedding(query: str, candidates: str):
    if not query.strip():
        return "Please enter a query text."
    if not candidates.strip():
        return "Please enter at least one candidate text."
    candidate_list = [c.strip() for c in candidates.split("\n") if c.strip()]
    if not candidate_list:
        return "Please enter at least one candidate text."
    m = get_model()
    all_texts = [query] + candidate_list
    embeddings = m.embed_text(all_texts)
    query_emb = embeddings[0:1]  # (1, D)
    cand_embs = embeddings[1:]   # (N, D)
    # Cosine similarity
    query_norm = query_emb / query_emb.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    cand_norm = cand_embs / cand_embs.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    similarities = (query_norm @ cand_norm.T).squeeze(0).tolist()  # (N,)
    results = [
        {"text": text, "similarity": round(sim, 4)}
        for text, sim in sorted(zip(candidate_list, similarities), key=lambda x: -x[1])
    ]
    return json.dumps(results, indent=2, default=str)


def run_multitask(
    text: str,
    ner_json: str, cls_json: str, relex_json: str,
    joint_json: str, struct_json: str,
    threshold: float, flat_ner: bool, multi_label: bool,
):
    if not text.strip():
        return "Please enter some text."
    entities = parse_label_groups(ner_json)
    classes = parse_label_groups(cls_json)
    relations = parse_label_groups(relex_json)
    joint_relations = parse_joint_relex_groups(joint_json)
    structures = parse_structure_groups(struct_json)

    if all(v is None for v in [entities, classes, relations, joint_relations, structures]):
        return "Please configure at least one task."

    m = get_model()
    results = m.inference(
        text,
        entities=entities,
        classes=classes,
        relations=relations,
        joint_relations=joint_relations,
        structures=structures,
        threshold=threshold,
        flat_ner=flat_ner,
        multi_label=multi_label,
    )
    formatter = build_structuring_formatter(structures)
    if formatter is not None and "structuring" in results:
        results["structuring"] = formatter.format_batch(results["structuring"])
    # Format: convert each task's results to first item (single text)
    output = {}
    for key, val in results.items():
        if val:
            output[key] = val[0]
    return json.dumps(output, indent=2, default=str)


# ── Build the UI ─────────────────────────────────────────────────────────────

EXAMPLE_TEXT = (
    "Elon Musk, CEO of Tesla and SpaceX, announced on January 15, 2025 that "
    "the company would invest $10 billion in a new Gigafactory in Austin, Texas. "
    "The factory will produce the next-generation Roadster and employ over 5,000 workers. "
    "Analysts at Goldman Sachs rated the stock as a strong buy."
)

with gr.Blocks(
    title="GLiNExT Demo",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # GLiNExT — Multi-task Information Extraction
        Extract entities, classify text, find relations, and structure information — all with a single model.

        Add label groups using the **+** button, then click **Run** to see results.
        """
    )

    # ── NER Tab ──────────────────────────────────────────────────────────
    with gr.Tab("NER"):
        with gr.Row():
            with gr.Column(scale=1):
                ner_text = gr.Textbox(label="Text", lines=5, value=EXAMPLE_TEXT)
                ner_groups = create_label_group_manager(
                    "ner",
                    default_parent="general",
                    default_labels="person, organization, location, date",
                )
                with gr.Row():
                    ner_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
                    ner_flat = gr.Checkbox(value=True, label="Flat NER")
                    ner_multi = gr.Checkbox(value=False, label="Multi-label")
                ner_btn = gr.Button("Run NER", variant="primary")
            with gr.Column(scale=1):
                ner_output = gr.Code(label="Results", language="json")

        ner_btn.click(
            fn=run_ner,
            inputs=[ner_text, ner_groups, ner_threshold, ner_flat, ner_multi],
            outputs=ner_output,
        )

    # ── Classification Tab ───────────────────────────────────────────────
    with gr.Tab("Classification"):
        with gr.Row():
            with gr.Column(scale=1):
                cls_text = gr.Textbox(label="Text", lines=5, value=EXAMPLE_TEXT)
                cls_groups = create_label_group_manager(
                    "classification",
                    default_parent="topic",
                    default_labels="technology, finance, politics, sports",
                )
                with gr.Row():
                    cls_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
                    cls_multi = gr.Checkbox(value=False, label="Multi-label")
                cls_btn = gr.Button("Run Classification", variant="primary")
            with gr.Column(scale=1):
                cls_output = gr.Code(label="Results", language="json")

        cls_btn.click(
            fn=run_classification,
            inputs=[cls_text, cls_groups, cls_threshold, cls_multi],
            outputs=cls_output,
        )

    # ── Open Relation Extraction Tab ─────────────────────────────────────
    with gr.Tab("Open Relation Extraction"):
        with gr.Row():
            with gr.Column(scale=1):
                relex_text = gr.Textbox(label="Text", lines=5, value=EXAMPLE_TEXT)
                relex_groups = create_label_group_manager(
                    "open_relex",
                    default_parent="general",
                    default_labels="CEO of, located in, invested in",
                )
                with gr.Row():
                    relex_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
                    relex_flat = gr.Checkbox(value=True, label="Flat NER")
                relex_btn = gr.Button("Run Relation Extraction", variant="primary")
            with gr.Column(scale=1):
                relex_output = gr.Code(label="Results", language="json")

        relex_btn.click(
            fn=run_open_relex,
            inputs=[relex_text, relex_groups, relex_threshold, relex_flat],
            outputs=relex_output,
        )

    # ── Joint Relex Tab ──────────────────────────────────────────────────
    with gr.Tab("Joint NER + Relations"):
        with gr.Row():
            with gr.Column(scale=1):
                joint_text = gr.Textbox(label="Text", lines=5, value=EXAMPLE_TEXT)
                joint_groups = create_label_group_manager(
                    "joint_relex",
                    default_parent="general",
                    extra_fields=True,
                )
                with gr.Row():
                    joint_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
                    joint_flat = gr.Checkbox(value=True, label="Flat NER")
                joint_btn = gr.Button("Run Joint Extraction", variant="primary")
            with gr.Column(scale=1):
                joint_output = gr.Code(label="Results", language="json")

        joint_btn.click(
            fn=run_joint_relex,
            inputs=[joint_text, joint_groups, joint_threshold, joint_flat],
            outputs=joint_output,
        )

    # ── Structuring Tab ──────────────────────────────────────────────────
    with gr.Tab("Structuring"):
        with gr.Row():
            with gr.Column(scale=1):
                struct_text = gr.Textbox(label="Text", lines=5, value=EXAMPLE_TEXT)
                gr.Markdown("Use `field:type` in labels for typed output, for example `name:str, founded_companies:list[str], employee_count:int`.")
                struct_groups = create_label_group_manager(
                    "structuring",
                    default_parent="company",
                    default_labels="name:str, CEO:str, location:str, investment_amount:float",
                )
                with gr.Row():
                    struct_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
                    struct_flat = gr.Checkbox(value=True, label="Flat NER")
                struct_btn = gr.Button("Run Structuring", variant="primary")
            with gr.Column(scale=1):
                struct_output = gr.Code(label="Results", language="json")

        struct_btn.click(
            fn=run_structuring,
            inputs=[struct_text, struct_groups, struct_threshold, struct_flat],
            outputs=struct_output,
        )

    # ── Embedding Tab ────────────────────────────────────────────────────
    with gr.Tab("Embedding"):
        with gr.Row():
            with gr.Column(scale=1):
                emb_query = gr.Textbox(
                    label="Query text",
                    lines=3,
                    value="Tesla is investing in a new factory in Texas.",
                )
                emb_candidates = gr.Textbox(
                    label="Candidate texts (one per line)",
                    lines=8,
                    value=(
                        "A new automotive plant is being built in Austin.\n"
                        "SpaceX launched a rocket to the International Space Station.\n"
                        "The stock market saw significant gains today.\n"
                        "Electric vehicle production is expanding in the US.\n"
                        "The weather in New York is sunny and warm."
                    ),
                )
                emb_btn = gr.Button("Compute Similarity", variant="primary")
            with gr.Column(scale=1):
                emb_output = gr.Code(label="Results (sorted by similarity)", language="json")

        emb_btn.click(
            fn=run_embedding,
            inputs=[emb_query, emb_candidates],
            outputs=emb_output,
        )

    # ── Multi-task Tab ───────────────────────────────────────────────────
    with gr.Tab("Multi-task"):
        mt_text = gr.Textbox(label="Text", lines=5, value=EXAMPLE_TEXT)
        with gr.Row():
            mt_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
            mt_flat = gr.Checkbox(value=True, label="Flat NER")
            mt_multi = gr.Checkbox(value=False, label="Multi-label")

        gr.Markdown("Configure any combination of tasks below:")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### NER")
                mt_ner = create_label_group_manager(
                    "mt_ner", default_parent="general",
                    default_labels="person, organization, location",
                )
            with gr.Column():
                gr.Markdown("#### Classification")
                mt_cls = create_label_group_manager(
                    "mt_cls", default_parent="topic",
                    default_labels="technology, finance",
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Open Relations")
                mt_relex = create_label_group_manager(
                    "mt_relex", default_parent="general",
                    default_labels="CEO of, located in",
                )
            with gr.Column():
                gr.Markdown("#### Joint NER + Relations")
                mt_joint = create_label_group_manager(
                    "mt_joint", default_parent="general",
                    extra_fields=True,
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Structuring")
                mt_struct = create_label_group_manager(
                    "mt_struct", default_parent="company",
                    default_labels="name:str, CEO:str, location:str",
                )
            with gr.Column():
                pass  # balance the layout

        mt_btn = gr.Button("Run Multi-task", variant="primary")
        mt_output = gr.Code(label="Results", language="json")

        mt_btn.click(
            fn=run_multitask,
            inputs=[mt_text, mt_ner, mt_cls, mt_relex, mt_joint, mt_struct,
                    mt_threshold, mt_flat, mt_multi],
            outputs=mt_output,
        )

demo.queue()

if __name__ == "__main__":
    demo.launch(debug=True)
