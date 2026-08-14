# GLiNExT evaluation

Run these commands from the `GLiNext/` project directory.

## NER benchmarks

```bash
python glinex_eval/eval.py
```

Override the defaults with `--model CHECKPOINT` and `--data NER_DIRECTORY`.

## Zero-shot classification

```bash
python glinex_eval/test_glinext.py --model PATH_TO_CHECKPOINT
```

## Sentence similarity

```bash
python glinex_eval/eval_sts.py --model PATH_TO_CHECKPOINT
```

## Text-to-JSON

```bash
python glinex_eval/eval_glinext_text2json.py \
  --model-path logs/structuring_multi_level/checkpoint-7000 \
  --data-path ../data/top_1000_en_annoated_changed_eval.jsonl
```

Use `--num-samples N` for a smaller run. Without `--data-path`, the evaluator
downloads the configured Hugging Face dataset. Results are written to
`eval_results_glinext_text2json/` by default.

## Structuring JSONL

```bash
python glinex_eval/eval_glinext_structuring.py \
  --model-path PATH_TO_CHECKPOINT \
  --test-data-path PATH_TO_DATA.jsonl
```

## Object-detection boxes

```bash
python glinex_eval/eval_box_accuracy.py PATH_TO_CHECKPOINT 32
```
