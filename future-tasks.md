# Future Tasks

| Task name | Description | What exists now | What we should do |
|---|---|---|---|
| Wire `inference_mode=True` for LLM object detection | Enable end-to-end visual object detection in the main trajectory pipeline instead of reading object lists from GT trajectory JSON. | `LLMObjectDetector` supports `inference_mode=True` and sends a 1-shot reference image + labels to the LLM, but `LLMVisualComponentsMixin.init_visual_components` always creates detectors with the default `inference_mode=False`, so `detect()` returns GT objects from `fewshot_examples[0][1]` without calling the LLM. Only component tests (e.g. depot/gripper) pass `inference_mode=True`. | Pass `inference_mode=True` from `LLMVisualComponentsMixin` (or make it configurable via handler init / `config.yaml`). Verify object lists are stable across frames, update prompts if needed, and compare pipeline quality vs the current GT-object shortcut. |
