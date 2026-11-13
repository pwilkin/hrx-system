# Test fixtures

Documents of the Qwen3.8-27B GPU/NPU hybrid, copied without payloads (no
programs, weights or tokenizer: the tests stub `loom-serve`).

- `hybrid-package/`: `model.json`, `deployment.json` and `devices.json` of the
  runnable package (the scenario a user serves).
- `hybrid-publish/`: the same plus `info.json` and `llm_config.json` from the
  publication profile, whose preset serves 40960 tokens of the model's native
  262144.
