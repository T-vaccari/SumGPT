# SumGPT

SumGPT is a compact arithmetic experiment migrated from the original CustomGPT implementation to `llm-library`.

The historical implementation is preserved by the local Git tag `legacy-customgpt-v1`. The current adapter imports `TransformerLM` from the GPU Conda environment's `llm-library` installation, pinned there to commit `ee3eb00b3826956ab518ccbd4a39abca172ad5f1`.

This task uses its fixed 13-character arithmetic vocabulary directly; a BPE tokenizer is unnecessary for this synthetic dataset. The task-specific adapter owns record construction, the loss mask before `=`, and greedy decoding.

Run on the server in `ml-gfx1010-rc`:

```bash
python dataset.py
python training.py
python main.py
```

New data and checkpoints are written to `data/` and `checkpoints/llm-library/`; historical checkpoint files are left untouched.

The initial GPU run reached 100.00% on 1,000 sampled held-out records after 1,000 training steps.
