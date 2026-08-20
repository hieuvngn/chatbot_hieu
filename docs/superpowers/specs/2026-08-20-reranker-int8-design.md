# Design: INT8 (bitsandbytes) quantization for LocalBgeReranker

Date: 2026-08-20
Status: Approved (design review passed, pending spec review)

## Goal

Switch `BAAI/bge-reranker-v2-m3` from FP32 (~2.27 GB) to INT8 8-bit quantization
(~570 MB weights, ~75% reduction) so the local reranker uses significantly less
GPU memory while keeping the rest of the RAG pipeline unchanged.

## Background

- The reranker is the only model that runs locally
  (`rag_core/reranker.py`, `LocalBgeReranker`).
- It is loaded via `AutoModelForSequenceClassification.from_pretrained(...)` and
  moved to CUDA when available, else CPU. No dtype or quantization is applied,
  so it runs FP32.
- Model is XLM-RoBERTa-large based (~568M params): FP32 = 2.27 GB, INT8 = ~568 MB.
- The environment: torch 2.13.0+cu130, transformers 4.57.6, CUDA available.
  `bitsandbytes` and `accelerate` are NOT installed yet.
- Unit tests use `FakeReranker` (tests/test_rag_core.py), so no test changes are
  required for behaviour.

## Approach

Use HuggingFace-native bitsandbytes 8-bit quantization:

- Add `bitsandbytes` and `accelerate` to project dependencies (pyproject.toml).
- In `LocalBgeReranker.__init__`, load the model with
  `BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["classifier"])`
  and `device_map="auto"`; bitsandbytes places the quantized model on GPU
  automatically, so the explicit `.to(device)` call is removed.
- The classifier (scoring head) is excluded from quantization (`llm_int8_skip_modules`):
  a fully-quantized head collapses all logits to a near-constant (~9.0) for every
  chunk, making reranking arbitrary. Keeping the head in FP32 restores
  discriminating scores matching FP32 (measured: INT8 [8.90, −8.69, −10.83] vs
  FP32 [8.98, −8.56, −10.78]) while the transformer body (the bulk of params)
  stays 8-bit.
- Tokenizer, `model.eval()`, and `rerank()` forward pass stay unchanged
  (interface unchanged: `rerank(query, chunks) -> list[Chunk]`).
- Fallback: if CUDA is unavailable (or the bitsandbytes load raises), fall back
  to the existing FP32 path so the pipeline still works on CPU-only machines and
  preserves the current lazy-import property.
- Expected memory: ~800 MiB measured (FP32 was 2165 MiB) — ~63% reduction.
- Expected accuracy impact: none observed on the smoke input (logits match FP32).

## Files changed

- `pyproject.toml` — add `bitsandbytes`, `accelerate` to dependencies.
- `rag_core/reranker.py` — quantized load path in `__init__` with FP32 fallback.

## Risks

- bitsandbytes wheel must be compatible with torch 2.13.0+cu130. If no prebuilt
  wheel matches, may need to build from source or pin a compatible
  bitsandbytes version. Handle during the install step.

## Verification

- Run the test suite: `uv run pytest`.
- Smoke test: load the real model once, rerank a small (query, chunk) list,
  assert the ranked order is plausible and the scores still discriminate
  (relevant chunk clearly ahead of irrelevant ones — not collapsed to a
  constant), and report measured GPU memory.
- Confirm the FP32 fallback still works on a CPU-only path (mocked/lint check:
  fallback branch reachable without CUDA).