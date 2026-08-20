# INT8 Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantize the local `BAAI/bge-reranker-v2-m3` reranker to INT8 (bitsandbytes `load_in_8bit`) so GPU memory drops from ~2.27 GB (FP32) to ~570 MB, with an FP32 fallback on CPU-only machines.

**Architecture:** `LocalBgeReranker.__init__` loads the model via `AutoModelForSequenceClassification.from_pretrained(..., quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")` when CUDA is available; otherwise falls back to the existing FP32 load. The `rerank()` interface is unchanged.

**Tech Stack:** Python 3.11+, torch 2.13.0+cu130, transformers 4.57.6, bitsandbytes, accelerate, pytest.

## Global Constraints

- Python >= 3.11 (pyproject.toml `requires-python`)
- torch >= 2.6, transformers >= 4.41,<5 (pyproject.toml — do not change versions)
- Import of `torch`/`transformers`/`bitsandbytes` inside `LocalBgeReranker.__init__` stays lazy (rest of pipeline must work without these installed)
- Interface `Reranker.rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]` must not change
- No config flag: INT8 is always used when CUDA is available
- Unit tests must not download or load the real model

---
## File Structure

- `pyproject.toml` — add `bitsandbytes`, `accelerate` to `[project].dependencies`
- `rag_core/reranker.py` — quantized load path in `LocalBgeReranker.__init__` (lines 26-37), FP32 fallback
- `tests/test_reranker_int8.py` — new mocked unit tests (no real model download)
- `scripts/smoke_reranker_int8.py` — throwaway smoke script (delete after use) that loads the real model and reports GPU memory

---

### Task 1: Add bitsandbytes + accelerate dependencies

**Files:**
- Modify: `pyproject.toml:6-15` (dependencies list)

**Interfaces:**
- Consumes: nothing
- Produces: `bitsandbytes` and `accelerate` importable in the venv

- [ ] **Step 1: Add dependencies**

Add to the `dependencies` list in `pyproject.toml` (keep alphabetical-ish order used there):

```toml
    "accelerate>=0.30",
    "bitsandbytes>=0.44",
```

- [ ] **Step 2: Install into the venv**

Run: `uv sync`
Expected: resolves and installs without errors. If resolution fails on `bitsandbytes>=0.44` with torch 2.13.0+cu130, try a newer compatible version (e.g. `bitsandbytes>=0.46`) or build from source as a fallback; record what was needed in the commit message.

- [ ] **Step 3: Verify imports**

Run: `uv run python -c "import bitsandbytes, accelerate; print(bitsandbytes.__version__)"`
Expected: prints a version number without error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add bitsandbytes and accelerate for INT8 reranker"
```

---

### Task 2: INT8 load path with FP32 fallback in LocalBgeReranker

**Files:**
- Modify: `rag_core/reranker.py:26-37` (`LocalBgeReranker.__init__`)
- Create: `tests/test_reranker_int8.py`

**Interfaces:**
- Consumes: `DEFAULT_RERANK_MODEL` from `rag_core.config` (already imported)
- Produces: unchanged — `LocalBgeReranker(device: str)`, `model_name: str`, `rerank(query, chunks) -> list[Chunk]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reranker_int8.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

from rag_core.reranker import LocalBgeReranker


def test_int8_load_used_when_cuda_available(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    captured: dict = {}

    def fake_from_pretrained(model_name: str, **kwargs) -> SimpleNamespace:
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return SimpleNamespace(to=lambda device: SimpleNamespace(eval=lambda: None))

    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda name: SimpleNamespace(),
    )
    LocalBgeReranker("fake/model")
    assert captured["kwargs"]["quantization_config"].load_in_8bit is True
    assert captured["kwargs"]["device_map"] == "auto"
    assert "quantization_config" in captured["kwargs"], (
        "must pass quantization_config on CUDA"
    )


def test_fp32_fallback_when_no_cuda(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    captured: dict = {}

    def fake_from_pretrained(model_name: str, **kwargs) -> SimpleNamespace:
        captured["kwargs"] = kwargs
        return SimpleNamespace(to=lambda device: SimpleNamespace(eval=lambda: None))

    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda name: SimpleNamespace(),
    )
    reranker = LocalBgeReranker("fake/model")
    assert reranker.device == "cpu"
    assert "quantization_config" not in captured["kwargs"]
```

Note: the monkeypatch of `torch.cuda.is_available` works only if the test imports
`LocalBgeReranker` (which imports `torch` inside `__init__`); the `from __future__
import annotations` on `reranker.py` keeps imports lazy. `"torch"` and
`"transformers"` are patched as module strings, so this works regardless.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reranker_int8.py -v`
Expected: FAIL — `LocalBgeReranker.__init__` currently calls `from_pretrained` without `quantization_config`, so `captured["kwargs"]["quantization_config"]` raises `KeyError`.

- [ ] **Step 3: Implement the INT8 load path**

Replace the body of `LocalBgeReranker.__init__` in `rag_core/reranker.py:26-37` with:

```python
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        import torch

        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            try:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
                self._device = "cuda"
            except Exception:
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                ).to("cpu")
                self._device = "cpu"
        else:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            ).to("cpu")
            self._device = "cpu"
        self._model.eval()
```

Keep the rest of the file unchanged (docstring, `device`, `model_name`, `rerank`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reranker_int8.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest`
Expected: all existing tests still PASS (they use `FakeReranker`, no real model).

- [ ] **Step 6: Type-check**

Run: `uv run mypy rag_core/reranker.py tests/test_reranker_int8.py`
Expected: no errors. If mypy complains about `AutoModelForSequenceClassification.from_pretrained` returns, keep the existing `# type: ignore` style or add `# type: ignore[no-untyped-call]` on the new call.

- [ ] **Step 7: Commit**

```bash
git add rag_core/reranker.py tests/test_reranker_int8.py
git commit -m "feat: load reranker in INT8 via bitsandbytes with FP32 fallback"
```

---

### Task 3: Real-model smoke test + memory measurement

**Files:**
- Create: `scripts/smoke_reranker_int8.py` (throwaway — delete before finishing)

**Interfaces:**
- Consumes: `LocalBgeReranker` from `rag_core.reranker`
- Produces: measured GPU memory allocation for the INT8-loaded model

- [ ] **Step 1: Write the smoke script**

Create `scripts/smoke_reranker_int8.py`:

```python
from __future__ import annotations

from rag_core.models import Chunk, Source
from rag_core.reranker import LocalBgeReranker

reranker = LocalBgeReranker()
print("device:", reranker.device)

import torch

if torch.cuda.is_available():
    before = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU alloc after load: {before / 2**20:.0f} MiB / {total / 2**20:.0f} MiB")


def source(doc_id: str, title: str, chapter: str) -> Source:
    return Source(
        document_id=doc_id,
        document_title=title,
        chapter=chapter,
        course_code="IT1",
        kind="slide",
        language="vi",
    )


chunks = [
    Chunk(source=source("doc1", "Cơ sở dữ liệu", "Chương 1"),
          text="Cơ sở dữ liệu là một tập hợp dữ liệu có tổ chức, được lưu trữ và truy xuất bằng hệ quản trị CSDL."),
    Chunk(source=source("doc2", "OOP", "Chương 2"),
          text="Lập trình hướng đối tượng mô hình hóa thế giới thực bằng các đối tượng và lớp."),
    Chunk(source=source("doc3", "Thuật toán", "Chương 3"),
          text="Thuật toán sắp xếp nổi bọt có độ phức tạp O(n^2)."),
]
ranked = reranker.rerank("Cơ sở dữ liệu là gì?", chunks)
print("top:", ranked[0].source)
```

- [ ] **Step 2: Run the smoke script**

Run: `uv run python scripts/smoke_reranker_int8.py`
Expected: prints `device: cuda`, a GPU allocation number well under 1 GiB (target ~570-800 MiB), and a top-1 chunk about databases. Also run it twice in a row to confirm stability.

- [ ] **Step 3: Compare with FP32 to confirm the reduction**

Run: `uv run python -c "
from rag_core.reranker import LocalBgeReranker
import torch
r = LocalBgeReranker('BAAI/bge-reranker-v2-m3')
torch.cuda.empty_cache()
print('alloc MiB:', torch.cuda.memory_allocated() // 2**20)
"` after temporarily forcing the FP32 path (or note the model file size 2.27 GB vs INT8 measurement).
Expected: INT8 allocation is roughly 25-30% of FP32 (i.e. ~70-75% reduction).

- [ ] **Step 4: Delete the smoke script**

Run: `rm scripts/smoke_reranker_int8.py` and `rmdir scripts` if now empty. (Script was throwaway; keep the repo clean.)

- [ ] **Step 5: Final verification**

Run: `uv run pytest && uv run mypy rag_core/reranker.py`
Expected: all tests pass, mypy clean.

- [ ] **Step 6: Commit any remaining changes**

```bash
git add -A
git commit -m "feat: verify INT8 reranker reduces GPU memory ~75%"
```

---

## Self-Review

- **Spec coverage:** spec's 4 requirements (deps, quantized load + fallback, no config flag, verification) map to Task 1 (deps), Task 2 (load path + fallback + tests), Task 3 (smoke + memory). No gaps.
- **Placeholder scan:** all steps contain concrete code/commands. No TBD/TODO.
- **Type consistency:** `LocalBgeReranker(model_name: str = DEFAULT_RERANK_MODEL)` and `rerank(query: str, chunks: list[Chunk]) -> list[Chunk]` unchanged everywhere; `device` property returns `"cuda"`/`"cpu"` as before.
