# finetune-method-picker

English · **[한국어](README.md)** · toggle EN/KR at the top-right of the web UI

**How do I fine-tune this model on my GPU?** — a static web tool that takes your GPU, model, and method and computes training VRAM, the largest trainable model size, data format, data volume, and training time/cost **with the formulas shown**.

Runs 100% in the browser. No server, no API key, no data leaves the page. All estimation logic is client-side JavaScript, and **every number has a "Show formula" you can expand to see the basis and the plugged-in values.** MIT.

> A public-spec, formula-based **planning estimate** — not a substitute for a measured benchmark. Training time is dominated by MFU, so read it as a range. When it won't fit (OOM), it shows exactly how to reduce it. That's what makes the tool trustworthy.

## What it answers

- **Pick a dataset + model and get the best automatically** — pick a real HF dataset (2026 meaningful cases) and objective, example count, and avg tokens fill in; pick a model and it auto-sets the smallest hardware + config that fits, **Full first** (auto-downgrading Full → LoRA → 8-bit LoRA → QLoRA if it won't fit). Change the GPU and it recomputes for that card.
- **OOM debugger** — sweep batch / seq_len / grad accum / gradient checkpointing / optimizer and see per-GPU VRAM live, with concrete fixes when it won't fit.
- **Complete config per tuning method** — Full (FSDP/ZeRO-3/DDP · CPU offload) · LoRA (r/alpha/dropout/target) · 8-bit LoRA (int8 base) · QLoRA (NF4/double-quant/compute dtype) + LR · scheduler · warmup · packing.
- **A runnable TRL scaffold** — copy-paste Python filled with your chosen values (per-objective trainer + LoraConfig/BitsAndBytesConfig + FSDP/DeepSpeed launch).
- **2026 task guide** — per-task method, data volume, and base model (e.g. "learn our house UI code") plus a **is fine-tuning even worth it** gate (three-track market, five win conditions, suspect RAG first).
- **Max trainable model** / **effective batch** = per-device × grad accum × GPU count / optimizer-step count.
- **LoRA helper** — rank r → trainable params (GQA/MLA-aware), α≈2r, target modules.
- **Training time/cost** — C≈6ND ÷ (FLOPS × MFU), rent cost.
- **EN/KR bilingual** — the whole screen renders in the selected language via the top-right toggle.

## Formulas (all visible under "Show formula" in the UI)

- **Training VRAM** = base (params×2, QLoRA ×0.5 NF4, 8-bit LoRA ×1 int8) + training state (trainable × [grad 2 + optim + master 4]) + activations + overhead. FSDP shards base+state across GPUs. — EleutherAI Transformer Math, QLoRA (arXiv:2305.14314)
- **LoRA trainable params** ≈ Σ r·(in+out) over targeted layers, GQA/MLA-aware via kv_dim — LoRA (arXiv:2106.09685)
- **Training time** = 6·N·D ÷ (dense-bf16 FLOPS × GPU count × MFU) — Kaplan (arXiv:2001.08361)
- FLOPS are **dense bf16 (sparsity off)**. Hardware with no defensible spec (pre-order/NPU/Apple) has its time estimate **omitted** (a blank is more honest than a guess).

## Run locally

```bash
python3 -m http.server   # open over http (file:// blocks fetch)
# http://localhost:8000
```

## Verify

```bash
node test/compute.test.cjs                              # deterministic unit gate
node audit/js_dump.cjs > /tmp/js.json && python3 audit/reference_audit.py /tmp/js.json  # JS↔Python parity
```
CI (`.github/workflows/validate.yml`) runs both plus JSON validation on every push.

## Data

- `data/gpus.json` — accelerator specs (VRAM, bandwidth, dense-bf16 FLOPS, price, power), public specs 2026-07
- `data/models.json` — recent open-weight models (params, layers, hidden, kv_dim)
- `data/methods.json` — objectives, tunings, data formats, data volume, distillation, all with source URLs
- `data/datasets.json` — real Hugging Face datasets for 2026 meaningful fine-tuning cases (verified row counts/licenses)
- `data/recipes.json` — task → data → method guide + the "when to fine-tune" gate
- `data/techniques-2026.json` — 2026 recent-techniques shelf (maturity-labeled)

Prose fields render bilingually via `{ko, en}` objects; UI chrome strings live in `assets/i18n.js`.

## Why this is a public good (not an ad)

Fine-tuning method and memory math is scattered across benchmarks and community threads, so only people who already know the answer know the answer. This tool publishes the estimation logic, formulas, and sources transparently so anyone can verify against their own GPU. It sells no product or service.

MIT License.
