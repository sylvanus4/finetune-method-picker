#!/usr/bin/env python3
"""Independent Python reimplementation of the fine-tuning formulas, compared field-by-field
against the shipped JS (audit/js_dump.cjs output). Catches formula drift between JS and the
documented math. Usage: node audit/js_dump.cjs > /tmp/js.json && python3 audit/reference_audit.py /tmp/js.json"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {m["id"]: m for m in json.loads((ROOT / "data/models.json").read_text())["models"]}
GPUS = {g["id"]: g for g in json.loads((ROOT / "data/gpus.json").read_text())["gpus"]}

WEIGHT_BF16, WEIGHT_NF4, WEIGHT_INT8, GRAD_BF16, MASTER_FP32 = 2, 0.5, 1, 2, 4
OPT_STATE = {"adamw": 8, "adamw_8bit": 2, "paged_adamw": 8}
ACT_BYTES, GC_FACTOR, GC_TAX = 16, 0.22, 1.30
OVER_BASE, OVER_FRAC = 1.2, 0.05
FLOP = 6

CFGS = [
    dict(tuning="full", numGpus=1, perDeviceBatch=1, gradAccum=1, seqLen=2048, gradCkpt=False, optimizer="adamw", datasetExamples=10000, avgTokensPerExample=512, epochs=3, mfu=0.35),
    dict(tuning="lora", numGpus=2, perDeviceBatch=2, gradAccum=8, seqLen=4096, gradCkpt=True, optimizer="adamw", loraR=16, targetModules="all", datasetExamples=5000, avgTokensPerExample=1024, epochs=2, mfu=0.4),
    dict(tuning="qlora", numGpus=1, perDeviceBatch=1, gradAccum=16, seqLen=2048, gradCkpt=True, optimizer="paged_adamw", loraR=32, targetModules="attn_all", datasetExamples=2000, avgTokensPerExample=512, epochs=1, mfu=0.3),
    dict(tuning="lora_8bit", numGpus=1, perDeviceBatch=1, gradAccum=16, seqLen=2048, gradCkpt=True, optimizer="adamw_8bit", loraR=16, targetModules="all", datasetExamples=3000, avgTokensPerExample=768, epochs=2, mfu=0.35),
]

def lora_params(hidden, n_layers, r, tgt, kv_dim=None, intermediate=None):
    if not hidden or not n_layers:
        return None
    h = hidden
    kv = kv_dim if (kv_dim and kv_dim > 0) else h
    i = intermediate if (intermediate and intermediate > 0) else 3.5 * h
    q = r * (h + h); o = r * (h + h); k = r * (h + kv); v = r * (h + kv)
    if tgt == "attn":
        per = q + v
    elif tgt == "attn_all":
        per = q + k + v + o
    else:
        per = q + k + v + o + r * (h + i) + r * (h + i) + r * (i + h)
    return per * n_layers

def compute(m, g, c):
    N = m["total_params_b"] * 1e9
    tuning = c.get("tuning", "lora"); ng = max(1, c.get("numGpus", 1))
    strategy = "fsdp" if ng > 1 else "ddp"
    pdb = max(1, c.get("perDeviceBatch", 1)); ga = max(1, c.get("gradAccum", 1))
    seq = max(1, c.get("seqLen", 1024)); ckpt = bool(c.get("gradCkpt")); opt = c.get("optimizer", "adamw")
    r = max(1, c.get("loraR", 16)); tgt = c.get("targetModules", "all")
    if tuning == "full":
        train = N
    else:
        ex = lora_params(m.get("hidden"), m.get("n_layers"), r, tgt, m.get("kv_dim"), m.get("intermediate"))
        train = ex if ex is not None else 0.005 * N
    base_bytes = WEIGHT_NF4 if tuning == "qlora" else (WEIGHT_INT8 if tuning == "lora_8bit" else WEIGHT_BF16)
    baseGB = N * base_bytes / 1e9
    per_param = GRAD_BF16 + OPT_STATE.get(opt, 8) + MASTER_FP32
    stateGB = train * per_param / 1e9
    actRaw = pdb * seq * (m.get("hidden") or 4096) * (m.get("n_layers") or 32) * ACT_BYTES / 1e9
    actGB = actRaw * (GC_FACTOR if ckpt else 1)
    sharded = baseGB + stateGB
    per_state = sharded / ng if strategy == "fsdp" else sharded
    overGB = OVER_BASE + OVER_FRAC * (baseGB / ng)
    perGpu = per_state + actGB + overGB
    fits = perGpu <= g["vram_gb"]
    eff = pdb * ga * ng
    D = max(0, c.get("datasetExamples", 0)) * max(1, c.get("avgTokensPerExample", 512)) * max(0, c.get("epochs", 1))
    flops = FLOP * N * D
    time_ok = g.get("flops_bf16_tf") is not None and D > 0
    hours = None
    if time_ok:
        fps = g["flops_bf16_tf"] * 1e12 * ng * c.get("mfu", 0.35)
        hours = (flops / fps) * (GC_TAX if ckpt else 1) / 3600
    return dict(baseGB=baseGB, trainStateGB=stateGB, activationsGB=actGB, perGpuVramGB=perGpu,
                fits=fits, effectiveBatch=eff, trainableParams=train, trainFlops=flops,
                trainHours=hours, timeAvailable=time_ok)

def near(a, b, tol=1e-4):
    if a is None or b is None:
        return a is None and b is None
    d = abs(a - b)
    return d <= tol or d <= tol * max(abs(a), abs(b))

def main():
    js = json.loads(Path(sys.argv[1]).read_text())
    fields = ["baseGB", "trainStateGB", "activationsGB", "perGpuVramGB", "effectiveBatch", "trainableParams", "trainFlops", "trainHours"]
    fail = 0
    print("== JS <-> Python parity (fine-tuning formulas) ==")
    for row in js:
        py = compute(MODELS[row["mid"]], GPUS[row["gid"]], CFGS[row["ci"]])
        if py["fits"] != row["fits"]:
            print(f"  FAIL fits {row['mid']}/{row['gid']}/cfg{row['ci']}: js={row['fits']} py={py['fits']}"); fail += 1
        if py["timeAvailable"] != row["timeAvailable"]:
            print(f"  FAIL timeAvailable {row['mid']}/{row['gid']}/cfg{row['ci']}"); fail += 1
        for f in fields:
            if not near(py[f], row[f]):
                print(f"  FAIL {f} {row['mid']}/{row['gid']}/cfg{row['ci']}: js={row[f]} py={py[f]}"); fail += 1
    print(f"\n{len(js)} scenarios checked, {fail} field mismatches")
    if fail == 0:
        print("ALL CHECKS PASSED")
    sys.exit(1 if fail else 0)

if __name__ == "__main__":
    main()
