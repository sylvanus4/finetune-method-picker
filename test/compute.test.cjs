/* Objective gate for the fine-tuning estimation core. Run: node test/compute.test.cjs
   Loads the SHIPPED compute.js + SHIPPED data files, asserts defensible properties of the
   training-memory, effective-batch, capacity, and time formulas. */
const fs = require("fs");
const path = require("path");
const { computeTraining, loraTrainableParams } = require("../assets/compute.js");

const root = path.join(__dirname, "..");
const models = JSON.parse(fs.readFileSync(path.join(root, "data/models.json"))).models;
const gpus = JSON.parse(fs.readFileSync(path.join(root, "data/gpus.json"))).gpus;
const M = id => { const m = models.find(x => x.id === id); if (!m) throw new Error("missing model " + id); return m; };
const G = id => { const g = gpus.find(x => x.id === id); if (!g) throw new Error("missing gpu " + id); return g; };

let pass = 0, fail = 0;
function ok(name, cond) { cond ? (pass++, console.log("  PASS " + name)) : (fail++, console.log("  FAIL " + name)); }

console.log("finetune compute.js gate:");
const base = { numGpus: 1, perDeviceBatch: 1, gradAccum: 1, seqLen: 2048, epochs: 1, datasetExamples: 1000, avgTokensPerExample: 512 };
const q8 = M("qwen3-8b"), h100 = G("h100-80");

// 1. Base bytes per param
ok("qlora base = 0.5 B/param", computeTraining(q8, h100, { ...base, tuning: "qlora" }).baseBytes === 0.5);
ok("full base = 2 B/param", computeTraining(q8, h100, { ...base, tuning: "full" }).baseBytes === 2);

// 2. VRAM ordering full > lora > qlora
const vFull = computeTraining(q8, h100, { ...base, tuning: "full" }).perGpuVramGB;
const vLora = computeTraining(q8, h100, { ...base, tuning: "lora" }).perGpuVramGB;
const vQ = computeTraining(q8, h100, { ...base, tuning: "qlora" }).perGpuVramGB;
ok("VRAM: full > lora", vFull > vLora);
ok("VRAM: lora > qlora", vLora > vQ);

// 3. LoRA trainable << full
const rLora = computeTraining(q8, h100, { ...base, tuning: "lora", loraR: 16 });
ok("LoRA trainable << N", rLora.trainableParams < 0.05 * rLora.N);
ok("LoRA trainablePct < 5%", rLora.trainablePct < 5);

// 4. LoRA formula ordering
const tAttn = loraTrainableParams(4096, 36, 16, "attn");
const tAll4 = loraTrainableParams(4096, 36, 16, "attn_all");
const tAll = loraTrainableParams(4096, 36, 16, "all");
ok("LoRA modules: attn < attn_all < all", tAttn < tAll4 && tAll4 < tAll);
ok("LoRA params scale with r", loraTrainableParams(4096, 36, 32, "all") > tAll);

// 4b. GQA-aware kv_dim and explicit intermediate refine the LoRA count
ok("GQA kv_dim<h cuts attn_all params", loraTrainableParams(4096, 36, 16, "attn_all", 1024) < tAll4);
ok("smaller intermediate -> fewer MLP LoRA params",
   loraTrainableParams(4096, 36, 16, "all", 1024, 11008) < loraTrainableParams(4096, 36, 16, "all", 1024));

// 5. Effective batch
const eb = computeTraining(q8, h100, { ...base, numGpus: 4, perDeviceBatch: 2, gradAccum: 8 });
ok("effective batch = perDev×accum×gpus", eb.effectiveBatch === 2 * 8 * 4);

// 6. FSDP shards vs DDP replicates
const fsdp = computeTraining(q8, h100, { ...base, tuning: "full", numGpus: 4, strategy: "fsdp" }).perGpuVramGB;
const ddp = computeTraining(q8, h100, { ...base, tuning: "full", numGpus: 4, strategy: "ddp" }).perGpuVramGB;
ok("FSDP per-gpu < DDP per-gpu (4 GPU)", fsdp < ddp);

// 7. Grad checkpointing cuts activations
const noCk = computeTraining(q8, h100, { ...base, gradCkpt: false }).activationsGB;
const ck = computeTraining(q8, h100, { ...base, gradCkpt: true }).activationsGB;
ok("grad checkpointing cuts activations", ck < noCk);

// 8. Longer seq -> more activations
const sShort = computeTraining(q8, h100, { ...base, seqLen: 1024 }).activationsGB;
const sLong = computeTraining(q8, h100, { ...base, seqLen: 8192 }).activationsGB;
ok("longer seq -> more activation VRAM", sLong > sShort);

// 9. Capacity
const maxFull = computeTraining(q8, h100, { ...base, tuning: "full" }).maxModelParamsB;
const maxQ = computeTraining(q8, h100, { ...base, tuning: "qlora" }).maxModelParamsB;
ok("max trainable: qlora > full", maxQ > maxFull);
const maxSmallGpu = computeTraining(q8, G("rtx4090"), { ...base, tuning: "qlora" }).maxModelParamsB;
ok("bigger GPU -> bigger max trainable", maxQ > maxSmallGpu);

// 10. Time: 6ND, only when FLOPS known
const t = computeTraining(q8, h100, { ...base, datasetExamples: 10000, avgTokensPerExample: 512, epochs: 3 });
ok("trainFlops = 6·N·D", Math.abs(t.trainFlops - 6 * t.N * t.trainTokens) < 1);
ok("time available on H100 (flops known)", t.timeAvailable === true && t.trainHours > 0);
const tApple = computeTraining(q8, G("m2-ultra-192"), { ...base, datasetExamples: 10000 });
ok("time hidden when FLOPS null (Apple)", tApple.timeAvailable === false && tApple.trainHours === null);

// 11. OOM
const oom = computeTraining(M("deepseek-v4-pro"), G("rtx4090"), { ...base, tuning: "full", numGpus: 1 });
ok("huge model full on 24GB -> !fits", oom.fits === false);
ok("OOM -> nonempty fixes", oom.oomFixes.length > 0);

// 12. LoRA alpha = 2r
ok("recommended alpha = 2r", computeTraining(q8, h100, { ...base, tuning: "lora", loraR: 32 }).lora.recommendedAlpha === 64);

// 13. Optimizer memory
const mAdam = computeTraining(q8, h100, { ...base, tuning: "full", optimizer: "adamw" }).trainStateGB;
const m8 = computeTraining(q8, h100, { ...base, tuning: "full", optimizer: "adamw_8bit" }).trainStateGB;
ok("8-bit optimizer uses less than adamw", m8 < mAdam);
ok("paged optimizer flagged offloaded", computeTraining(q8, h100, { ...base, optimizer: "paged_adamw" }).optimizerOffloaded === true);

// 14. All finite for every model on H100 (qlora ×8)
let allFinite = true;
for (const m of models) {
  const x = computeTraining(m, h100, { ...base, tuning: "qlora", numGpus: 8 });
  if (![x.perGpuVramGB, x.trainableParams, x.effectiveBatch, x.maxModelParamsB].every(Number.isFinite)) allFinite = false;
}
ok("all models finite on H100 qlora×8", allFinite);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
