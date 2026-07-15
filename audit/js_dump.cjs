/* Dump JS computeTraining results over a fixed grid, for the independent Python parity audit.
   Usage: node audit/js_dump.cjs > /tmp/js.json */
const fs = require("fs"), path = require("path");
const { computeTraining } = require("../assets/compute.js");
const root = path.join(__dirname, "..");
const models = JSON.parse(fs.readFileSync(path.join(root, "data/models.json"))).models;
const gpus = JSON.parse(fs.readFileSync(path.join(root, "data/gpus.json"))).gpus;

const modelIds = ["qwen3-8b", "gemma-4-12b", "deepseek-v4-pro"];
const gpuIds = ["h100-80", "rtx4090", "b200", "m2-ultra-192"];
const cfgs = [
  { tuning: "full", numGpus: 1, perDeviceBatch: 1, gradAccum: 1, seqLen: 2048, gradCkpt: false, optimizer: "adamw", datasetExamples: 10000, avgTokensPerExample: 512, epochs: 3, mfu: 0.35 },
  { tuning: "lora", numGpus: 2, perDeviceBatch: 2, gradAccum: 8, seqLen: 4096, gradCkpt: true, optimizer: "adamw", loraR: 16, targetModules: "all", datasetExamples: 5000, avgTokensPerExample: 1024, epochs: 2, mfu: 0.4 },
  { tuning: "qlora", numGpus: 1, perDeviceBatch: 1, gradAccum: 16, seqLen: 2048, gradCkpt: true, optimizer: "paged_adamw", loraR: 32, targetModules: "attn_all", datasetExamples: 2000, avgTokensPerExample: 512, epochs: 1, mfu: 0.3 },
];
const out = [];
for (const mid of modelIds) for (const gid of gpuIds) for (let ci = 0; ci < cfgs.length; ci++) {
  const m = models.find(x => x.id === mid), g = gpus.find(x => x.id === gid);
  const r = computeTraining(m, g, cfgs[ci]);
  out.push({ mid, gid, ci, baseGB: r.baseGB, trainStateGB: r.trainStateGB, activationsGB: r.activationsGB,
    perGpuVramGB: r.perGpuVramGB, fits: r.fits, effectiveBatch: r.effectiveBatch, trainableParams: r.trainableParams,
    trainFlops: r.trainFlops, trainHours: r.trainHours, timeAvailable: r.timeAvailable, maxModelParamsB: r.maxModelParamsB });
}
process.stdout.write(JSON.stringify(out));
