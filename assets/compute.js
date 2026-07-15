/* Pure fine-tuning estimation core — no DOM, no I/O. Runs in the browser (global LLMFT) and in
   Node (module.exports) so the SAME code path is unit-tested (test/compute.test.cjs) and shipped.
   Every number the UI shows comes from here with its intermediates exposed, so the page can render
   the formula + the plugged-in values ("show your work"). Sources are cited inline per constant. */
(function (root) {
  // --- Memory accounting, bytes per parameter (mixed-precision training) ---
  // Source: EleutherAI "Transformer Math 101" (blog.eleuther.ai/transformer-math) — full Adam mixed
  // precision ≈ 16 B/param = 2 (bf16 weight) + 2 (bf16 grad) + 4+4 (fp32 Adam m,v) + 4 (fp32 master).
  const WEIGHT_BF16 = 2;            // frozen/base weight in bf16
  const WEIGHT_NF4  = 0.5;          // QLoRA 4-bit NF4 base (arXiv:2305.14314); +doublequant ~0.03 ignored
  const GRAD_BF16   = 2;            // gradient, per TRAINABLE param
  const MASTER_FP32 = 4;            // fp32 master copy, per trainable param
  const OPT_STATE = { adamw: 8, adamw_8bit: 2, paged_adamw: 8 }; // Adam m+v; 8-bit optim = 1B each (bitsandbytes)

  // Activation memory is the ROUGHEST term — heuristic coefficient, no exact vendor spec (labeled in UI).
  // Grad checkpointing trades ~+30% compute to cut stored activations ~5x (Transformer Math / HF perf docs).
  const ACT_BYTES = 16;            // heuristic bytes/(token·layer·hidden-ish) without checkpointing
  const GC_FACTOR = 0.22;          // grad-checkpointing residual (~5x reduction), heuristic
  const GC_COMPUTE_TAX = 1.30;     // grad-checkpointing recompute → step ~30% slower

  const OVERHEAD_BASE = 1.2;       // CUDA context + framework, GB (heuristic)
  const OVERHEAD_FRAC = 0.05;      // + 5% of base weights, heuristic

  const FLOP_PER_TOKEN_PER_PARAM = 6; // C ≈ 6·N·D  (Kaplan et al. arXiv:2001.08361)

  // LoRA trainable params ≈ Σ_targeted r·(in+out). Source: LoRA paper arXiv:2106.09685. We approximate
  // module in/out dims from hidden h and intermediate i≈3.5h (SwiGLU-typical).
  // targetModules: 'attn' (q,v) | 'attn_all' (q,k,v,o) | 'all' (attn + gate,up,down).
  function loraTrainableParams(hidden, nLayers, r, targetModules) {
    if (!hidden || !nLayers) return null; // caller falls back to a % heuristic
    const h = hidden, i = 3.5 * h;
    let perLayer;
    if (targetModules === "attn")          perLayer = 2 * (r * (h + h));            // q,v
    else if (targetModules === "attn_all") perLayer = 4 * (r * (h + h));            // q,k,v,o (ignores GQA)
    else /* all */                         perLayer = 4 * (r * (h + h))             // q,k,v,o
                                                     + 2 * (r * (h + i))            // gate, up
                                                     + 1 * (r * (i + h));           // down
    return perLayer * nLayers;
  }

  function overhead(baseGB) { return OVERHEAD_BASE + OVERHEAD_FRAC * baseGB; }

  // === MAIN ===
  function computeTraining(model, gpu, cfg) {
    const N = model.total_params_b * 1e9;
    const tuning = cfg.tuning || "lora";            // 'full' | 'lora' | 'qlora'
    const numGpus = Math.max(1, cfg.numGpus || 1);
    const strategy = cfg.strategy || (numGpus > 1 ? "fsdp" : "ddp");
    const perDeviceBatch = Math.max(1, cfg.perDeviceBatch || 1);
    const gradAccum = Math.max(1, cfg.gradAccum || 1);
    const seqLen = Math.max(1, cfg.seqLen || 1024);
    const gradCkpt = !!cfg.gradCkpt;
    const optimizer = cfg.optimizer || "adamw";
    const r = Math.max(1, cfg.loraR || 16);
    const targetModules = cfg.targetModules || "all";

    // --- trainable params ---
    let trainableParams, trainableSource;
    if (tuning === "full") { trainableParams = N; trainableSource = "full (all params)"; }
    else {
      const exact = loraTrainableParams(model.hidden, model.n_layers, r, targetModules);
      if (exact != null) { trainableParams = exact; trainableSource = "LoRA formula 2·r·Σ(in+out)"; }
      else { trainableParams = 0.005 * N; trainableSource = "LoRA ~0.5% of N (fallback, arch unknown)"; }
    }
    const trainablePct = (trainableParams / N) * 100;

    // --- VRAM components (GB) ---
    const baseBytes = tuning === "qlora" ? WEIGHT_NF4 : WEIGHT_BF16;
    const baseGB = (N * baseBytes) / 1e9;
    const optBytes = OPT_STATE[optimizer] != null ? OPT_STATE[optimizer] : OPT_STATE.adamw;
    const trainPerParam = GRAD_BF16 + optBytes + MASTER_FP32;     // per trainable param
    const trainStateGB = (trainableParams * trainPerParam) / 1e9;
    const optimizerOffloaded = optimizer === "paged_adamw";

    const actRaw = (perDeviceBatch * seqLen * (model.hidden || 4096) * (model.n_layers || 32) * ACT_BYTES) / 1e9;
    const activationsGB = actRaw * (gradCkpt ? GC_FACTOR : 1);

    // FSDP shards base+state across GPUs; DDP replicates. Activations are per-device.
    const shardedGB = baseGB + trainStateGB;
    const perGpuStateGB = strategy === "fsdp" ? shardedGB / numGpus : shardedGB;
    const overheadGB = overhead(baseGB / numGpus);
    const perGpuVramGB = perGpuStateGB + activationsGB + overheadGB;
    const fits = perGpuVramGB <= gpu.vram_gb;
    const vramHeadroomGB = gpu.vram_gb - perGpuVramGB;

    // --- effective batch (top beginner confusion) ---
    const effectiveBatch = perDeviceBatch * gradAccum * numGpus;

    // --- OOM fixes (only when it doesn't fit) ---
    const oomFixes = [];
    if (!fits) {
      if (tuning === "full") oomFixes.push("full → LoRA/QLoRA (base 동결로 학습 상태 급감)");
      if (tuning === "lora") oomFixes.push("LoRA → QLoRA (base를 4-bit NF4로)");
      if (!gradCkpt) oomFixes.push("gradient checkpointing 켜기 (액티베이션 ~5x↓, ~30% 느려짐)");
      if (perDeviceBatch > 1) oomFixes.push("per-device 배치를 1로, grad_accum↑로 effective batch 유지");
      if (seqLen > 1024) oomFixes.push("max_seq_len 낮추기 (액티베이션 선형↓)");
      if (optimizer === "adamw") oomFixes.push("8-bit 또는 paged optimizer로 옵티마이저 상태↓");
      if (strategy === "ddp" && numGpus > 1) oomFixes.push("DDP → FSDP (base+상태 샤딩)");
      oomFixes.push("GPU 수 늘리기 (FSDP 샤딩)");
    }

    // --- max trainable model size on this setup (skipped in the recursive search to avoid infinite loop) ---
    const maxModelParamsB = cfg._skipMax ? null : maxTrainableModelB(gpu, cfg, model.hidden, model.n_layers);

    // --- training time (only if GPU has a defensible dense-bf16 FLOPS spec) ---
    const datasetExamples = Math.max(0, cfg.datasetExamples || 0);
    const avgTok = Math.max(1, cfg.avgTokensPerExample || 512);
    const epochs = Math.max(0, cfg.epochs || 1);
    const trainTokens = datasetExamples * avgTok * epochs;
    const trainFlops = FLOP_PER_TOKEN_PER_PARAM * N * trainTokens; // LoRA ~doesn't cut FLOPs (base fwd/bwd dominates)
    const mfu = cfg.mfu != null ? cfg.mfu : 0.35;
    let timeAvailable = false, trainHours = null, gpuHours = null, costUSD = null, flopsPerSec = null;
    if (gpu.flops_bf16_tf != null && trainTokens > 0) {
      timeAvailable = true;
      flopsPerSec = gpu.flops_bf16_tf * 1e12 * numGpus * mfu;
      const seconds = (trainFlops / flopsPerSec) * (gradCkpt ? GC_COMPUTE_TAX : 1);
      trainHours = seconds / 3600;
      gpuHours = trainHours * numGpus;
      if (gpu.rent_usd_hr != null) costUSD = gpuHours * gpu.rent_usd_hr;
    }

    // --- overfitting heuristic (eval divergence isn't pre-computable; flag risky configs) ---
    const overfitFlags = [];
    if (epochs > 3) overfitFlags.push(epochs + " epochs — SFT는 보통 1–3 (반복=과적합·품질저하, Raschka)");
    if (datasetExamples > 0 && datasetExamples < 500 && tuning === "full")
      overfitFlags.push("작은 데이터(<500) + full FT → 과적합·망각 위험, LoRA 권장");
    if (datasetExamples > 0 && datasetExamples < 100)
      overfitFlags.push("데이터 <100 — 대부분 목표에 부족(heuristic)");

    const lora = (tuning !== "full") ? { r, recommendedAlpha: 2 * r, trainableParams, trainablePct, targetModules } : null;

    return {
      N, tuning, strategy, numGpus, perDeviceBatch, gradAccum, seqLen, gradCkpt, optimizer, r, targetModules,
      trainableParams, trainablePct, trainableSource,
      baseGB, trainStateGB, activationsGB, overheadGB, perGpuVramGB, fits, vramHeadroomGB, optimizerOffloaded,
      baseBytes, trainPerParam,
      effectiveBatch, oomFixes, maxModelParamsB,
      timeAvailable, trainTokens, trainFlops, mfu, flopsPerSec, trainHours, gpuHours, costUSD,
      overfitFlags, lora,
    };
  }

  // Largest model (B params) that still fits at the given tuning/batch/seq on this GPU set.
  function maxTrainableModelB(gpu, cfg, hidden, nLayers) {
    let lo = 0.1, hi = 5000, best = 0;
    for (let it = 0; it < 40; it++) {
      const mid = (lo + hi) / 2;
      const r = computeTraining({ total_params_b: mid, hidden: hidden, n_layers: nLayers }, gpu,
        Object.assign({}, cfg, { datasetExamples: 0, _skipMax: true }));
      if (r.fits) { best = mid; lo = mid; } else { hi = mid; }
    }
    return best;
  }

  const api = { computeTraining, loraTrainableParams, maxTrainableModelB,
    consts: { WEIGHT_BF16, WEIGHT_NF4, GRAD_BF16, MASTER_FP32, OPT_STATE, ACT_BYTES, GC_FACTOR, GC_COMPUTE_TAX, FLOP_PER_TOKEN_PER_PARAM } };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LLMFT = api;
})(typeof self !== "undefined" ? self : this);
