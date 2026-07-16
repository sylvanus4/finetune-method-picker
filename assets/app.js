/* finetune-method-picker — client-side UI glue. No backend, no keys, no data leaves the browser.
   Pure estimation core lives in compute.js (LLMFT.computeTraining), shared with the Node unit tests. */
(function () {
  const $ = id => document.getElementById(id);
  const fmt = (x, d = 1) => (x == null || isNaN(x) ? "—" : Number(x).toLocaleString("en-US", { maximumFractionDigits: d }));
  const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  let MODELS = [], GPUS = [], KB = null, TECH = null, RECIPES = null, DATASETS = [];

  // --- Best-practice hardware: cheapest GPU config that fits the preset with real headroom ---
  // Order = cost-efficient training GPUs small->large, all with a known dense-BF16 FLOPS spec (so time
  // shows). Try 1,2,4,8 GPUs; pick the first (fewest, smallest) that fits with >=12% VRAM headroom.
  function recommendBestGpu(model, presetCfg) {
    const order = ["rtx4090", "a100-40", "l40s", "a6000", "a100-80", "h100-80", "h200", "b200", "gb200"];
    for (const ng of [1, 2, 4, 8, 16, 32, 72]) {
      for (const gid of order) {
        const g = GPUS.find(x => x.id === gid);
        if (!g || g.flops_bf16_tf == null) continue;
        const r = LLMFT.computeTraining(model, g, Object.assign({}, presetCfg, { numGpus: ng, datasetExamples: 0, _skipMax: true }));
        if (r.fits && r.vramHeadroomGB >= g.vram_gb * 0.12) return { gpu: gid, numGpus: ng };
      }
    }
    return null; // nothing in the sane cluster range fits -> caller downgrades tuning
  }

  // Full config set per tuning method (every method carries ITS complete knobs, defaults sane).
  function presetFor(model, tuning) {
    const moe = !!model.moe;
    const p = { tuning, targetModules: moe ? "attn_all" : "all", loraR: 16, loraAlpha: 32, loraDropout: 0.05,
      perDeviceBatch: 1, gradAccum: 16, seqLen: 2048, gradCkpt: true, epochs: 2,
      scheduler: "cosine", warmup: 0.03, precision: "bf16", packing: true,
      parallelism: "fsdp", cpuOffload: false, quantType: "nf4", computeDtype: "bfloat16", doubleQuant: true };
    if (tuning === "full")      { p.optimizer = "adamw";       p.lr = "1e-5"; }   // full: LR ~10x lower
    else if (tuning === "lora") { p.optimizer = "adamw_8bit";  p.lr = "2e-4"; }
    else /* qlora */            { p.optimizer = "paged_adamw"; p.lr = "2e-4"; }
    return p;
  }

  // Best practice = highest-quality tuning that FITS (full-first), + the hardware that fits it.
  function recommendSetup(model) {
    for (const tuning of ["full", "lora", "qlora"]) {
      const ps = presetFor(model, tuning);
      const hw = recommendBestGpu(model, ps);
      if (hw) return Object.assign(ps, hw);
    }
    return Object.assign(presetFor(model, "qlora"), { gpu: "h200", numGpus: 8 }); // last resort (may OOM, shown)
  }

  // Best case for a SPECIFIC GPU: highest-quality tuning + fewest GPUs of THAT card that fit.
  function recommendSetupForGpu(model, gpuId) {
    const g = GPUS.find(x => x.id === gpuId);
    if (!g) return recommendSetup(model);
    for (const tuning of ["full", "lora", "qlora"]) {
      const ps = presetFor(model, tuning);
      for (const ng of [1, 2, 4, 8, 16, 32, 72]) {
        const r = LLMFT.computeTraining(model, g, Object.assign({}, ps, { numGpus: ng, datasetExamples: 0, _skipMax: true }));
        if (r.fits && r.vramHeadroomGB >= g.vram_gb * 0.12) return Object.assign(ps, { gpu: gpuId, numGpus: ng });
      }
    }
    return Object.assign(presetFor(model, "qlora"), { gpu: gpuId, numGpus: 72 }); // best effort on this card (may OOM, shown)
  }

  function setRadio(name, val) { const el = document.querySelector(`input[name=${name}][value=${val}]`); if (el) el.checked = true; }

  function applyPreset(model, announce) { applyPresetObj(recommendSetup(model), announce, "model"); }

  function applyPresetObj(ps, announce, origin) {
    setRadio("tuning", ps.tuning);
    $("optimizer").value = ps.optimizer;
    $("loraR").value = ps.loraR;
    $("loraAlpha").value = ps.loraAlpha;
    $("loraDropout").value = ps.loraDropout;
    $("targetModules").value = ps.targetModules;
    $("batch").value = ps.perDeviceBatch;
    $("accum").value = ps.gradAccum;
    $("seq").value = ps.seqLen;
    $("gradCkpt").checked = ps.gradCkpt;
    $("epochs").value = ps.epochs;
    $("lr").value = ps.lr;
    $("scheduler").value = ps.scheduler;
    $("warmup").value = ps.warmup;
    $("precision").value = ps.precision;
    $("packing").checked = ps.packing;
    $("parallelism").value = ps.parallelism;
    $("cpuOffload").checked = ps.cpuOffload;
    $("quantType").value = ps.quantType;
    $("computeDtype").value = ps.computeDtype;
    $("doubleQuant").checked = ps.doubleQuant;
    if (GPUS.find(g => g.id === ps.gpu)) { $("gpu").value = ps.gpu; $("numGpus").value = ps.numGpus; }
    if (announce) {
      const gName = (GPUS.find(g => g.id === ps.gpu) || {}).name || ps.gpu;
      const tail = origin === "gpu"
        ? `(이 GPU 기준 best-case — Full 우선, 안 맞으면 LoRA/QLoRA + GPU 개수 자동. 직접 조정 가능)`
        : `(Full 우선 — 안 맞으면 LoRA/QLoRA로 자동 강등. 아래에서 직접 조정 가능)`;
      $("presetNote").style.display = "";
      $("presetNote").innerHTML = `권장 셋업: <b>${ps.tuning.toUpperCase()}</b> · <b>${gName.split(" (")[0]} ×${ps.numGpus}</b> · LR ${ps.lr} · ${ps.optimizer} · batch ${ps.perDeviceBatch}×accum ${ps.gradAccum} · seq ${ps.seqLen} · ${ps.epochs}ep <span class="dim">${tail}</span>`;
    }
  }

  // Deterministic TRL scaffold from the chosen config (code owns the format).
  function genCommand(model, c, r) {
    const isLora = c.tuning !== "full", isQ = c.tuning === "qlora";
    const trainer = { sft: "SFTTrainer", cpt: "SFTTrainer", dpo: "DPOTrainer", grpo: "GRPOTrainer", gkd: "GKDTrainer" }[c.objective] || "SFTTrainer";
    const cfgName = { sft: "SFTConfig", cpt: "SFTConfig", dpo: "DPOConfig", grpo: "GRPOConfig", gkd: "GKDConfig" }[c.objective] || "SFTConfig";
    const tmods = c.targetModules === "all" ? "all-linear" : (c.targetModules === "attn_all" ? '"q_proj","k_proj","v_proj","o_proj"' : '"q_proj","v_proj"');
    const modelId = model.hf || model.name;
    let s = "";
    s += `# pip install trl peft transformers accelerate${isQ ? " bitsandbytes" : ""}\n`;
    s += `# ${model.name} · ${c.tuning.toUpperCase()} · ${c.objective.toUpperCase()} · ${c.numGpus}x ${(GPUS.find(g=>g.id===$("gpu").value)||{}).name?.split(" (")[0] || "GPU"}\n`;
    s += `import torch\nfrom transformers import AutoModelForCausalLM${isQ ? ", BitsAndBytesConfig" : ""}\n`;
    s += `from trl import ${cfgName}, ${trainer}\n`;
    if (isLora) s += `from peft import LoraConfig\n`;
    s += `\nMODEL = "${modelId}"\n`;
    if (isQ) s += `bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="${c.quantType}",\n    bnb_4bit_use_double_quant=${c.doubleQuant ? "True" : "False"}, bnb_4bit_compute_dtype=torch.${c.computeDtype})\n`;
    s += `model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.${isQ ? c.computeDtype : (c.precision === "bf16" ? "bfloat16" : "float16")}${isQ ? ", quantization_config=bnb" : ""})\n`;
    if (isLora) s += `peft_config = LoraConfig(r=${c.loraR}, lora_alpha=${c.loraAlpha}, lora_dropout=${c.loraDropout},\n    target_modules=${c.targetModules === "all" ? '"all-linear"' : "[" + tmods + "]"}, task_type="CAUSAL_LM")\n`;
    s += `\nargs = ${cfgName}(\n`;
    s += `    per_device_train_batch_size=${c.perDeviceBatch}, gradient_accumulation_steps=${c.gradAccum},\n`;
    s += `    num_train_epochs=${c.epochs}, learning_rate=${c.lr}, lr_scheduler_type="${c.scheduler}", warmup_ratio=${c.warmup},\n`;
    s += `    ${c.precision}=True, gradient_checkpointing=${c.gradCkpt ? "True" : "False"}, optim="${c.optimizer}", max_length=${c.seqLen},\n`;
    if (c.objective === "sft" || c.objective === "cpt") s += `    packing=${c.packing ? "True" : "False"},\n`;
    s += `    output_dir="out", logging_steps=10, save_steps=200)\n`;
    s += `\ntrainer = ${trainer}(model=model, args=args${isLora ? ", peft_config=peft_config" : ""}, train_dataset=ds)  # ds = 데이터셋 로드해서 채우기\ntrainer.train()`;
    if (c.tuning === "full" && c.numGpus > 1) {
      s += `\n\n# 실행 (Full ${c.numGpus}-GPU): ` + (c.parallelism === "deepspeed_zero3"
        ? `deepspeed --num_gpus ${c.numGpus} train.py --deepspeed zero3.json` + (c.cpuOffload ? " (offload_optimizer+param)" : "")
        : `accelerate launch --num_processes ${c.numGpus} --use_fsdp --fsdp_sharding_strategy FULL_SHARD` + (c.cpuOffload ? " --fsdp_offload_params true" : "") + " train.py");
    }
    return s;
  }

  function applyDataset(id) {
    const d = DATASETS.find(x => x.id === id);
    if (!d) { $("datasetNote").style.display = "none"; return; }
    if (d.objective) $("objective").value = d.objective;
    if (d.examples != null) $("examples").value = d.examples;
    if (d.avgTokens != null) $("avgTok").value = d.avgTokens;
    $("datasetNote").style.display = "";
    $("datasetNote").innerHTML = `<b>${esc(d.name)}</b> · ${esc(d.size)} · ${esc(d.format)} · ${esc(d.license)} <a class="ref-tag" href="${d.url}" target="_blank" rel="noopener">HF</a><br>` +
      `<span class="dim">${esc(d.why)} · objective <b>${esc(d.objective)}</b>, 예시 ${Number(d.examples).toLocaleString()} × ~${d.avgTokens}tok 적용됨(추정). 모델을 고르면 best 하드웨어가 잡힙니다.</span>`;
    // surface the matching task recipe in the guide too (card only — don't override dataset's real values)
    if (d.recipe) { const tp = $("taskPicker"); if (tp) tp.value = d.recipe; applyRecipe(d.recipe, false); }
    render();
  }

  function opt(v, t) { const o = document.createElement("option"); o.value = v; o.textContent = t; return o; }
  function radioVal(name) { const el = document.querySelector(`input[name=${name}]:checked`); return el ? el.value : null; }

  function currentModel() {
    const custom = parseFloat($("customParams").value);
    if (!isNaN(custom) && custom > 0) return { id: "custom", name: `custom ${custom}B`, total_params_b: custom, hidden: null, n_layers: null };
    return MODELS.find(m => m.id === $("model").value) || MODELS[0];
  }

  function cfg() {
    const parallelism = $("parallelism").value;
    return {
      tuning: radioVal("tuning"),
      objective: $("objective").value,
      numGpus: +$("numGpus").value,
      perDeviceBatch: +$("batch").value,
      gradAccum: +$("accum").value,
      seqLen: +$("seq").value,
      gradCkpt: $("gradCkpt").checked,
      optimizer: $("optimizer").value,
      loraR: +$("loraR").value,
      loraAlpha: +$("loraAlpha").value,
      loraDropout: +$("loraDropout").value,
      targetModules: $("targetModules").value,
      datasetExamples: +$("examples").value,
      avgTokensPerExample: +$("avgTok").value,
      epochs: +$("epochs").value,
      mfu: +$("mfu").value,
      // training args (feed the generated scaffold; math uses strategy/optimizer/gradCkpt only)
      lr: $("lr").value.trim() || "2e-4",
      scheduler: $("scheduler").value,
      warmup: +$("warmup").value,
      precision: $("precision").value,
      packing: $("packing").checked,
      parallelism: parallelism,
      cpuOffload: $("cpuOffload").checked,
      quantType: $("quantType").value,
      computeDtype: $("computeDtype").value,
      doubleQuant: $("doubleQuant").checked,
      // parallelism -> compute strategy (fsdp/zero3 = full shard; ddp = replicate)
      strategy: parallelism === "ddp" ? "ddp" : "fsdp",
    };
  }

  function seg(gb, colorVar, capGB) {
    const pct = Math.max(0, Math.min(100, (gb / capGB) * 100));
    return `<div class="seg" style="width:${pct}%;background:var(${colorVar})"></div>`;
  }

  function render() {
    const model = currentModel(), gpu = GPUS.find(g => g.id === $("gpu").value) || GPUS[0], c = cfg();
    const r = LLMFT.computeTraining(model, gpu, c);
    const N = c.numGpus;

    // labels
    $("numGpusLabel").textContent = N + "×";
    $("batchLabel").textContent = c.perDeviceBatch;
    $("accumLabel").textContent = c.gradAccum;
    $("seqLabel").textContent = c.seqLen.toLocaleString() + " tok";
    $("rLabel").textContent = "r=" + c.loraR;
    $("exLabel").textContent = c.datasetExamples.toLocaleString();
    $("tokLabel").textContent = c.avgTokensPerExample;
    $("epochLabel").textContent = c.epochs;
    $("mfuLabel").textContent = c.mfu.toFixed(2);
    $("alphaLabel").textContent = "α=" + c.loraAlpha;
    $("dropoutLabel").textContent = c.loraDropout.toFixed(2);
    $("warmupLabel").textContent = c.warmup.toFixed(2);
    $("lrHint").textContent = c.tuning === "full" ? "full 1e-6~1e-5" : "LoRA 1e-4~3e-4";
    // per-tuning config blocks: each method shows ITS full knob set
    $("fullInputs").style.display = c.tuning === "full" ? "" : "none";
    $("loraInputs").style.display = c.tuning === "full" ? "none" : "";
    $("qloraInputs").style.display = c.tuning === "qlora" ? "" : "none";
    $("loraHelperBlock").style.display = c.tuning === "full" ? "none" : "";
    $("packingWrap").style.display = (c.objective === "sft" || c.objective === "cpt") ? "" : "none";

    // chips
    $("modelChips").innerHTML =
      `<span class="chip">${esc(model.name)}</span><span class="chip">${fmt(model.total_params_b, 0)}B params</span>` +
      (model.n_layers ? `<span class="chip">${model.n_layers}L · h${model.hidden}</span>` : `<span class="chip">arch 미상 → LoRA 근사</span>`) +
      `<span class="chip">${c.tuning.toUpperCase()}</span><span class="chip">${gpu.name.split(" (")[0]} ×${N}</span>`;

    // objective note
    const obj = KB.objectives.find(o => o.id === c.objective);
    if (obj) $("objectiveNote").innerHTML =
      `${esc(obj.goal)} · 포맷 <b>${esc(obj.dataFormat)}</b> · <a href="${obj.source}" target="_blank" rel="noopener">${esc(obj.tool)}</a>`;

    // VRAM bar
    const cap = gpu.vram_gb;
    $("vramBar").innerHTML = seg(r.baseGB, "--w", cap) + seg(r.trainStateGB, "--k", cap) + seg(r.activationsGB + r.overheadGB, "--o", cap);
    $("fitBadge").innerHTML = r.fits
      ? `<span class="badge ok">✅ ${gpu.name.split(" (")[0]} ${N}개에 들어감</span>`
      : `<span class="badge no">⚠️ 안 들어감 (OOM)</span>`;
    $("vramNums").innerHTML =
      `base <b>${fmt(r.baseGB)}GB</b> + 학습상태 <b>${fmt(r.trainStateGB)}GB</b> + 액티베이션 <b>${fmt(r.activationsGB)}GB</b> + 오버헤드 <b>${fmt(r.overheadGB)}GB</b> ` +
      `= <b>${fmt(r.perGpuVramGB)}GB</b> / ${cap}GB${r.optimizerOffloaded ? " · paged optim (CPU offload)" : ""}` +
      (r.fits ? ` · 여유 ${fmt(r.vramHeadroomGB)}GB` : "");
    $("vramFormula").innerHTML =
      `<code>base = ${fmt(model.total_params_b,0)}B × ${r.baseBytes} byte = ${fmt(r.baseGB)}GB</code><br>` +
      `<code>학습상태 = ${fmt(r.trainableParams/1e6)}M trainable × ${r.trainPerParam} byte (grad2 + optim + master4) = ${fmt(r.trainStateGB)}GB</code><br>` +
      `<code>액티베이션 ≈ batch×seq×hidden×layers×16 ${c.gradCkpt ? "× 0.22(checkpointing)" : ""} = ${fmt(r.activationsGB)}GB <span class="dim">(heuristic)</span></code><br>` +
      `<code>${r.strategy.toUpperCase()}: ${r.strategy === "fsdp" ? `(base+상태) ÷ ${N} GPU로 샤딩` : "DDP는 GPU마다 복제"}</code><br>` +
      `<span class="dim">출처: EleutherAI Transformer Math (16 B/param), QLoRA arXiv:2305.14314 (NF4). 액티베이션 계수는 heuristic.</span>`;
    if (!r.fits) {
      $("vramNums").innerHTML += `<div class="verdict no" style="margin-top:10px"><b>OOM — 이렇게 줄이세요:</b><ul class="fixlist">` +
        r.oomFixes.map(f => `<li>${esc(f)}</li>`).join("") + `</ul></div>`;
    }

    // max model + effective batch
    $("maxModel").innerHTML = r.maxModelParamsB
      ? `<b>${fmt(r.maxModelParamsB, 0)}B</b> <span class="dim">이 ${N}×${gpu.name.split(" (")[0]}·${c.tuning.toUpperCase()}·batch ${c.perDeviceBatch}·seq ${c.seqLen}에서 최대</span>`
      : "—";
    $("effBatch").innerHTML =
      `<b>${r.effectiveBatch}</b> <span class="dim">= per-device ${c.perDeviceBatch} × accum ${c.gradAccum} × GPU ${N}</span>` +
      (c.gradAccum > 1 ? `<div class="dim" style="margin-top:4px">⚠️ grad accum은 실제 대형 배치와 완전히 같지 않습니다(작은 micro-batch에서 차이).</div>` : "");

    // time / cost
    if (r.timeAvailable) {
      $("timeBox").innerHTML =
        `<div class="cost-row"><span>학습 시간 (추정)</span><b>${fmt(r.trainHours)} h</b> <span class="dim">= ${fmt(r.trainHours/24,1)}일</span></div>` +
        `<div class="cost-row"><span>GPU-시간</span><b>${fmt(r.gpuHours)}</b></div>` +
        (r.costUSD != null ? `<div class="cost-row"><span>임대 비용 (@$${gpu.rent_usd_hr}/hr)</span><b>$${fmt(r.costUSD, 0)}</b></div>` : "") +
        `<div class="verdict" style="background:color-mix(in srgb,var(--accent) 10%, transparent)">토큰 <b>${fmt(r.trainTokens/1e6,1)}M</b> (예시 ${c.datasetExamples.toLocaleString()} × ${c.avgTokensPerExample}tok × ${c.epochs}ep) · MFU ${c.mfu} 가정. <b>MFU가 시간을 좌우</b>하니 범위로 보세요.</div>`;
      $("timeFormula").innerHTML =
        `<code>C = 6 × N × D = 6 × ${fmt(model.total_params_b,0)}B × ${fmt(r.trainTokens/1e6,1)}M tok = ${(r.trainFlops).toExponential(2)} FLOP</code><br>` +
        `<code>시간 = C ÷ (${gpu.flops_bf16_tf} TFLOPS × ${N} GPU × MFU ${c.mfu})${c.gradCkpt ? " × 1.3(checkpointing)" : ""} = ${fmt(r.trainHours)} h</code><br>` +
        `<span class="dim">출처: Kaplan et al. C≈6ND (arXiv:2001.08361). FLOPS는 dense bf16(sparsity off). LoRA도 base fwd/bwd가 지배해 시간은 full과 유사.</span>`;
    } else {
      $("timeBox").innerHTML = `<div class="dim">이 하드웨어는 방어 가능한 dense-bf16 학습 FLOPS 스펙이 없어(사전예약/NPU/Apple) <b>시간 추정을 생략</b>합니다. VRAM·적합성만 참고하세요. (추측보다 공백이 정직합니다.)</div>`;
      $("timeFormula").innerHTML = `<span class="dim">시간 = 6ND ÷ (FLOPS × MFU) — FLOPS 미상이라 계산 안 함.</span>`;
    }

    // LoRA helper
    if (r.lora) {
      $("loraHelper").innerHTML =
        `<div class="cost-row"><span>학습 파라미터</span><b>${fmt(r.lora.trainableParams/1e6)}M</b> <span class="dim">(${fmt(r.lora.trainablePct,2)}% of ${fmt(model.total_params_b,0)}B)</span></div>` +
        `<div class="cost-row"><span>권장 alpha</span><b>${r.lora.recommendedAlpha}</b> <span class="dim">(α≈2r 규칙 — 스케일 α/r 고정)</span></div>` +
        `<div class="cost-row"><span>target</span><b>${r.lora.targetModules}</b> <span class="dim">all-linear가 품질↑ (Raschka)</span></div>` +
        `<div class="dim" style="margin-top:6px">${r.trainableSource}. LR은 LoRA ~1e-4–3e-4, full ~10x 낮게. <a href="https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms" target="_blank" rel="noopener">근거</a></div>`;
    }

    // warnings (overfit)
    const warns = r.overfitFlags;
    $("warnBlock").style.display = warns.length ? "" : "none";
    if (warns.length) $("warns").innerHTML =
      `<div class="block-title">주의 신호</div>` + warns.map(w => `<div class="verdict no" style="margin:6px 0">⚠️ ${esc(w)}</div>`).join("");

    // optimizer steps (needs no FLOPS spec) — appended to the effective-batch box
    if (c.datasetExamples > 0) {
      const steps = Math.ceil((c.datasetExamples * c.epochs) / Math.max(1, r.effectiveBatch));
      $("effBatch").innerHTML += `<div class="dim" style="margin-top:4px">≈ <b>${fmt(steps, 0)}</b> optimizer step (예시 ${c.datasetExamples.toLocaleString()} × ${c.epochs}ep ÷ eff.batch ${r.effectiveBatch})</div>`;
    }

    // runnable TRL scaffold reflecting every chosen knob
    $("runCmd").textContent = genCommand(model, c, r);
  }

  function renderReference() {
    $("refMethods").innerHTML = KB.objectives.map(o =>
      `<div class="ref-row"><span class="ref-name"><b>${esc(o.name)}</b><br><span class="dim">${esc(o.goal)}</span></span>` +
      `<span class="ref-price">${esc(o.dataFormat)}<br><a class="ref-tag" href="${o.source}" target="_blank" rel="noopener">docs</a></span></div>`).join("");
    $("refVolume").innerHTML = KB.dataVolume.map(v =>
      `<div class="ref-row"><span class="ref-name">${esc(v.goal)} <span class="ref-tag">${esc(v.note)}</span></span>` +
      `<span class="ref-price">${esc(v.range)}${v.source ? `<br><a class="ref-tag" href="${v.source}" target="_blank" rel="noopener">src</a>` : ""}</span></div>`).join("");
    $("refDistill").innerHTML = KB.distillation.map(d =>
      `<div class="ref-row"><span class="ref-name"><b>${esc(d.name)}</b><br><span class="dim">${esc(d.how)}</span></span>` +
      `<span class="ref-price">${esc(d.produces)}<br><a class="ref-tag" href="${d.source}" target="_blank" rel="noopener">src</a></span></div>`).join("") +
      `<div class="dim" style="margin-top:8px">${esc(KB._tos_warning)}</div>`;
    $("refTech").innerHTML =
      `<div class="dim" style="margin-bottom:6px">확립 코어(기본 추천): ${KB ? "" : ""}${TECH.established.map(e => `<a href="${e.url}" target="_blank" rel="noopener">${esc(e.name)}</a>`).join(" · ")}</div>` +
      TECH.emerging_2026.map(t =>
        `<div class="ref-row"><span class="ref-name"><b>${esc(t.name)}</b> <span class="ref-tag">${esc(t.date)} · ${esc(t.maturity)}</span><br><span class="dim">${esc(t.what)} — ${esc(t.when)}</span></span>` +
        `<span class="ref-price"><a class="ref-tag" href="${t.url}" target="_blank" rel="noopener">src</a></span></div>`).join("");
  }

  function renderGuide() {
    if (!RECIPES) return;
    const ms = RECIPES.marketShift;
    $("marketShift").innerHTML = `<b>${esc(ms.headline)}</b><div class="ref-grid" style="margin-top:10px">` +
      ms.tracks.map(t => `<div class="ref-card"><h3 style="font-size:13px">${esc(t.name)}</h3><div class="dim" style="font-size:12px;line-height:1.5">${esc(t.what)} ${t.src ? `<a class="ref-tag" href="${t.src}" target="_blank" rel="noopener">src</a>` : ""}</div></div>`).join("") + `</div>`;
    const g = RECIPES.whenToFineTune;
    $("ftGate").innerHTML = `<div class="block-title">파인튜닝이 확실히 이기는 5조건</div><ul class="fixlist">` +
      g.winConditions.map(c => `<li><b>${esc(c.cond)}</b> — ${esc(c.detail)} ${c.src ? `<a class="ref-tag" href="${c.src}" target="_blank" rel="noopener">src</a>` : ""}</li>`).join("") + `</ul>` +
      `<div class="verdict no" style="margin-top:8px">⚠️ 먼저 RAG를 의심하라: ${esc(g.firstAskRAG)}</div>` +
      `<div class="verdict" style="margin-top:6px;background:color-mix(in srgb,var(--accent) 10%, transparent)">양보할 영역: ${esc(g.deferToFrontier)}</div>`;
    $("taskPicker").innerHTML = `<option value="">— 작업을 고르면 레시피 + 계산기 입력이 채워집니다 —</option>` +
      RECIPES.recipes.map(r => `<option value="${r.id}">${esc(r.task)}</option>`).join("");
  }

  function applyRecipe(id, setInputs) {
    const r = RECIPES.recipes.find(x => x.id === id);
    if (!r) { $("recipeDetail").innerHTML = ""; return; }
    // setInputs=true (guide task picker, no dataset): anchor the calculator to the recipe.
    // setInputs=false (dataset-driven): only render the card — dataset owns data, model preset owns config.
    if (setInputs !== false) {
      if (r.objective) $("objective").value = r.objective;
      if (r.tuning) setRadio("tuning", r.tuning);
      if (r.loraR) $("loraR").value = r.loraR;
      if (r.epochs) $("epochs").value = r.epochs;
      const anchor = { cpt_repo: 30000, narrow: 600, domain: 20000, style: 350, reasoning: 1500, instruction: 5000 };
      if (anchor[r.dataKey] != null) $("examples").value = anchor[r.dataKey];
    }
    const objName = (KB.objectives.find(o => o.id === r.objective) || {}).name || r.objective;
    const thenName = r.objectiveThen ? (KB.objectives.find(o => o.id === r.objectiveThen) || {}).name || r.objectiveThen : "";
    $("recipeDetail").innerHTML =
      `<div class="block-title">${esc(r.task)}</div>` +
      `<div class="cost-row"><span>방법</span><b>${esc(objName)}${thenName ? " → " + esc(thenName) : ""} · ${r.tuning.toUpperCase()} (r=${r.loraR})</b></div>` +
      `<div class="cost-row"><span>데이터</span><b>${esc(r.dataRange)}</b></div>` +
      `<div class="cost-row"><span>베이스 모델</span><b>${esc(r.base)}</b></div>` +
      `<div class="cost-row"><span>판정</span><b>${esc(r.verdict)}</b></div>` +
      `<div class="verdict no" style="margin-top:8px">정직하게: ${esc(r.honesty)}</div>` +
      `<div class="dim" style="margin-top:6px">${esc(r.note)} · 계산기 입력(방법·데이터량)을 이 레시피로 채웠습니다. GPU·모델을 골라 VRAM·시간을 확인하세요.</div>`;
    render();
  }

  async function boot() {
    try {
      const [models, gpus, kb, tech, recipes, datasets] = await Promise.all([
        fetch("data/models.json").then(r => r.json()),
        fetch("data/gpus.json").then(r => r.json()),
        fetch("data/methods.json").then(r => r.json()),
        fetch("data/techniques-2026.json").then(r => r.json()),
        fetch("data/recipes.json").then(r => r.json()),
        fetch("data/datasets.json").then(r => r.json()),
      ]);
      MODELS = models.models; GPUS = gpus.gpus; KB = kb; TECH = tech; RECIPES = recipes; DATASETS = datasets.datasets;
      $("dataset").appendChild(opt("", "— 직접 설정 (데이터셋 선택 안 함) —"));
      DATASETS.forEach(d => $("dataset").appendChild(opt(d.id, `${d.name} · ${d.size}`)));
      MODELS.forEach(m => $("model").appendChild(opt(m.id, `${m.name} · ${fmt(m.total_params_b,0)}B`)));
      GPUS.forEach(g => $("gpu").appendChild(opt(g.id, `${g.name}${g.flops_bf16_tf == null ? " (시간 추정 불가)" : ""}`)));
      KB.objectives.forEach(o => $("objective").appendChild(opt(o.id, o.name)));
      if (MODELS.find(m => m.id === "qwen3-8b")) $("model").value = "qwen3-8b"; // friendly default (else best-GPU for 753B looks odd)
      $("gpu").value = "rtx4090";
      document.querySelectorAll("select, input").forEach(el => { el.addEventListener("input", render); el.addEventListener("change", render); });
      // model change -> auto-apply that model's recommended preset (if toggle on), then render
      const onModelChange = () => { if ($("autoPreset").checked) applyPreset(currentModel(), true); render(); };
      $("model").addEventListener("change", onModelChange);
      $("customParams").addEventListener("input", () => { if ($("autoPreset").checked) applyPreset(currentModel(), true); render(); });
      // GPU change -> recompute best case FOR THAT GPU (tuning + count), unless autoPreset is off
      $("gpu").addEventListener("change", () => { if ($("autoPreset").checked) applyPresetObj(recommendSetupForGpu(currentModel(), $("gpu").value), true, "gpu"); render(); });
      $("copyCmd").addEventListener("click", () => {
        const t = $("runCmd").textContent;
        navigator.clipboard && navigator.clipboard.writeText(t);
        $("copyCmd").textContent = "복사됨 ✓"; setTimeout(() => $("copyCmd").textContent = "복사", 1500);
      });
      $("dataset").addEventListener("change", () => applyDataset($("dataset").value));
      $("taskPicker").addEventListener("change", () => applyRecipe($("taskPicker").value, true));
      renderReference();
      renderGuide();
      applyPreset(currentModel(), true); // apply preset for the initial model on load
      render();
    } catch (e) {
      $("app").innerHTML = `<div class="err">데이터를 불러오지 못했습니다. 로컬에서는 <code>python3 -m http.server</code>로 여세요 (file://는 fetch가 막힙니다). GitHub Pages에서는 정상 동작합니다.<br><span class="dim">${esc(e.message || e)}</span></div>`;
    }
  }
  boot();
})();
