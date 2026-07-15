/* finetune-method-picker — client-side UI glue. No backend, no keys, no data leaves the browser.
   Pure estimation core lives in compute.js (LLMFT.computeTraining), shared with the Node unit tests. */
(function () {
  const $ = id => document.getElementById(id);
  const fmt = (x, d = 1) => (x == null || isNaN(x) ? "—" : Number(x).toLocaleString("en-US", { maximumFractionDigits: d }));
  const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  let MODELS = [], GPUS = [], KB = null, TECH = null;

  function opt(v, t) { const o = document.createElement("option"); o.value = v; o.textContent = t; return o; }
  function radioVal(name) { const el = document.querySelector(`input[name=${name}]:checked`); return el ? el.value : null; }

  function currentModel() {
    const custom = parseFloat($("customParams").value);
    if (!isNaN(custom) && custom > 0) return { id: "custom", name: `custom ${custom}B`, total_params_b: custom, hidden: null, n_layers: null };
    return MODELS.find(m => m.id === $("model").value) || MODELS[0];
  }

  function cfg() {
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
      targetModules: $("targetModules").value,
      datasetExamples: +$("examples").value,
      avgTokensPerExample: +$("avgTok").value,
      epochs: +$("epochs").value,
      mfu: +$("mfu").value,
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
    $("loraInputs").style.display = c.tuning === "full" ? "none" : "";
    $("loraHelperBlock").style.display = c.tuning === "full" ? "none" : "";

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

  async function boot() {
    try {
      const [models, gpus, kb, tech] = await Promise.all([
        fetch("data/models.json").then(r => r.json()),
        fetch("data/gpus.json").then(r => r.json()),
        fetch("data/methods.json").then(r => r.json()),
        fetch("data/techniques-2026.json").then(r => r.json()),
      ]);
      MODELS = models.models; GPUS = gpus.gpus; KB = kb; TECH = tech;
      MODELS.forEach(m => $("model").appendChild(opt(m.id, `${m.name} · ${fmt(m.total_params_b,0)}B`)));
      GPUS.forEach(g => $("gpu").appendChild(opt(g.id, `${g.name}${g.flops_bf16_tf == null ? " (시간 추정 불가)" : ""}`)));
      KB.objectives.forEach(o => $("objective").appendChild(opt(o.id, o.name)));
      $("gpu").value = "rtx4090";
      document.querySelectorAll("select, input").forEach(el => { el.addEventListener("input", render); el.addEventListener("change", render); });
      renderReference();
      render();
    } catch (e) {
      $("app").innerHTML = `<div class="err">데이터를 불러오지 못했습니다. 로컬에서는 <code>python3 -m http.server</code>로 여세요 (file://는 fetch가 막힙니다). GitHub Pages에서는 정상 동작합니다.<br><span class="dim">${esc(e.message || e)}</span></div>`;
    }
  }
  boot();
})();
