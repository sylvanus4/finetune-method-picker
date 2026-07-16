# finetune-method-picker

**[English](README.en.md)** · 한국어 · 웹 UI는 우측 상단에서 EN/KR 전환

**내 GPU로 이 모델을 어떻게 파인튜닝하지?** — GPU·모델·기법을 넣으면 학습 VRAM, 최대 학습 가능 모델 크기, 데이터 포맷, 데이터량, 학습 시간·비용을 **공식과 함께** 계산하는 정적 웹 도구.

100% 브라우저에서 동작합니다. 서버·API 키·데이터 전송이 전혀 없어요. 계산 로직은 전부 클라이언트 JavaScript고, **숫자마다 "공식 보기"로 근거와 대입값을 펼칠 수 있습니다.** MIT.

> 공개 스펙·공식 기반 **계획용 추정**입니다. 실측 벤치를 대신하지 않아요 — 학습 시간은 MFU에 크게 좌우되니 범위로 보세요. 안 들어가면(OOM) "이렇게 줄이세요"를 그대로 띄웁니다. 그래야 도구를 믿을 수 있으니까요.

## 무엇을 답해주나

- **데이터셋 + 모델만 고르면 best 자동** — 실제 HF 데이터셋(2026 의미있는 케이스)을 고르면 objective·예시수·평균토큰이 채워지고, 모델을 고르면 **Full 우선**으로 맞는 최소 하드웨어+설정이 자동 셋팅(안 맞으면 LoRA→8-bit→QLoRA 자동 강등). GPU를 바꾸면 그 카드 기준으로 재계산.
- **OOM 디버거** — batch / seq_len / grad accum / gradient checkpointing / optimizer를 돌리며 GPU당 VRAM을 실시간으로 보고, 안 들어가면 구체적 처방을 제시
- **튜닝 방식별 완전 config** — Full(FSDP/ZeRO-3/DDP·CPU offload) · LoRA(r/alpha/dropout/target) · 8-bit LoRA(int8 base) · QLoRA(NF4/double-quant/compute dtype) + LR·스케줄러·warmup·packing
- **실행 가능한 TRL 스캐폴드** — 선택값을 채운 복붙 Python 코드(objective별 trainer + LoraConfig/BitsAndBytesConfig + FSDP/DeepSpeed 런치)
- **2026 태스크 가이드** — "회사 UI 코드 학습" 같은 작업별 방법·데이터량·베이스 모델 + **파인튜닝을 애초에 할 가치가 있나** 게이트(3갈래 시장·5조건·먼저 RAG 의심)
- **최대 학습 가능 모델** / **effective batch** = per-device × grad accum × GPU 수 / optimizer step 수
- **LoRA 도우미** — rank r → 학습 파라미터 수(GQA/MLA-aware), α≈2r, target modules
- **학습 시간·비용** — C≈6ND ÷ (FLOPS × MFU), 임대 비용
- **EN/KR 이중언어** — 우측 상단 토글로 화면 전체가 실제 언어로 렌더

## 공식 (전부 화면 "공식 보기"에서 확인 가능)

- **학습 VRAM** = base(파라미터×2, QLoRA는 ×0.5 NF4) + 학습상태(trainable × [grad 2 + optim + master 4]) + 액티베이션 + 오버헤드. FSDP는 base+상태를 GPU에 샤딩. — EleutherAI Transformer Math, QLoRA(arXiv:2305.14314)
- **LoRA 학습 파라미터** ≈ 2·r·Σ(in+out) over targeted layers — LoRA(arXiv:2106.09685)
- **학습 시간** = 6·N·D ÷ (dense-bf16 FLOPS × GPU수 × MFU) — Kaplan(arXiv:2001.08361)
- FLOPS는 **dense bf16(sparsity off)** 기준. 방어 가능한 스펙이 없는 하드웨어(사전예약/NPU/Apple)는 시간 추정을 **생략**합니다(추측보다 공백이 정직).

## 로컬 실행

```bash
python3 -m http.server   # file://는 fetch가 막히므로 http로 열기
# http://localhost:8000
```

## 검증

```bash
node test/compute.test.cjs                              # 결정론 유닛 게이트
node audit/js_dump.cjs > /tmp/js.json && python3 audit/reference_audit.py /tmp/js.json  # JS↔Python 파리티
```
CI(`.github/workflows/validate.yml`)가 push마다 위 둘 + JSON 검증을 돌립니다.

## 데이터

- `data/gpus.json` — 가속기 스펙(VRAM·대역폭·dense bf16 FLOPS·가격·전력), 공개 스펙 2026-07
- `data/models.json` — 최신 오픈웨이트 모델(파라미터·층·hidden)
- `data/methods.json` — 목표·튜닝·데이터 포맷·데이터량·증류, 전부 출처 URL
- `data/techniques-2026.json` — 2026 최근 기법 셸프(성숙도 라벨)

## 왜 공익(광고 아님)

파인튜닝 기법·메모리 계산은 흩어진 벤치마크와 커뮤니티 스레드에서 파편적으로만 얻을 수 있어, 답을 아는 사람만 정답을 압니다. 계산 로직과 공식·출처를 투명하게 공개해 누구나 자기 GPU로 검증하게 합니다. 특정 제품·서비스를 팔지 않습니다.

MIT License.
