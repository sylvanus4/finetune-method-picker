# Security / 보안

**EN** — `llm-selfhost-calculator` is a **100% client-side static site** (HTML +
CSS + vanilla JS) served on GitHub Pages. It has no backend, no build step, no
package dependencies, and no API keys. Everything runs in your browser. Security
here is a property of the architecture, not a promise in prose.

**KO** — `llm-selfhost-calculator`는 GitHub Pages로 서빙되는 **100% 클라이언트
사이드 정적 사이트**(HTML + CSS + 순수 JS)입니다. 백엔드도, 빌드 단계도, 패키지
의존성도, API 키도 없습니다. 모든 계산은 사용자의 브라우저 안에서 실행됩니다.
보안은 산문으로 하는 약속이 아니라 구조 자체의 속성입니다.

## What this tool does — and does NOT do / 동작과 비동작

| | EN | KO |
|---|---|---|
| ✅ Runs fully **in your browser** | All math runs client-side in vanilla JS. | 모든 계산이 순수 JS로 브라우저에서 실행됩니다. |
| ✅ **No data egress** | Nothing you enter is transmitted. No telemetry, no analytics, no phone-home, no tracking cookies. | 입력값은 어디로도 전송되지 않습니다. 텔레메트리·애널리틱스·트래킹 쿠키가 없습니다. |
| ✅ **No secrets, no accounts** | No API key, no login, no server call is ever made. | API 키·로그인·서버 호출이 전혀 없습니다. |
| ✅ **Reads only local data** | It fetches static JSON (GPU/model/price tables) from the same GitHub Pages origin — nothing else. | 같은 오리진의 정적 JSON(모델·GPU·가격 표)만 읽고 그 외에는 아무것도 읽지 않습니다. |
| 🚫 No backend / database | There is no server to compromise and no stored user data. | 침해할 서버도, 저장된 사용자 데이터도 없습니다. |
| 🚫 No dependencies / build | No npm, no pip, no bundler. Nothing to `npm audit`. | npm·pip·번들러가 없습니다. 감사할 의존성 자체가 없습니다. |
| 🚫 No `eval` of input | Inputs are parsed as numbers/strings for arithmetic; data is never executed as code. | 입력은 산술용 숫자/문자열로만 파싱되며, 데이터가 코드로 실행되지 않습니다. |

## Threat surface / 위협 표면

**EN** — With no server, no build pipeline, and no third-party dependencies, the
attack surface is limited to (1) the static files served from this repo and
(2) the browser that runs them. Realistic risks are the same as for any static
page: a malicious commit, or a compromised GitHub Pages account. There are no
secrets to steal and no data to exfiltrate.

**KO** — 서버·빌드 파이프라인·서드파티 의존성이 없으므로 공격 표면은 (1) 이
저장소가 서빙하는 정적 파일과 (2) 그것을 실행하는 브라우저로 한정됩니다. 현실적인
위험은 여느 정적 페이지와 동일합니다 — 저장소에 대한 악의적 커밋, 또는 GitHub
Pages 계정 탈취. 훔칠 비밀도, 유출할 데이터도 없습니다.

## Data & privacy / 데이터와 개인정보

**EN** — Nothing leaves the browser. The calculator does not collect, store, or
transmit any input. Reloading discards your inputs (any URL/localStorage use is
purely client-side, never sent anywhere).

**KO** — 어떤 것도 브라우저를 벗어나지 않습니다. 입력값을 수집·저장·전송하지
않습니다. 새로고침하면 입력이 사라지며, URL/localStorage를 쓰더라도 전적으로
클라이언트 측이고 외부로 전송되지 않습니다.

## Accuracy of figures / 수치의 정확성

**EN** — Model sizes, GPU specs, throughput, and prices are **estimates** for
planning and may be outdated or approximate. Do not treat the output as a quote or
guarantee. Verify against vendor pricing before any purchasing or capacity
decision. Inaccurate data is a correctness issue, not a security vulnerability —
report it as a normal issue.

**KO** — 모델 크기·GPU 사양·처리량·가격은 계획용 **추정치**이며 오래되었거나 근사값일
수 있습니다. 결과를 견적이나 보장으로 여기지 마세요. 구매·용량 결정 전 공급사 가격으로
검증하세요. 데이터 부정확은 보안 취약점이 아니라 정확성 문제이므로 일반 이슈로 신고해
주세요.

## Reporting a vulnerability / 취약점 신고

**EN** — Found a genuine security issue (e.g. an XSS vector in how inputs render,
or a malicious data file)?

- Preferred: open a **private GitHub Security Advisory** (`Security` tab →
  `Report a vulnerability`).
- Or open a normal **GitHub issue** — but do **not** include anything sensitive.
- This is a solo, best-effort project. There is no SLA; expect a reply when the
  maintainer is available.

**KO** — 실제 보안 문제(예: 입력 렌더링의 XSS 경로, 악성 데이터 파일 등)를 발견하셨나요?

- 권장: **비공개 GitHub Security Advisory**로 신고 (`Security` 탭 →
  `Report a vulnerability`).
- 또는 일반 **GitHub 이슈**로 신고 — 단 민감한 정보는 포함하지 마세요.
- 1인 베스트에포트 프로젝트입니다. SLA는 없으며 메인테이너가 가능할 때 답변합니다.

## Out of scope / 범위 밖

**EN** — Inaccurate/outdated numbers (→ normal issue); GitHub Pages / GitHub
platform itself (→ report to GitHub); your own browser/OS/network; feature
requests and UI bugs (→ normal issue).

**KO** — 부정확·오래된 수치(→ 일반 이슈); GitHub Pages / GitHub 플랫폼 자체(→ GitHub에
신고); 사용자 본인의 브라우저·OS·네트워크; 기능 요청 및 UI 버그(→ 일반 이슈).

## Supported versions / 지원 버전

**EN** — A single always-latest static site. Only the currently deployed `main`
branch (GitHub Pages) is supported. There are no back-ported releases.

**KO** — 항상 최신 상태인 단일 정적 사이트입니다. `main` 브랜치의 현재 배포본만
지원합니다. 백포트 릴리스는 없습니다.

| Version | Supported |
|---|---|
| `main` (deployed) | ✅ |
| older commits | ❌ |

> Dependabot: **N/A** — there are no dependency manifests to scan. This will be
> revisited if the project ever adds dependencies.
