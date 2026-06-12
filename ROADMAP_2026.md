# faceswap_pro v2 — SOTA Analysis & Feature Roadmap

**Date:** 2026-06-11
**Author:** session synthesis
**Scope:** competitive landscape (OSS + commercial), gap analysis, prioritized roadmap

---

## 1. v2's current production surface (from code, not memory)

**Tabs shipped (8):** Lip-Sync · Face Swap · Webcam · Rotoscoping · Queue · History · Presets · Cache.

**Engine stack:**

| Domain | Component | License posture |
|---|---|---|
| Face swap | `inswapper_128.onnx` (InsightFace) | community weights, commercial gray |
| Face detection / 5-pt landmarks | InsightFace `buffalo_l` (FaceAnalysis) | community weights |
| Face enhance | GFPGAN 1.4 (subprocess worker) | Apache 2.0 |
| Lip-sync | LatentSync 1.6 + KeySync option | Apache 2.0 / Apache 2.0 |
| Lip-sync per-clip fine-tune | `lipsync_finetune` LoRA UNet adapter | derivative of LatentSync |
| Rotoscoping | SAM2.1-hiera-base-plus + daemon | Apache 2.0 |
| Occlusion mask gate | XSeg (DeepFaceLab) | open |
| Voice clone | RVC v2 (subprocess) | open |
| Vocal isolation | Demucs (subprocess) | MIT |
| Identity blend | embedding-journey across ArcFace 512-d space | own code |
| Background restyle | ~~ripped out today~~ (Flux Schnell weights ~35 GB still on disk) | — |
| Creature swap | ~~ripped out today~~ | — |

**Unique strengths vs the OSS field:**
- SAM2 multi-object rotoscope (most face-swap apps don't ship this)
- Per-clip LatentSync fine-tune (one-shot identity adapter)
- Voice clone + vocal isolation in the same app (most apps stop at lipsync)
- Batch queue with cancel/resume
- Watermarking + history with timecoded previews
- Webcam tab with virtual-cam output
- Snapshot system (`_snapshots/`) for recovering from bad edits
- Public-symbol + tail-integrity checkers (added today)

**Known production blockers / partial work:**
- One pending task: rotoscope `gr.Video` Error overlay (#129) — transcode source to browser-safe preview
- One pending task: user-verify lipsync fine-tune after path fix (#34)
- KeySync stripped from release tree but kept in v2 working tree (intentional, per #88)

---

## 2. SOTA landscape — what shipped in 2025–2026

### Open-source

| Tool | What it does | 2026 status | Distinct strength |
|---|---|---|---|
| **FaceFusion 3.6** | Face swap + lip-sync + enhance + colorize | Active leader, "industry standard" branding | Pixel Boost, multi-face, LivePortrait integration, frame colorizer |
| **Roop / Roop-Unleashed** | Face swap | **Roop archived March 2026** (developer cited misuse concerns); community moved to Roop-Unleashed fork with 256px output + restoration | Simple one-click swap, mature |
| **ReActor** | Face swap as Stable Diffusion / Forge extension | Active; community workflow plugin | Integration into SD workflows |
| **DeepFaceLab / DeepFaceLive** | High-quality face swap via custom-trained per-identity models | Active for advanced users | Highest possible quality if you train, but high effort |
| **SimSwap / FaceDancer** | Identity-conditioned end-to-end swap nets | Research benchmarks; **best temporal stability in recent comparisons** | Smoother frame-to-frame than inswapper |
| **LivePortrait** (Kuaishou) | Driving-video animation of a still portrait | Open, often paired with MuseTalk | Real-time facial expression transfer from a driver |
| **MuseTalk** (Tencent) | Real-time lip-sync from audio onto a portrait | Open | Genuinely real-time on consumer GPU |
| **Hallo3** (Fudan) | Diffusion-Transformer talking head from image + audio | Open, CVPR 2025 | Highly dynamic motion, supports wild scenes |
| **EMO** (Alibaba) | Audio-driven expressive talking-head | Open | Strongest emotional expressiveness |
| **DynamicFace** (CVPR 2025) | Face swap with 3D facial priors | Research code | Strong consistency on aggressive pose changes |
| **VFace** (2026) | Training-free diffusion-based video face swap | New | No fine-tune required, leverages base diffusion |
| **InstantID / IP-Adapter FaceID** | Identity injection into diffusion outputs | Active | Used as building block for stylized swaps |

### Commercial / closed

| Tool | Best at | Pricing model |
|---|---|---|
| **Pika** — Pikaswaps + Pikaformance + Pikaffects | Quick social-format face/character swap + lip-synced talking image | Per-credit / subscription |
| **Kling 3.0 Omni** | Text-instructed edits, 1080p 48fps with synced audio + lipsync | Subscription |
| **Runway Gen-4 + Aleph** | Director-mode editing, motion brushes, commercial creative workflows | Subscription |
| **Veo 3 (Google)** | Text-to-video with native audio, strong identity consistency | API + Vertex |
| **Sora 2 (OpenAI)** | Long-form coherence, strongest world model | Plus / Pro |
| **Hailuo / MiniMax** | Cheap, fast, decent quality | Subscription |

---

## 3. Feature comparison matrix (v2 vs the field)

Legend: **✓** = solid · ◐ = partial / dated · **✗** = absent · n/a = out of scope

| Capability | **v2 today** | **FaceFusion 3.6** | **DeepFaceLab** | **Pika / Kling** |
|---|---|---|---|---|
| One-click face swap | ✓ inswapper_128 | ✓ multiple models | ✗ (training-required) | ✓ |
| Multi-face / face picker | ✓ (Chunk B) | ✓ multiple | ✗ | ✓ |
| Live webcam face swap | ✓ + virtual cam | ✓ | ✓ (DFL Live) | ✗ |
| Per-clip identity fine-tune | ✓ LoRA UNet (unique) | ✗ | ✓ (training) | ✗ |
| **Diffusion-based face swap (temporal-stable)** | **✗** | ✗ | ✗ | ✓ |
| **Talking head from still image** | **✗** | ✓ via LivePortrait | ✗ | ✓ Pikaformance |
| Audio-driven lipsync onto video | ✓ LatentSync 1.6 | ✓ | ◐ | ✓ |
| Voice clone (TTS / RVC) | ✓ RVC | ✗ | ✗ | ✗ |
| Vocal isolation | ✓ Demucs | ✗ | ✗ | ✗ |
| Multi-object rotoscope (SAM2) | ✓ (unique) | ✗ | ✗ | ✗ |
| Background restyle / scene replace | ✗ (ripped today) | ◐ | ✗ | ✓ |
| **Pose-aware swap on extreme angles** | ◐ inswapper limits | ◐ same limits | ✓ if trained | ✓ |
| **Identity preservation across long video** | ◐ | ◐ | ✓ | ✓ |
| Frame colorization (B&W → color) | ✗ | ✓ | ✗ | ✓ |
| Hardware-only / private / offline | ✓ | ✓ | ✓ | ✗ |
| Batch queue + history | ✓ | ◐ | ✗ | ✗ |
| Watermarking | ✓ | ✗ | ✗ | ✓ varies |
| Aspect-ratio re-frame | ✓ | ◐ | ✗ | ✓ |
| Browser-based output | ✗ desktop only | ✓ | ✗ | ✓ |
| Self-checked codebase (tail/symbol) | ✓ (added today) | ✗ | ✗ | n/a |

**Honest read:** v2 is ahead of FaceFusion 3.6 in three specific places (per-clip LoRA, SAM2 rotoscope, voice/audio stack) and behind it in two (talking head from still image, frame colorization). It is *categorically* behind Pika/Kling on diffusion-quality temporal stability and on multi-modal compositing — but those are not realistic catch-up fights as a solo OSS effort.

---

## 4. Where v2 should actually invest — strategic frame

Three honest positioning options. Pick one consciously.

### Position A — "FaceFusion-plus-audio-stack"
Stay in the OSS face-swap arena. Match FaceFusion on the capabilities they have (LivePortrait, multi-face, colorizer), keep beating them on the things they don't ship (voice clone, fine-tune, rotoscope, snapshot/checker hygiene).
*Effort:* months, focused.
*Risk:* FaceFusion has the brand. Hard to win on inswapper quality alone.

### Position B — "Production pipeline for one-person content creators"
Stop competing on raw face-swap quality. Lean into the full creator workflow: lipsync + voice clone + rotoscope + audio remix + history + watermark + virtual-cam — the things FaceFusion *doesn't* do. Become the tool you reach for when you're shipping a clip end-to-end, not when you're benchmarking face-swap PSNR.
*Effort:* moderate; mostly polish + workflow integration.
*Risk:* market segment is small. But it's the segment that already uses v2.

### Position C — "Stylized character production toolkit"
The thing you actually keep trying to use v2 for (lizard, monkey, demon). Lean into stylized faces, animation, RotoBrush parity, integration with diffusion pipelines for character work. This is where Pika/Kling don't fit (private/offline) and where FaceFusion isn't aimed.
*Effort:* high, requires diffusion stack you keep deleting.
*Risk:* this session showed the architecture mismatch is fundamental; needs research, not feature-builds.

**My honest read:** **B is the right answer.** It plays to v2's existing strengths, doesn't pretend you'll out-research Pika, and doesn't repeat today's burn cycle on Path C diffusion experiments. Path A is reasonable if you actively enjoy the face-swap quality arms race. Path C only makes sense after a real motion-transfer architecture (LIA/LivePortrait class) is integrated *and* validated end-to-end on your clips.

---

## 5. Proposed roadmap (Position B as primary)

Tiers ranked by **leverage** (value per LOC), not chronology. Each item has a one-line "ship gate" — concrete success criterion before merging.

### Tier 0 — Hygiene & technical debt (do first, low risk)

| # | Item | Why | Effort | Ship gate |
|---|---|---|---|---|
| T0-1 | Wire `tools/check_tail_integrity.py` + `check_public_symbols.py` into `launch.bat` as a precheck | Catches today's truncation pattern before app start | 1h | Launch fails fast on null bytes or symbol drift |
| T0-2 | Resolve pending task #129 — rotoscope `gr.Video` Error overlay | Stops the recurring video preview failure | 2h | Rotoscope tab loads source video without "Error" message in 3 browsers |
| T0-3 | Verify task #34 — lipsync per-clip fine-tune | Confirms feature works after path fix | 1h on user side | One training run produces a non-empty ckpt + identity improvement on 1 test clip |
| T0-4 | `git init` v2 and v2_github_release; commit baselines | Stops three-way drift like Titan suffered | 30 min | Both trees have `baseline-2026-06-11` tag pushed to private remote |

### Tier 1 — Workflow polish (high leverage, low risk)

| # | Item | Why | Effort | Ship gate |
|---|---|---|---|---|
| T1-1 | Cross-tab "Send to ..." plumbing | Reduces copy-paste between Lip-Sync / Face Swap / Rotoscope | 1-2 days | Render output appears as input in target tab with one click |
| T1-2 | Per-render JSON sidecar (all knob values) in `recordings/` | Reproducibility; lets History tab re-load a render's exact settings | 1 day | Reload from history reproduces an identical render bit-for-bit (modulo nondeterminism) |
| T1-3 | Project-level "session" concept: bundle source + masks + presets + outputs | One folder per project; clean handoff between sessions | 2-3 days | Open project → all state restored without re-uploading |
| T1-4 | "Generate timecoded preview" extended to all video outputs (already partial per #125) | Easier review on long clips | 4h | Every video output widget has the preview button |

### Tier 2 — Quality wins on existing tabs (medium leverage, medium risk)

| # | Item | Why | Effort | Ship gate |
|---|---|---|---|---|
| T2-1 | **Add SimSwap-512 / FaceDancer backend** | Better temporal stability on inswapper-hard clips; we already have `swap_backends/` abstraction | 3-5 days | A/B on 3 reference clips: equal or better identity, equal or better temporal flicker score (LPIPS frame-pair) |
| T2-2 | **GFPGAN → CodeFormer + RestoreFormer++ option** | More restoration models, comparable to FaceFusion | 2 days | Each model can be selected per-render; produces visible quality delta |
| T2-3 | Inswapper "pixel boost" parity (256/512/768 already done as #45 — verify and document) | Match FaceFusion's headline feature | 2h doc + audit | Existing slider, documented in README + visible in History sidecar |
| T2-4 | Temporal smoothing across frames (already done as #99 — A/B test honest impact) | Verify the win is real | 4h | Side-by-side on a head-turn clip shows reduced jitter |
| T2-5 | Identity reference picker — multi-image average embedding | Closer to commercial quality; we have the embedding stack | 2 days | 3-image average produces measurably more stable identity over 10s clip |

### Tier 3 — Catch FaceFusion's headline features (medium leverage, medium risk)

| # | Item | Why | Effort | Ship gate |
|---|---|---|---|---|
| T3-1 | **LivePortrait integration as new "Animate" tab** | FaceFusion's standout 2026 feature. Drives a still image with an audio file or a driver video | 1-2 weeks | Audio-driven talking head from one photo works end-to-end |
| T3-2 | Frame colorization (B&W → color) | FaceFusion 3.6 feature; useful + small | 3-4 days | One reference B&W clip produces a recognizable colorized output |
| T3-3 | Multi-face replace-by-cluster | FaceFusion has it; we have face_selector_mode (#41) but not multi-target | 1 week | A 2-person clip swaps both faces independently |

### Tier 4 — Research-grade features (high risk, high uncertainty)

These are the kind of thing today went sideways on. Don't start them until Tier 0-2 is done and there's a clean validation gate.

| # | Item | Why | Effort | Risk |
|---|---|---|---|---|
| T4-1 | MuseTalk integration as second lipsync option | Real-time, lower latency than LatentSync | 1-2 weeks + GPU validation | License compat, real-time tradeoffs |
| T4-2 | Diffusion-based face swap (DynamicFace / VFace) | Better temporal stability than inswapper | 3-4 weeks + research | High; today's session demonstrated the failure mode |
| T4-3 | Motion-transfer architecture for stylized characters (LIA, FOMM) | The actual right path for your monkey/lizard clips | 3-4 weeks | The thing you keep trying to ship — needs real Phase 0 spike first |
| T4-4 | Background restyle (re-attempted with motion-transfer or video-diffusion) | Replaces ripped Restyle | 3-4 weeks | High; today proved single-frame img2img doesn't ship |

---

## 6. Anti-roadmap — things to NOT do

Each cost real time today or in past sessions. Listed so they don't sneak back in:

1. **"Add Flux because it's better than SD1.5"** — true, but the architectural mismatch (close-up portrait + flat-background depth CN) is what killed Restyle. Flux doesn't fix that.
2. **Building click-landmark UI for non-face subjects** — Creature Swap died on this. ArcFace is human-shaped; clicking points on a demon doesn't make the template generalize.
3. **Big-bang refactors of `faceswap/ui.py`** — 96 KB monolith, but every edit risks the Edit-tool truncation pattern that bit twice today. Split when you have a real reason, not for cleanliness.
4. **Building features without ORT / diff validation gates** — Titan worklog already flagged this as the #1 anti-pattern. Same applies here.
5. **More tabs without first deduplicating workflows** — already at 8 tabs. Adding another (Animate, Colorize, etc.) should consolidate, not splinter.

---

## 7. Suggested first sprint (2 weeks, honest pace)

Sprint goal: **Close all Tier 0 + ship one Tier 1 (T1-1 "Send to") + ship one Tier 2 (T2-1 SimSwap backend) with honest A/B results.**

Why this sprint: it closes the existing pending work, reduces today's recurring failure modes structurally (git, prechecks), and adds one demonstrable quality win + one workflow win — without touching the diffusion stack that keeps eating sessions.

**Week 1**

- Day 1: T0-1 launcher prechecks, T0-4 git init both trees, T0-2 rotoscope video overlay
- Day 2: T0-3 finetune verification, T1-2 JSON sidecar groundwork
- Day 3-5: T1-1 "Send to" plumbing across all tabs

**Week 2**

- Day 1-3: T2-1 SimSwap-512 backend research + integration (`swap_backends/` already abstracted)
- Day 4: A/B harness — 3 reference clips, LPIPS + ArcFace cosine drift
- Day 5: Document A/B results in `_eval/` dir, ship to user

**Sprint exit criteria:**
- Symbol checker + tail-integrity run green at launch
- Both trees in git with tagged baselines
- Rotoscope source video loads without overlay error
- Lipsync fine-tune verified on one user clip
- "Send to" works across at least 3 tab pairs
- SimSwap backend ships with A/B numbers documented

---

## 8. Open questions for the user before commitment

1. **Position B vs A vs C** — which framing do you actually want? Without picking one, T3 and T4 are coin-flips.
2. **Hardware target** — current code assumes A6000 48GB. Should v2 ship to consumers with 8-12 GB cards? Affects every Tier 4 item.
3. **Commercial use** — inswapper_128's license is gray. Should v2 push toward only-commercial-clear backends, or leave it user-choice?
4. **Distribution** — desktop-only or eventually browser? Affects whether T1-3 sessions get cloud-sync or stay local-only.
5. **Stylized character work** — Creature Swap died but the use case (your lizard/monkey/demon clips) keeps coming up. Is that the actual product or a side-quest?

Don't commit to T1-T4 until 1 and 5 are answered honestly. Otherwise this roadmap becomes another Restyle.

---

## Sources

- [FaceFusion changelog](https://docs.facefusion.io/introduction/changelog) — 3.6.0 features (Pixel Boost, LivePortrait, frame colorizer)
- [FaceFusion 2026 guide](https://magichour.ai/blog/how-to-use-facefusion) — current state
- [Hallo3 paper](https://arxiv.org/html/2412.00733v1) — diffusion-transformer talking head
- [LiveTalk-Unity (LivePortrait + MuseTalk pipeline)](https://github.com/arghyasur1991/LiveTalk-Unity)
- [Roop archive note + Roop-Unleashed](https://www.tooljunction.io/ai-tools/roop)
- [Pika / Kling / Runway 2026 comparison](https://chatcut.io/blog/best-ai-video-generator-2026)
- [DynamicFace (3D priors, CVPR 2025)](https://arxiv.org/pdf/2501.08553)
- [VFace (training-free diffusion swap, 2026)](https://arxiv.org/pdf/2602.07835)
- [Face-swap temporal stability assessment](https://arxiv.org/pdf/2505.20985)
- [SimSwap / FaceDancer in OSS landscape](https://www.jaiportal.com/alternatives/facefusion-faceswap-alternatives)
