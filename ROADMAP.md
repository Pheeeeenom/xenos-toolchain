# Roadmap — forward phases

Phases 0–2 (toolchain) and 3a (GLSL frontend v1) are **done and
hardware-verified**. This is the plan from here. Each phase has a goal,
scope, and a single exit criterion. Phases are roughly independent — they
can be reordered; ⭐ marks the ones that need a **user decision**, not just
engineering.

---

## Phase 3b — GLSL frontend: control flow + completeness
**Goal:** xsc compiles "real" shaders, not just straight-line ones.
- **Control flow** (the headline rock): `if`/`else` → predication or
  `cjmp` (encoders + `feat_*` shaders already staged); `for` with constant
  bounds → unroll first, real loops later. Structured-CF lowering from
  SPIR-V's block/merge structure.
- **Long tail:** non-identity swizzles (`.bgr`), integer ops, general
  mid-expression `CompositeConstruct`, more `GLSL.std.450` builtins
  (`normalize`, `length`, `pow`, `floor`, `sin/cos`), arbitrary `clamp`,
  fold `0.0`/`1.0` literals into the free source-select swizzle.
- 🎯 **Exit:** a genuinely branchy shader (e.g. a lit shader with an
  `if`-gated highlight, or a procedural pattern with a small loop) compiles
  and renders correctly in `app/glslgallery` on console.
- *Effort:* control flow is real work (days, not hours); the long tail is
  incremental, one builtin per demand-driven shader. Risk: predication
  semantics on hardware (the `feat_pred` shader is the canary).

## Phase 4 — Toolchain release ⭐
**Goal:** the scene can find, build, and use this. It's the keystone 360
homebrew has lacked since 2005 — it should ship.
- ⭐ **Decisions:** project name (drop "xost" working title); license
  (MIT vs BSD-3 — both safe, see `docs/license-audit.md`); corpus blobs
  (keep w/ provenance note vs regenerate-only); build-artifact cleanup.
- Add `LICENSE` + `THIRD_PARTY_NOTICES.md` (Xenia BSD-3, freedreno MIT,
  XenosRecomp MIT); polish `README.md` with the hardware photos; `git rm
  --cached` the committed build artifacts; move demo `.elf32`s to a release.
- GitHub repo + a tagged release; a factual scene announcement
  (GBAtemp / XboxHacker), linking the photos and the honest scope.
- 🎯 **Exit:** public repo, tagged release, one external person builds a
  shader with it. *Effort:* a focused session once the decisions are made.

## Phase 5 — Dashboard foundations (the real end goal pivot)
**Goal:** a bare-metal libxenon **dashboard** that browses and launches
homebrew — the first working version of the custom-OS vision, with this
toolchain as its renderer.
- **The crux problem:** libxenon's ELF loader is *replace-yourself* (it
  reboots into the loaded ELF). A dashboard needs **launch-and-return** —
  either a resident kernel with a callback, or a reload-on-exit trampoline.
  Solve this first; everything else depends on it.
- **Renderer (already have it!):** text/font rendering, 2D UI quads, image
  loading — all shadeable via the combiner API and the GLSL frontend. The
  `hud_*` 7-seg code and `glslgallery` are the seeds.
- **Platform:** input (XInput — `input_xinput` exists), storage browsing
  (FATX/USB via libxenon drivers), config persistence.
- 🎯 **Exit:** a bootable dashboard ELF that lists homebrew on USB/HDD,
  renders a real UI, and launches a selected ELF (and ideally returns).
- *Effort:* weeks. This is a new sub-project; the renderer being done is
  the big head start. ⭐ Worth a go/no-go before starting.

## Phase 6 — Toward a true custom OS (long game)
**Goal:** the full vision — resident kernel, scheduler, syscalls, a real
dashboard as the OS shell.
- MMU/memory management, task scheduling, a syscall surface, driver model
  on top of (or replacing) libxenon's bare-metal services.
- 🎯 **Exit:** undefined by design — this is the years-horizon north star
  that Phase 5 is the honest first increment of.

---

## Cross-cutting (always on)
- **Hardware-verify every milestone** — the `glslgallery` / demo-app rig
  makes this one boot per feature. Never let "oracle-green" stand in for
  "rendered on console" for anything value-dependent.
- **Oracle + tests first** — XenosRecomp cross-check, byte-parity where a
  golden exists, regression suite green before commit.
- **Honest scope in docs** — `docs/xsc.md` capability table stays current;
  no feature claimed that isn't tested.

## Suggested order
3b (finish the compiler) → 4 (release it) → 5 (dashboard). But 4 can come
*first* if you'd rather ship the keystone now and finish the compiler in the
open. The dashboard (5) is the destination; everything before it is
foundation that's largely built.
