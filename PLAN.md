# Xenos Open Shader Toolchain — Project Plan

> Working title: **`xost`** (Xenos Open Shader Toolchain) — name TBD.
> The missing keystone of Xbox 360 homebrew: an open-source way to author GPU
> shaders for the Xenos, replacing the leaked-XDK compiler as the scene's only
> path to hardware-accelerated 3D. The playbook is CafeGLSL (Exzap's Mesa-fork
> GLSL compiler that unblocked Wii U homebrew); the ISA knowledge is already
> public in Xenia; the consumer API already exists in libxenon.

## Win condition

**A libxenon "hello triangle" running on a real RGH console with every shader
byte produced by our toolchain** — followed by adoption: at least one real
homebrew project (candidate: the Mupen N64 combiner generator) emitting its
shaders through us instead of XDK-compiled blobs.

## The gap, in one artifact

`reference/mupen64-360/source/shaders.S` — the flagship *open-toolchain* 360
emulator ships its shaders as raw byte dumps containing the strings `vs_3_0`
and `2.0.11626.0`: Microsoft XDK compiler output, pasted in. Even the legal
scene runs on smuggled shader binaries because no open compiler exists. That
is what we replace.

## Reference corpus (cloned, `reference/`)

| Repo | What it gives us | License |
|---|---|---|
| `xenia` | `src/xenia/gpu/ucode.h` (1,925 ln): complete ISA encodings as bitfield structs (ControlFlowOpcode, FetchOpcode, AluScalarOpcode, AluVectorOpcode). `shader_translator.cc`: full decoder (consumer direction). `shader_interpreter.cc` (1,192 ln): executes ucode on PC = our execution oracle. `shader_compiler_main.cc`: standalone CLI harness. | BSD-3 |
| `libxenon` | `drivers/xenos/xe.c/h`: the consumer API — `Xe_LoadShaderFromMemory`, `XenosShaderHeader/Data` container structs, `Xe_ShaderApplyVFetchPatches` (driver patches vfetch to match vertex format — we must emit patchable vfetch like the XDK did). `ucode.c`: small embedded shader blobs. | GPL |
| `mupen64-360` | `source/shaders.S`: golden corpus of real XDK-compiled VS/PS blobs incl. container headers; `Rice_GX_Xenos/` shows real consumption. | GPL |
| `XenosRecomp` | Independent second reference for ucode semantics + the constant-table container layout (`constant_table.h`). | — |
| (later) `mesa` | `src/gallium/drivers/freedreno/a2xx/` — the ir2 NIR backend targets Adreno 200, the Xenos-derived mobile GPU: a shipping open compiler for a near-identical ISA. Basis for Phase 3. | MIT |

## Architecture (three artifacts, smallest-first)

```
            Phase 2/3: xsc (compiler)            Phase 1: xasm (assembler)
  GLSL/NIR (Mesa ir2 adapt)  or  v1 mini-lang ──► xenos assembly text
                                                       │
                                                  encoder (ucode.h tables)
                                                       │
                                              ucode words (BE) + vfetch map
                                                       │
                                          container writer (XenosShaderHeader,
                                          constant table, vfetch patch table)
                                                       │
                        ┌──────────────────────────────┴───────────────┐
                  PC validation                                  console use
            Xenia translator round-trip                    Xe_LoadShaderFromMemory
            Xenia shader_interpreter exec                  (libxenon, unmodified)
```

## Phases

### Phase 0 — Disassembler + golden corpus  *(decoder first: learn by reading)*
- Port/wrap Xenia's decoder into a standalone `xdis` (no Xenia build needed);
  dump the mupen64-360 + libxenon blobs to readable assembly.
- Parse + document the container format (header magic `10 2a 11 01`, offset
  table, constant names, ucode offset) against `xe.c`'s loader.
- **🎯 Exit:** every blob in the corpus disassembles cleanly; container format
  documented in `docs/container.md`; ambiguities resolved against XenosRecomp.

### Phase 1 — `xasm` assembler + container writer  *(the unlock)*
- Assembly syntax ≈ our disassembler's output (round-trip by construction).
- Encoder from `ucode.h` bitfields; big-endian word emission; vfetch patch
  table generation; minimal constant table.
- Tests: (a) disassemble→reassemble the golden corpus **byte-identical**;
  (b) hand-write passthrough VS + solid-color PS, validate via Xenia's
  translator (accepts → emits valid DXBC/SPIR-V) and shader_interpreter
  (executes with expected outputs).
- **🎯 Exit (the milestone):** libxenon triangle sample on the **real console**
  with our assembled shaders replacing the XDK blobs. First fully-open
  hardware-3D on 360. (PC proxy until console session: Xenia validation green.)

### Phase 2 — `xsc` v1: minimal compiler
- Small typed shading language (or strict GLSL subset) → ucode. Scope to what
  emulator renderers actually need: mat4 transform, texture sample, combiner
  arithmetic, constants. No control flow in v1 (Xenos CF ops exist; defer).
- **🎯 Exit:** the N64 color-combiner shader family (from the Mupen project's
  generator) expressible and correct vs the interpreter oracle.

### Phase 3 — Mesa/NIR backend  *(the CafeGLSL moment — stretch)*
- Adapt freedreno a2xx ir2 (NIR→a2xx) to emit Xenos ucode + our container.
  Full GLSL, real optimizer. Offline PC tool first; on-console runtime
  compiler (CafeGLSL-style) only if demand justifies.
- **🎯 Exit:** vitasdk-style GXM samples... (n/a) — concretely: a textured,
  lit demo written in GLSL runs on console via our pipeline.

### Phase 4 — Adoption & release
- Docs, examples, libxenon PR (sample apps use xost), port one real project's
  shaders. Candidate: user's Mupen combiner generator gains a legal backend.
- **🎯 Exit:** a scene project ships with our toolchain in its build.

## Validation stack (no hardware needed until the milestone moments)
1. **Golden round-trip** — corpus disasm→asm→bytes identical.
2. **Xenia translator** — our ucode → valid DXBC/SPIR-V (it rejects malformed
   programs loudly).
3. **Xenia shader_interpreter** — execute ucode on PC, assert register/export
   results (the dynarec-lockstep idea, applied to shaders).
4. **Real console** — user's zero-fuse RGH unit, libxenon samples. Ground truth.

## Risks
1. **vfetch patch semantics** — the XDK compiler emits vfetch in a form the
   runtime patches per vertex-format; we must match libxenon's expectations
   exactly (`Xe_ShaderApplyVFetchPatches`). Mitigation: golden corpus diffing.
2. **Container unknowns** (interpolator counts, export masks, header fields
   beyond what xe.c reads). Mitigation: three independent references (xe.c,
   XenosRecomp constant_table.h, byte-level corpus).
3. **Pixel-shader export/EDRAM interactions** (sRGB, exponent bias, depth
   export). Mitigation: keep v1 exports simple; Xenia semantics as guide.
4. **License hygiene** — encoder derived from Xenia (BSD-3, fine); keep GPL
   (libxenon/mupen blobs) on the test side, not in the toolchain proper.

## Status
- 2026-06-11: Project chosen after scene gap analysis (see memory). Reference
  corpus cloned. ISA/loader/corpus surveyed; feasibility confirmed.
- 2026-06-11: **Phase 0 substantially complete.** Corpus extracted
  (`corpus/vs0.bin`, `ps_fb0.bin`); container format parsed + documented
  (`docs/container.md`); full ISA transcribed from Xenia into
  `xdis/xenos_ucode.py` (~2030 ln, BSD-attributed, self-tested; zero
  ucode.h↔XenosRecomp encoding discrepancies; 3 AMBIGUITY notes in code).
  Both golden blobs disassemble cleanly and semantically coherently
  (VS: 4×vfetch + ortho mat4 dp4→oPos + interpolator exports; PS:
  cjmp !b128 use_tex switch, tfetch2D tf0, dual alloc colors). Saved as
  `corpus/*.dis`. vfetch slots confirmed to be unpatched templates for
  Xe_ShaderApplyVFetchPatches (Phase 1 design input: emit patchable vfetch).
  Toolchain note: host has VS18 Community (MSVC 19.50 x64) + CMake/Ninja.
- 2026-06-11: **Phase 0 COMPLETE — oracle convergence.** XenosRecomp built
  with VS18 and run on both corpus blobs (`corpus/*.recomp.hlsl`); its HLSL
  matches xdis instruction-for-instruction (same dp4/ortho transform, same
  b128 use_tex branch, same swizzles) and it resolved constant names from the
  container, validating docs/container.md.
- 2026-06-11: **Phase 1 PC-side COMPLETE** (scope note: the *text* parser
  moved to Phase 2 — the builder API is the assembler-as-library and is how
  generators will emit shaders anyway; corpus byte-round-trip is covered by
  exact mode instead).
  - git repo initialized (after one agent stall; narrow-lane agent prompts
    from now on).
  - `--exact` disasm mode: lossless ;; raw/pad/tail annotations;
    reconstruct_from_exact → byte-identical (tests/test_exact_total.py).
  - `xasm/encode.py`: builders for all CF/vfetch/tfetch/ALU forms, 16k-case
    fuzz vs decoders green (tests/test_encode_fuzz.py).
  - Container emitter regenerates both corpus blobs byte-identically from
    semantic params (tests/test_container_regen.py). SQ_PROGRAM_CNTL now
    computed from xenia registers.h layout (validated vs corpus values).
    Decoded: interpolator flags = (export_index<<4)|type_nibble; ps_field18
    = (interp_count<<16)|7; VertexElement.address = template vfetch slot.
  - **examples/author_passthrough.py: first fully open-toolchain Xenos
    shader pair** (passthrough VS 228 B + PS 148 B), inversion-proof against
    corpus instructions, validated by XenosRecomp (correct HLSL semantics
    both).
- 2026-06-11: **🏆 PHASE 1 EXIT GATE PASSED ON HARDWARE.** Two RGB triangles
  (opposite windings) rendered on the user's RGH console via libxenon
  (Debian WSL xenon-gcc 9.2; app/triangle). On-console dump confirms
  Xe_ShaderApplyVFetchPatches consumed our patch table exactly as designed
  (templates → fmt 0x26/FLOAT4, stride 8, offsets 0/4 per our
  VertexElements). Mystery fields (0x290, field10) hardware-confirmed
  non-blocking. Lesson: libxenon default cull state after InvalidateState
  culls — set Xe_SetCullMode explicitly (build-1 "no triangle" cause).
  First known fully open-toolchain hardware 3D on Xbox 360.
  NEXT (Phase 2): minimal compiler / combiner-style shader generator on top
  of xasm builders; project naming + release prep; text-assembler frontend.
- 2026-06-11: **Phase 2 core landed, hardware-verified same day.**
  (a) spincube: mvp dp4 transform VS (xost_vs1), per-frame constants, depth,
  time-based animation — verified. (b) Live CPU/GPU% HUD (7-seg vector font
  from quads, identity-matrix overlay pass) — verified: CPU 0% / GPU 3%.
  (c) **xasm/program.py Program builder**: declarative authoring (auto
  slot/CF layout, exec chunking, sequence bits, reg scan → SQ_PROGRAM_CNTL,
  VertexElement/Interpolator/export metadata). (d) **Textured pipeline**
  (xost_vs2 + xost_ps1 via Program API, tfetch2D × vertex color) + texcube
  app (two shader pairs per frame, procedural checkerboard, linear ARGB) —
  **verified on console: 6 tinted checkered faces.** tfetch, sampler
  binding, multi-interpolator routing, FLOAT2 vfetch component fixup all
  proven. Hardware-proven feature matrix now: vfetch (F4/F2), tfetch2D,
  ALU vector ops (dp4/max/mul), exports (oPos/oN/oC0), constants (float4
  matrices), depth, shader switching. REMAINING unproven: bool constants +
  cjmp control flow, scalar co-issue, loops/predication, ubyte4 verts,
  depth export/MRT. NEXT: combiner generator (the emulator enabler) or
  naming/release.
