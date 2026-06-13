# xost — contributor handoff

Pick-up notes for anyone (human or agent) continuing this toolchain. The
**non-obvious** stuff is in "Invariants & gotchas" — read that first; it has
already cost multiple hardware debug cycles. Capability/feature details live in
[xsc.md](xsc.md); this doc is the orientation + the traps + the roadmap.

## What xost is

An open, clean-room shader toolchain for the Xbox 360's Xenos GPU — no leaked
XDK, no Microsoft tooling. North star: a complete **GLSL ES 2.0** compiler so
GLES2 apps/emulators can render on the 360. Three layers:

- **`xdis/`** — container format + ucode disassembler/decoder (the ISA model).
- **`xasm/`** — encoder (`encode.py`), the `Program` builder + 16-GPR register
  allocator + hardware-loop emission (`program.py`), and the N64 combiner
  generator (`combiner.py` / `xcomb`).
- **`xsc/`** — the GLSL→Xenos compiler: `spirv.py` (reader), `resolve.py`
  (interface/types → Xenos roles), `lower.py` (SPIR-V body → Program calls),
  `unroll.py` (constant-trip loop unroll + mem2reg pre-pass).

Pipeline: **GLSL → glslang → SPIR-V → xsc → Xenos container**
(`Xe_LoadShaderFromMemory`-compatible). glslang does parsing/opt; xsc lowers.

## Current state (see the gap map below)

The **fragment** frontend is strong and hardware-verified: ~35 builtins, nested
if/else (branch-free `cnd` selects), constant-trip loops, uniforms/samplers/
literals, real lighting/multi-texture/fog shaders. The **vertex** frontend is
thin — it lowers essentially one pattern (`mvp*pos` + passthrough). Real loop
control flow (`LOOP_START/END`) is hardware-proven but not yet wired into the
GLSL frontend. Renders a real app: mupen64-360's shaders go through it.

## Invariants & gotchas (READ THIS)

1. **The 16-GPR ceiling is a silent GPU hang.** Pixel shaders have r0..r15.
   Exceeding it does not error — it *hangs the console*. `program.py`'s
   linear-scan allocator (`PixelProgram.finish`, runs only when over budget)
   keeps shaders inside it. `scalar_dest` (rcp/sqrt/sin/cos results) **must** be
   counted toward the GPR total — forgetting it under-declares and hangs.

2. **The XenosRecomp oracle validates DECODE, not EXECUTION.** It round-trips
   ucode→HLSL and confirms structure. It will happily accept shaders that *hang
   or miscompile on silicon*. Every oracle-pass still needs hardware
   verification. Bugs the oracle structurally cannot see, all found the hard
   way on real hardware:
   - A pixel shader must **`ALLOC` its color export** before the export
     instruction (`p.alloc_colors()`). Missing → hang.
   - A pixel shader must **declare interpolators matching the VS export layout**
     (`p.input(r, usage)`), and must **not reuse an interpolator register
     (r0/r1...) as a temp**. Missing → the VS-exported varying leaks into the
     output as per-pixel garbage.
   - Scalar division with a scalar (n==1) divisor must **broadcast** to all
     components, or .yzw are garbage.
   These are PS-setup requirements `lower_fragment` does automatically; hand-
   authored shaders must replicate them.

3. **Byte-identity is a regression gate.** `modulate.frag` must stay
   byte-for-byte identical to the hardware-photographed combiner
   (`examples/out/comb_0_modulate.bin`). The unroll/loop pre-passes are no-ops
   when there's no loop precisely to preserve this. If you change `program.py`
   or `lower.py`, run the suite and confirm `modulate` is still byte-identical.

4. **Hardware verification is the real gate.** The app `app/glslgallery` cycles
   GLSL shaders (compiled at build time by `gen_header.py`) on a rotating cube
   with a HUD. Add a slot, build the elf32, the user flashes it on their RGH
   console. Place never-tested ops **last** in the gallery so a hang can't hide
   the slots before it.

## Build & test

- **Python / xsc / oracle (Windows or WSL):** `glslang.exe` and
  `XenosRecomp.exe` are Windows binaries under `tools/` and
  `reference/XenosRecomp/build/`. Run tests with plain `python`:
  - `python tests/test_xsc_lower.py` — 30 GLSL shaders, oracle-validated +
    modulate byte-identity (the main gate).
  - `python tests/test_program_loop.py` — hardware-loop CF emission.
  - others: `test_xsc_resolve`, `test_encode_fuzz`, `test_xcomb_parity`,
    `test_textasm`, `test_container_regen`.
- **C / libxenon (Debian WSL):** the C toolchain lives in the **Debian** WSL
  distro at `/usr/local/xenon` (xenon-gcc 9.2.0, libxenon BSD-3). Build an app:
  ```
  wsl -d Debian -- bash -lc 'export DEVKITXENON=/usr/local/xenon; \
    cd /mnt/d/claude/fable/xenos-toolchain/app/glslgallery && make'
  ```
  PowerShell→WSL quoting mangles `$()`/spaces — use a script file or `bash -lc`.
  Regenerate the gallery header first: `python app/glslgallery/gen_header.py`.

## The GLES2 gap (data-driven punch list, prioritized)

Measured by compiling representative shaders through glslang→xsc:

1. **Vertex frontend — still the biggest remaining gap.** `lower_vertex` now
   handles chained mat4 x vec4 transforms (`model*pos` then
   `viewproj*world`) and basic ALU values (`+`, `-`, `*`, `vec*scalar`,
   `dot`, swizzled temps) feeding position or interpolator exports. Remaining
   high-value work is the larger fragment/value-lowering refactor for VS:
   normalize/sqrt/sin/cos/etc., general component writes/constructs, and
   broader matrix forms (`mat*mat`, `vec*mat`, inverse/determinant).

2. **Fragment control-flow generality:**
   - **if/else that assigns a local** then outputs it is now lowered as the
     same branch-free `cnd*` select tree used for direct output assignments.
   - **`discard`** (→ Xenos `kill_*`) now compiles and oracle-validates.
     Because the oracle only checks decode, hardware is still the gate: the
     final `glslgallery` slot G26 (`DISCARD`) should cut away half the cube's
     pixels (`v_uv.x < 0.5`) on console.

3. **Data-dependent loops (Phase B, in flight).** Mechanism proven: see Loop CF
   below. Wire `unroll.py`'s data-dependent bail to a real-loop lowering that
   keeps the loop-carried var in a persistent GPR and binds the uniform bound to
   a loop constant.

4. **Texture types.** xsc has one fetch template (`tfetch3`, 2D). `samplerCube`
   currently *silently miscompiles* to a 2D fetch — at minimum make it a clean
   error. 3D/array/LOD/bias also unsupported.

## Loop control flow (Phase A, hardware-proven) — reference for Phase B

`program.py` `begin_loop(loop_id)` / `end_loop()` wrap a body's exec block in
`LOOP_START`/`LOOP_END`. Confirmed semantics (XenosRecomp + xenia + silicon):

- `LOOP_START.address` = **skip** target = CF index *after* `LOOP_END` (taken
  when count==0). `LOOP_END.address` = the **body** exec (LOOP_START+1) to
  repeat. Addresses are absolute CF-instruction indices.
- Trip count = `loopconst & 0xFF` (countdown); `aL` = `iter*step + start`.
- **Loop constants are GPU register `0x4908+id` = libxenon
  `integer_constants[8+id]`** (words 0–7 = booleans, 8–39 = the 32 loop
  consts). `rput(integer_constants, 40)` already uploads them, so the app sets
  the count directly — **no libxenon patch needed**:
  ```c
  xe->integer_constants[8+id] = count | (start<<8) | (step<<16);
  xe->dirty |= DIRTY_INTEGER;   // both are public in xe.h
  ```
- Proven on console via `glslgallery` G23 (LOOPHW): `gen_header.build_loop_proof`.

Phase C (later): `aL`-relative constant addressing (`c[base+aL]`) for array
indexing — the genuinely useful GLES2 loop pattern — and predicated `LOOP_END`
for `while(computed condition)`.

## Conventions

- **Work only inside `D:\claude\fable`.** Don't browse elsewhere; cloning public
  GitHub repos *into* fable is fine (`reference/` holds them).
- **Commit identity (the maintainer's call):** this repo's history is authored
  by `xost <xost@local>` — a per-project identity, not the contributor. Local
  git config is already set to it. Match the existing `git log` author.
- Build artifacts (`.elf32`, `build/*.o`) are tracked in-repo by existing
  convention; a release-time `.gitignore` sweep is noted in
  `docs/release-checklist.md`.
