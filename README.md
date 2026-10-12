<div align="center">
  <img src="assets/xost.png" alt="xost" width="640">
</div>

# xost (working title)

> **Name is TBD** — "xost" = Xenos Open Shader Toolchain, a placeholder until a
> proper name is chosen before release.

**An open-source toolchain for authoring Xbox 360 (Xenos) GPU shaders, with
zero Microsoft tooling — every shader byte produced by open code, verified on
real hardware.**

## The gap this fills

Since the earliest days of Xbox 360 homebrew, there has been exactly one way
to get hardware-accelerated 3D: shader binaries compiled with Microsoft's
leaked XDK. Even the flagship open projects ship them — `mupen64-360`, the
open-toolchain N64 emulator, carries its shaders as raw byte dumps in
`shaders.S`, with the strings `vs_3_0` and `2.0.11626.0` (the XDK compiler's
version stamp) plainly visible inside. No open assembler or compiler for the
Xenos has ever existed, so the legal homebrew scene has run on smuggled
compiler output for nearly two decades.

This toolchain replaces that. It can disassemble the XDK-era shader
containers, author new shaders from scratch through a builder API, generate
N64-style color-combiner pixel shaders on the fly (in Python offline, or in
dependency-free C99 at runtime on the console), and emit containers that
libxenon's `Xe_LoadShaderFromMemory` loads unmodified — all proven on a real
RGH console.

The playbook is CafeGLSL, the Mesa-fork GLSL compiler that unblocked Wii U
homebrew. The ISA knowledge was already public in the Xenia emulator; the
consumer API already existed in libxenon. This project connects them.

## It runs on real hardware

*(Photo placeholders — files to be added under `docs/photos/`.)*

| | |
|---|---|
| [photo: triangles] `docs/photos/triangle.jpg` | **First fully open-toolchain hardware 3D on Xbox 360:** two RGB triangles, every shader byte authored by this toolchain, on an RGH console via libxenon. |
| [photo: textured cube] `docs/photos/texcube.jpg` | Textured pipeline: spinning cube with six tinted checkerboard faces — mat4 transform VS + `tfetch2D` PS, two shader pairs per frame. |
| [photo: combiner grid] `docs/photos/combiner_modes.jpg` | N64-style combiner modes generated as native Xenos pixel shaders: modulate, tex×PRIM, lerp, decal, shade, inverted, multitexture, 2-cycle chains. |
| [photo: HUD] `docs/photos/hud_cpu0_gpu3.jpg` | On-screen performance HUD reading **CPU 0% / GPU 3%** while rendering the animated textured cube. |

## What works today

Everything below is implemented in this repository and verified — "verified on
hardware" means rendered on a real RGH Xbox 360, via libxenon with photos, or
via the xsdk homebrew SDK with every pixel read back and checked.

- **`xdis/` — disassembler + container tooling.** Full Xenos microcode decoder
  (control flow, vfetch, tfetch, vector/scalar ALU; ISA tables transcribed
  from Xenia, BSD-3). Parses the XDK-era shader container format
  (documented in [`docs/container.md`](docs/container.md)) and an `--exact`
  mode whose output reconstructs the input **byte-identically**.
- **`xasm/` — assembler-as-library.** Encoders for every instruction form
  (`encode.py`, fuzz-tested as exact inverses of the decoder), and a
  declarative `Program` builder (`program.py`) that handles slot/CF layout,
  register scanning → `SQ_PROGRAM_CNTL`, vfetch-patch / interpolator / export
  metadata, and container emission. A text-assembler frontend is in progress
  and not yet committed.
- **`xasm/combiner.py` — N64 color-combiner compiler.** Compiles the RDP
  `(A − B) × C + D` model (1-cycle and 2-cycle, RGB + alpha) directly to
  Xenos pixel-shader containers, with algebraic folding so trivial modes cost
  a single instruction. This is the emulator-enabling piece: the role
  GLideN64-class plugins fill by generating GLSL, with no shader compiler
  involved.
- **`xcomb/` — the same combiner compiler in dependency-free C99.** No
  allocation, endian-safe serialization, byte-identical output to the Python
  generator for every valid configuration (9/9 demo bank + 360-case fuzz
  parity). Compiles clean under MSVC `/W4 /WX` and xenon-gcc `-Wall -Werror`
  (~8 KB of PPC code) — suitable for generating shaders **at runtime on the
  console**.
- **`app/` — demo apps, all hardware-verified:** `triangle` (first open
  shaders on silicon), `spincube` (mat4 transform, per-frame constants,
  depth, live CPU/GPU HUD), `texcube` (textured pipeline), `combiner`
  (cycles 9 generated combiner modes with animated PRIM/ENV constants).
- **Hardware-proven feature matrix:** vfetch (FLOAT4/FLOAT2 + driver
  patching), tfetch2D (two samplers), vector ALU (dp4/mul/max/mad, negation),
  exports, float4 constants (vertex + pixel), depth, multi-interpolator
  routing, shader switching, 2-cycle combiner chains. Through xsdk: boolean
  constants and loops, vfetch in every 32-bit-word vertex format (float1-4,
  half2/4, byte4, short2/4, signed or not, normalised or whole numbers) from
  16 vertex streams, and tfetch2D with the fetch constant's filters,
  addressing, LOD bias and mips across 13 texture formats including DXT1-5
  (see [`docs/feature-tests.md`](docs/feature-tests.md#hardware-results)).

## Honest non-goals (for now)

- **No HLSL/GLSL frontend yet.** You author shaders through the Python
  builder API or the combiner generators. The plan (Phase 3 in
  [`PLAN.md`](PLAN.md)) is to adapt Mesa's freedreno a2xx `ir2` NIR backend —
  the Adreno 200 is a Xenos-derived GPU, so a shipping open compiler for a
  near-identical ISA already exists.
- **Text assembler is in progress** — the builder API is the supported
  authoring path today.
- **Unproven on hardware so far:** conditional jumps (`cjmp`), scalar
  co-issue, predication, depth export/MRT. The encoders exist; the hardware
  validation does not yet. Boolean constants (through `cexec`), real loops on
  loop constants, ubyte4 and the other vertex formats, and fetch-constant
  sampling are proven; see
  [`docs/feature-tests.md`](docs/feature-tests.md#hardware-results).
- This is a toolchain for homebrew (libxenon), not a retail SDK replacement.

## Quickstart

PC side needs only **Python 3.10+ (standard library only)**.

```sh
# Disassemble a shader container (corpus blob or your own)
python xdis/xdis.py corpus/vs0.bin
python xdis/xdis.py corpus/ps_fb0.bin --exact   # lossless, byte-reconstructible

# Author the example shaders + the combiner bank (writes examples/out/*.bin)
python examples/author_passthrough.py
python examples/author_textured.py
python examples/author_combiners.py
```

Building the demo apps needs the libxenon toolchain (xenon-gcc). The repo's
helper assumes Debian under WSL with libxenon installed at
`/usr/local/xenon`:

```sh
# inside WSL, from the repo root
bash app/wslbuild.sh app/triangle      # → app/triangle/triangle.elf32
bash app/wslbuild.sh app/combiner
```

Each app embeds its shaders via `gen_header.py` (regenerates
`source/xost_shaders.h` from `examples/out/*.bin`). Boot the resulting
`.elf32` on an RGH/JTAG console with XeLL as usual.

To generate combiner shaders at runtime in your own libxenon app, drop in
`xcomb/xcomb.c` + `xcomb/xcomb.h` (C99, no dependencies, caller-provided
buffer) and load the output with `Xe_LoadShaderFromMemory`.

## Architecture

```
                    authoring                          consuming
        ─────────────────────────────────    ──────────────────────────────
        xasm/combiner.py   xcomb/ (C99)          corpus/*.bin (XDK blobs)
        (N64 combiner →    (same, runtime,            │
         Xenos, Python)     byte parity)              ▼
                │               │                xdis/container.py ── parse
                ▼               │                xdis/xenos_ucode.py ─ decode
        xasm/program.py         │                     │
        (Program builder:       │                     ▼
         slots, CF layout,      │                disassembly (+ --exact
         reg scan, metadata)    │                 byte-identical roundtrip)
                │               │
                ▼               │
        xasm/encode.py          │            validation oracles
        (field → dwords,        │            ────────────────────
         inverse of decoder)    │            XenosRecomp (HLSL semantics)
                │               │            fuzz: encode↔decode (16k cases)
                ▼               ▼            parity: xcomb ↔ Python (byte-exact)
        xdis/container_emit.py  │            golden corpus byte-regeneration
        (XDK-style container)   │
                └───────┬───────┘
                        ▼
          Xe_LoadShaderFromMemory (libxenon, unmodified)
                        ▼
               real RGH Xbox 360 console
```

## How it's validated

1. **Golden corpus round-trip** — real XDK-compiled blobs disassemble and
   regenerate byte-identically (`tests/test_exact_total.py`,
   `tests/test_container_regen.py`).
2. **Encoder↔decoder fuzzing** — thousands of randomized cases per
   instruction family prove the encoders are exact inverses of the verified
   decoders (`tests/test_encode_fuzz.py`).
3. **XenosRecomp as independent oracle** — every authored shader is fed to
   XenosRecomp, an independent open implementation of ucode semantics, and
   its recovered HLSL is checked instruction-for-instruction
   ([`docs/xenosrecomp-oracle.md`](docs/xenosrecomp-oracle.md)).
4. **Python↔C99 byte parity** — `xcomb` output is compared byte-for-byte
   against the Python generator across the demo bank and randomized fuzz
   configurations (`tests/test_xcomb_parity.py`).
5. **Real hardware** — every milestone rendered on an RGH console; the
   on-console vfetch dump confirmed `Xe_ShaderApplyVFetchPatches` consumed
   our patch tables exactly as designed.

No benchmarks are claimed beyond what the on-screen HUD shows in the photos.

## Repository map

| Path | Contents |
|---|---|
| `xdis/` | Disassembler, container parser/emitter, corpus extraction |
| `xasm/` | Encoders, Program builder, N64 combiner compiler (Python) |
| `xcomb/` | C99 runtime combiner generator (byte parity with Python) |
| `app/` | Hardware-verified libxenon demo apps |
| `examples/` | Authoring scripts + their emitted containers |
| `corpus/` | XDK-compiled reference blobs used as test vectors (see provenance in [`docs/license-audit.md`](docs/license-audit.md)) |
| `tests/` | Round-trip, fuzz, parity, golden tests |
| `docs/` | Container format, oracle setup, license audit |
| `PLAN.md` | Full project history, phases, and hardware-verified milestones |

## Credits

- **[Xenia](https://github.com/xenia-project/xenia)** (BSD-3-Clause) — the
  complete Xenos ISA knowledge. `xdis/xenos_ucode.py` is transcribed from
  Xenia's `ucode.h` / shader translator and carries its attribution. Parts of
  Xenia's ucode definitions in turn derive from freedreno (Rob Clark, MIT).
- **[XenosRecomp](https://github.com/hedge-dev/XenosRecomp)** (MIT) — the
  independent validation oracle for ucode semantics and the constant-table
  layout.
- **[Free60 / libxenon](https://github.com/Free60Project/libxenon)** — the
  homebrew library whose Xenos driver consumes our containers; the entire
  reason this toolchain has somewhere to run.
- **GliGli's mupen64-360** — source of the golden corpus blobs that taught us
  the container format, and the project whose XDK-blob situation motivated
  the whole effort.
- **CafeGLSL (Exzap)** — the strategic blueprint.

## License

To be finalized before release — see [`docs/license-audit.md`](docs/license-audit.md)
for the file-by-file provenance audit and the proposed licensing (BSD-3/MIT
for the toolchain; demo ELFs linking libxenon are a separate question).
