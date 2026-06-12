# xsc — GLSL → Xenos shader compiler

Pipeline: **GLSL → glslang → SPIR-V → xsc → Xenos container**. glslang
(Khronos reference compiler) does parsing/semantics/optimization; xsc lowers
the SPIR-V to microcode through the hardware-proven Program builder. The
Mesa/NIR optimizer is a someday option; the runtime combiner case is already
served by `xcomb`.

## Use

```
python -m xsc.lower shader.vert.spv -o shader.bin     # SPIR-V -> container
```
Library: `lower(module) -> bytes`, `lower_ex(module) -> (bytes, manifest)`.
Get the SPIR-V with: `glslang -G --target-env opengl shader.frag -o out.spv`.

## Supported (v1, branch-free)

| GLSL / SPIR-V | Xenos lowering |
|---|---|
| vertex `mvp * pos` (MatrixTimesVector) | 4× `dp4` → oPos |
| passthrough varyings | interpolator exports (max o*N*) |
| `texture()` (ImageSampleImplicitLod) | `tfetch2D` |
| `*` `+` `-` (FMul/FAdd/FSub) | mul / add / add-neg |
| `/` (FDiv) | `a · rcp(b)`; scalar-pipe `rcp` per component (scalar divisor: once + broadcast) |
| `mod()` (OpFMod) | `a − b·floor(a/b)` |
| `vec * float` (VectorTimesScalar) | mul, scalar `.xxxx` broadcast |
| `dot()` / `cross()` | `dp3`/`dp4` broadcast / yzx·zxy mad |
| `mix` / `max` / `min` / `clamp(x,0,1)` (GLSL.std.450) | sub+mad / max / min / `_sat` modifier (free) |
| `step` / `smoothstep` | `kSge` / `t·t·(3−2t)`, `t=sat((x−e0)/(e1−e0))` |
| `normalize` / `length` / `distance` | dp3·rsq·mul / dp+sqrt / sub+dp+sqrt (scalar pipe) |
| `sqrt` `inversesqrt` `exp2` `log2` `exp` `log` | scalar-pipe `kSqrt`/`kRsq`/`kExp`/`kLog` (+const for natural) |
| `pow(x,y)` | per-component `exp2(y·log2 x)` (scalar pipe) |
| `reflect(i,n)` | `i - 2·dot(n,i)·n` (dp + mad) |
| `sin` / `cos` / `tan` | scalar `kSin`/`kCos`; `tan = sin·rcp(cos)` — **range-limited ~[-π,π]** (see below) |
| `asin` / `acos` / `atan` | polynomial approx (A-S sqrt form / π/2−asin / odd poly); ~7e-5 over [-1,1], atan range [-1,1] (see below) |
| `.rgb` `.xy` (identity) / `.bgra` (reorder) / two-source `vec4(a.xy,b.zw)` | same regs / swizzled mov / per-component masked movs |
| `v.x` (CompositeExtract), `vec4(a,b,c,d)` (CompositeConstruct) | masked exports |
| `floor` / `ceil` / `trunc` / `round` / `fract` | `kFloor` / `-floor(-x)` / `kTrunc` / `floor(x+.5)` / `kFrc` |
| `abs()` / `sign()` | `max(x,-x)` / `sgt(x,0)−sgt(0,x)` (pooled `0.0`) |
| `radians` / `degrees` | mul by pooled `π/180` / `180/π` |
| `clamp(x, lo, hi)` arbitrary | `min(max(x,lo),hi)`, lo/hi pooled literals |
| **`if/else`**, incl. **nested**, assigning the output | branch-free chained `cndgt`/`cndge`/`cndeq` selects |
| **`for`/`while`** with a compile-time trip count, incl. **nested** | unrolled to branch-free SSA; induction variable constant-folded (see below) |
| `vec4 t = ...;` (function-local) | store-forwarding |
| uniforms (block + bare), samplers | constant regs / fetch consts |
| float / vec literals (0.5, vec4(...)) | constant pool (see below) |

## Interface conventions (GLSL ↔ Xenos)

- VS inputs → vfetch in Location order; vertex-element usage by location:
  0=POSITION, 1=COLOR, 2=TEXCOORD, 3+=TEXCOORD(idx++). Host vertex format
  must match. VS output BuiltIn Position → oPos; Location *N* → interpolator
  o*N*. FS input Location *N* → register r*N*; output Location 0 → oC0.
- Uniform block members / bare uniforms → constant registers (block by
  Offset/16; bare by Location). Scalar uniforms read from the register's
  `.x`. Samplers (UniformConstant) → tfetch fetch-constant = Binding.

## Literal constant pool (the host-upload contract)

The Xenos has no ALU float immediates, so GLSL literals are allocated
constant registers **after** the uniform-declared constants.
`lower_ex()` returns a **manifest** `[(register, (f0,f1,f2,f3))]`; the host
must `Xe_SetPixelShaderConstantF` those values before drawing, exactly as it
sets uniforms. Example: `mix(tex, color, 0.5)` →
`add r3, color, -tex; mad oC0, r3, c0, tex` with manifest
`[(0, (0.5,0.5,0.5,0.5))]` — upload 0.5 to c0. (Future: fold 0.0/1.0 into the
ALU source-select swizzle, which is free and needs no register.)

## Scalar pipe (the transcendentals)

Xenos `rsq`/`sqrt`/`exp2`/`log2`/`sin`/`cos` are **scalar-pipe** ops: the ALU
instruction's vector half is a nop (`vector_write_mask=0`) and the scalar
half reads one component of src3 and broadcasts its result. `normalize`,
`length`, `pow`, etc. compose these with `dp3`/`dp4` and `mul`. No common-
subexpression elimination yet, so `normalize(v)` + `length(v)` recomputes
`dot(v,v)` twice — correct, slightly redundant.

## Approximate builtins (documented accuracy / range)
- **`sin`/`cos`/`tan`** — the bare hardware `kSin`/`kCos` are accurate only
  near `[-π,π]`; arbitrary arguments need a `frac`-based range-reduction
  prologue (math in `docs/builtin-lowering.md`). `tan = sin·rcp(cos)`.
- **`asin`/`acos`** — Abramowitz-Stegun sqrt form, max error ~7e-5 over
  `[-1,1]`. `acos = π/2 − asin`.
- **`atan`** — odd polynomial, accurate on `[-1,1]` (no range reduction for
  `|x|>1` yet; `atan2`/wide-range is a refinement).

## Register budget (16 GPRs)

Xenos pixel shaders have a hard **16-GPR budget** (r0..r15) — exceeding it is a
silent GPU **hang** on hardware (oracle can't see this). `xasm/program.py` runs
a **linear-scan register allocator** in `PixelProgram.finish()` whenever a
shader needs >16 registers: it recycles dead temporaries (live range = first
def → last use), pins the read-only interpolator inputs, and packs everything
into the minimum register count (e.g. `invtrig` 33→6, `mathfns` 20→6). Shaders
already within budget are left byte-for-byte unchanged. A self-check asserts no
two simultaneously-live temps share a register.

## Loops (unrolling + mem2reg)

glslang (`-G --target-env opengl`) lowers `for`/`while` to mutable-memory form:
an `OpLoopMerge` header, loop-carried values held in Function-storage
`OpVariable`s mutated by `OpStore`, and an integer induction variable. xsc's
`unroll.py` runs as a pre-pass **whenever a module contains a loop** and rewrites
the whole function body into flat, branch-free SSA by symbolically executing the
structured CFG:

- **Trip count** is recovered from the induction variable's constant
  init / comparison / step and the body is replicated that many times. The
  induction variable is **constant-folded**, so `float(i)`, `i*k`,
  `float(i)*0.2`, etc. become literals (no integer ALU reaches the GPU).
- **mem2reg**: every Function-local variable is threaded through an environment,
  so loop-carried accumulation *and* post-loop read-modify-write (`c *= k;`)
  resolve to SSA values — the lowering's single-store-per-variable model never
  sees the multiple stores.
- **Nested** loops expand innermost-first; **`while`** (increment in the body)
  and **`for`** (increment in the continue block) are both handled.

Because the pass is a no-op when there is no loop, loop-free shaders still lower
byte-for-byte as before. Unrolled shaders then go through the normal pipeline,
including the 16-GPR register allocator (e.g. `loop_stair` 17→3 GPRs). See
`loop_*` in `xsc/testdata/` and the `LOOP` gallery slot.

**Still the frontier:**
- **Data-dependent loops** — a non-constant trip count (uniform/variable bound)
  can't be unrolled; `unroll.py` raises `UnrollError` rather than miscompile.
  These need real Xenos loop control-flow (`kLoopStart`/`kLoopEnd`, encoded in
  xost but **unproven on hardware**), or a `[[unroll]]`-style bounded fallback.
- **`if/else` inside a loop body** — branchy loop bodies aren't unrolled yet
  (the body must be straight-line); also rejected with `UnrollError`.

## Not yet (the frontier)
- **Integer ops** (`int`/`uint` arithmetic, bit ops) — Xenos ALU is float-
  centric; awkward, rarely needed in fragment work.
- **Matrix math beyond `mvp*pos`** — `mat*mat`, `vec*mat`, `inverse`,
  `determinant`.
- Folding `0.0`/`1.0` literals into the free source-select swizzle (they
  currently use a pooled constant register).

## Validation
Every test shader is checked by the XenosRecomp oracle (which does the
inverse ucode→SPIR-V translation); `modulate.frag` is additionally
byte-identical to the hardware-photographed combiner. See
`tests/test_xsc_lower.py`, `tests/test_xsc_resolve.py`. The end-to-end proof
app is `app/glslcube` (cube whose VS+PS compile from GLSL at build time).
