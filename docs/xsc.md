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
| `vec * float` (VectorTimesScalar) | mul, scalar `.xxxx` broadcast |
| `mix` / `max` / `min` (GLSL.std.450) | sub+mad / max / min |
| `.rgb` `.xy` (identity-prefix VectorShuffle) | same registers |
| `v.x` (CompositeExtract), `vec4(a,b)` (CompositeConstruct) | masked exports |
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

## Not yet (honest scope)
- Control flow (`if`/`for`) — the cjmp/predication encoders exist
  (examples/author_features.py), structured-CF lowering is v2.
- `dot()`, `clamp`→saturate, non-identity swizzles (`.bgr`), general
  mid-expression CompositeConstruct, integer ops.

## Validation
Every test shader is checked by the XenosRecomp oracle (which does the
inverse ucode→SPIR-V translation); `modulate.frag` is additionally
byte-identical to the hardware-photographed combiner. See
`tests/test_xsc_lower.py`, `tests/test_xsc_resolve.py`. The end-to-end proof
app is `app/glslcube` (cube whose VS+PS compile from GLSL at build time).
