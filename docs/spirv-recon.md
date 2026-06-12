# Phase 3 base camp — SPIR-V recon

glslang installed (`tools/glslang/bin/glslang.exe`, main-tot). Two real GLSL
shaders compiled to SPIR-V; their function bodies ARE the translator spec.

## Finding: the instruction surface is tiny and already covered

Every op the two canonical shaders emit maps 1:1 onto a Program-builder
primitive we have ALREADY HARDWARE-PROVEN. The translator is plumbing, not
invention.

### passthrough.vert body
| SPIR-V op | maps to | proven by |
|---|---|---|
| `Load a_position` (Input) | vfetch template | every VS |
| `AccessChain`+`Load` mvp (Uniform member) | constant c0–c3 | xost_vs1 |
| `MatrixTimesVector` | 4× `dp4` oPos.{x,y,z,w} | xost_vs1 (identical!) |
| `Store gl_Position` | oPos export | every VS |
| `Load a_color` / `Store v_color` | interpolator export | xost_vs2 |

### textured.frag body
| SPIR-V op | maps to | proven by |
|---|---|---|
| `Load v_uv`/`v_color` (Input) | interpolator inputs r0/r1 | xost_ps1 |
| `Load u_tex` + `ImageSampleImplicitLod` | `tfetch2D` | xost_ps1 |
| `FMul` | `mul` | combiner "modulate" |
| `Store o_color` | oC0 export | every PS |

## Translator (`xsc/`) design, confirmed by the recon

1. **SPIR-V reader** — header (magic 0x07230203, LE words) + linear
   instruction stream. Need: capabilities (ignore), `EntryPoint`
   (stage + interface var ids), `Decorate`/`MemberDecorate` (Location,
   Binding, BuiltIn, Offset/ColMajor), the type/constant table, `Variable`
   storage classes (Input/Output/Uniform/UniformConstant), and the single
   function body (v1: one function, no calls, no control flow).
2. **Interface resolution** — Input vars → vfetch (VS by Location) or
   interpolator-in (PS by Location); Output → export (gl_Position→oPos via
   BuiltIn Position; Location N→oN / oC0); Uniform block members → float4
   constants by Offset/16; UniformConstant sampler → tfetch fetch-const.
3. **Body lowering** — walk SSA; def-use already explicit (every result is
   an SSA id). Map op→Program-builder call. Op table for 3a:
   `Load/Store/AccessChain` (resolve, no code), `MatrixTimesVector`→4 dp4,
   `VectorTimesMatrix`→4 dp4 (transposed), `FMul/FAdd/FSub`→mul/add/add-neg,
   `Dot`→dp4/dp3, `ExtInst FMix`→ (a-b)*c+b style mad chain (the combiner
   algebra we already wrote!), `Clamp`→ via min/max, `CompositeExtract/
   Construct`→swizzle/mask bookkeeping, `ImageSampleImplicitLod`→tfetch2D.
4. **Register allocation** — SSA temps → r-registers, vec4 granular,
   linear-scan over the (already linear, branch-free) body. Generous: 32+
   GPRs, our shaders use ~6.
5. **Emit** — Program builder → container. Validate via XenosRecomp oracle
   (it does ucode→SPIR-V — the inverse — so round-tripping *meaning* is a
   real cross-check) + on-console.

## Scope notes / v1 exclusions
- No control flow (`OpBranch`/loops/`OpSelect`) — encoder has cjmp/pred
  (feat_* shaders) but lowering structured CF is v2.
- `mat4` assumed ColMajor (glslang default; seen in decorations).
- Builtins beyond Position (PointSize, etc.) ignored in v1.
- gl_PerVertex struct: only member 0 (Position) consumed.

## Exit gate 3a (unchanged)
spincube's VS + a modulate-style FS, written in real GLSL, compile through
xsc and render on console; oracle agrees across the test set.
