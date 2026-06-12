# Phase 3b recon — next batch of GLSL→Xenos compiler features

Recon-only. **No lowering is implemented here.** Each shader below is authored GLSL,
compiled to SPIR-V with glslang (`-G --target-env opengl`), dumped via `python xsc/spirv.py`,
and analyzed for the SPIR-V opcodes / `GLSL.std.450` ext-instructions it uses that
`xsc/lower.py` does **not** yet handle.

Compile + dump commands used for each shader:

```
tools\glslang\bin\glslang.exe -G --target-env opengl xsc\testdata\<name>.frag -o xsc\testdata\<name>.frag.spv
python xsc\spirv.py xsc\testdata\<name>.frag.spv
```

All four compiled with glslang **rc 0**.

## What xsc already handles (baseline for the "new op" comparison)

From `lower_fragment`'s `operand()` / `_extinst()` dispatch and `docs/xsc.md`:

- **Core opcodes:** `Load`(61), `Store`(62), `AccessChain`(65), `ImageSampleImplicitLod`(87),
  `VectorShuffle`(79) **identity-prefix only** (indices `== range(n)`), `FAdd`(129),
  `FSub`(130), `FMul`(133), `VectorTimesScalar`(142), `Dot`(148), `Constant`(43),
  `ConstantComposite`(44), `CompositeConstruct`(80) **broadcast-of-a-scalar-`OpConstant` only**,
  `CompositeExtract`(81), `ExtInst`(12).
- **GLSL.std.450 ext-instructions:** `FMix`(46), `FMin`(37), `FMax`(40),
  `FClamp`(43) **but only with bounds (0.0, 1.0)** → free saturate modifier.
- **Explicitly NOT yet handled** (per `docs/xsc.md` "Not yet"): non-identity swizzles,
  general mid-expression CompositeConstruct, integer ops, arbitrary (non-[0,1]) clamp.

The four shaders below each target one gap in that list. The transcendental/geometric
builtins (normalize/length/pow) need a multi-instruction decomposition that a parallel
agent owns — **see `docs/builtin-lowering.md`**. This doc only *identifies* which builtins
appear and gives the swizzle/clamp lowerings that live in xsc proper.

Xenos ALU vocabulary referenced below (from `xdis.xenos_ucode.AluVectorOpcode`, used by
`lower.py`): `kAdd`, `kMul`, `kMad`, `kMax`, `kMin`, `kDp3`, `kDp4`. Per-source there is a
free **swizzle** (2 bits/component, the `src{n}_swizzle` field), a free **negate**
(`src{n}_negate`), and the vector op has a free **saturate**/clamp-to-[0,1] modifier
(`vector_clamp`). There is no scalar float immediate — literal constants are pooled into
host-uploaded constant registers (the `lit_manifest`).

---

## 1. sw_swizzle.frag — non-identity VectorShuffle

### GLSL source
```glsl
#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(binding = 0) uniform sampler2D u_tex;
layout(location = 0) out vec4 o_color;

void main()
{
    vec4 tex = texture(u_tex, v_uv);
    o_color = tex.bgra * v_color;
}
```

### SPIR-V function body
```
%9  = Variable [7]
%14 = Load [13]
%18 = Load [17]
%19 = ImageSampleImplicitLod [14, 18]
Store [9, 19]
%22 = Load [9]
%23 = VectorShuffle [22, 22, 2, 1, 0, 3]
%26 = Load [25]
%27 = FMul [23, 26]
Store [21, 27]
Return []
```

### New ops (not yet handled by xsc)
- **`VectorShuffle`(79) with a non-identity index list.** Operands are
  `[vec1, vec2, idx0, idx1, ...]`; here `vec1 == vec2 == %22` (the texture) and the index
  list is `[2, 1, 0, 3]` = `.bgra`. xsc's `operand()` only accepts `VectorShuffle` when the
  indices equal `range(len)` (an identity prefix like `.rgb`/`.xy`); it raises
  `non-identity swizzle (v1)` for any reorder. Every index here is `< 4`, so both sources are
  the same vector — the second-vector (`idx >= width`) case does not arise in this shader.

No other new ops: `FMul`(133) of two vec4 and `ImageSampleImplicitLod`(87) are already handled.

### Proposed Xenos lowering
The shuffle is **free** — it folds into the source-select swizzle of the consuming ALU op.
`tex.bgra` is the texture register read through a per-source swizzle that maps
component-out → component-in: `x←z, y←y, z←x, w←w`. That is exactly the index list
`[2,1,0,3]`, encoded into `src{n}_swizzle` using the same 2-bits-per-component layout the
existing code already uses for `_BCAST_X`/`emit_construct`:

```
swizzle = sum(((idx[p] - p) & 3) << (2*p) for p in range(width))
```

(The field is stored as a *delta* from the natural component, which is why the existing
`_BCAST_X` and `emit_construct` helpers compute `(src - dst) & 3`.) For `[2,1,0,3]` this is
`p0: (2-0)=2, p1:(1-1)=0, p2:(0-2)=-2→2, p3:(3-3)=0` → packed `0b00_10_00_10`.

Implementation sketch: extend `operand()`'s `VectorShuffle` branch so that, when both source
ids are the same vector, it returns the inner operand **plus a swizzle delta tuple**
(the 3-tuple form `(opnd, negate, swizzle)` that `_src`/`_emit` already accept), instead of
raising. The consuming `FMul` then reads `tex` with that swizzle at zero extra instructions.
The two-source case (shuffle across two different vectors) is out of scope for this shader and
would need a second mov; not exercised here.

---

## 2. mathfns.frag — normalize + length + pow + floor

### GLSL source
```glsl
#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(binding = 0) uniform sampler2D u_tex;
layout(location = 0) uniform float u_gamma;
layout(location = 0) out vec4 o_color;

void main()
{
    vec4 tex = texture(u_tex, v_uv);
    vec3 dir = normalize(v_color.rgb);
    float len = length(v_color.rgb);
    vec3 g = pow(tex.rgb, vec3(u_gamma));
    vec2 fl = floor(v_uv);
    o_color = vec4(dir * len + g, fl.x);
}
```

### SPIR-V function body
```
%9  = Variable [7]
%22 = Variable [7]
%29 = Variable [7]
%33 = Variable [7]
%42 = Variable [7]
%14 = Load [13]
%18 = Load [17]
%19 = ImageSampleImplicitLod [14, 18]
Store [9, 19]
%25 = Load [24]
%26 = VectorShuffle [25, 25, 0, 1, 2]          ; v_color.rgb (identity prefix)
%27 = ExtInst [1, 69, 26]                        ; Normalize(v_color.rgb)
Store [22, 27]
%30 = Load [24]
%31 = VectorShuffle [30, 30, 0, 1, 2]          ; v_color.rgb (identity prefix)
%32 = ExtInst [1, 66, 31]                        ; Length(v_color.rgb)
Store [29, 32]
%34 = Load [9]
%35 = VectorShuffle [34, 34, 0, 1, 2]          ; tex.rgb (identity prefix)
%38 = Load [37]                                  ; u_gamma (scalar uniform)
%39 = CompositeConstruct [38, 38, 38]            ; vec3(u_gamma) — non-const broadcast
%40 = ExtInst [1, 26, 35, 39]                    ; Pow(tex.rgb, vec3(gamma))
Store [33, 40]
%43 = Load [17]                                  ; v_uv
%44 = ExtInst [1, 8, 43]                         ; Floor(v_uv)
Store [42, 44]
%47 = Load [22]
%48 = Load [29]
%49 = VectorTimesScalar [47, 48]                 ; dir * len
%50 = Load [33]
%51 = FAdd [49, 50]                              ; + g
%54 = AccessChain [42, 53]                       ; &fl.x
%55 = Load [54]
%56 = CompositeExtract [51, 0]
%57 = CompositeExtract [51, 1]
%58 = CompositeExtract [51, 2]
%59 = CompositeConstruct [56, 57, 58, 55]        ; vec4(rgb, fl.x) — general construct
Store [46, 59]
Return []
```

### New ops (not yet handled by xsc)
- **`ExtInst` GLSL.std.450 `#69` = `Normalize`** (geometric builtin).
- **`ExtInst` GLSL.std.450 `#66` = `Length`** (geometric builtin).
- **`ExtInst` GLSL.std.450 `#26` = `Pow`** (transcendental builtin).
- **`ExtInst` GLSL.std.450 `#8` = `Floor`** (rounding builtin).
- **`CompositeConstruct`(80) broadcasting a *non-constant* scalar** (`%39 = vec3(u_gamma)`,
  where `%38` is a `Load`, not an `OpConstant`). xsc's broadcast path only fires when the
  replicated operand is an `OpConstant`(43); a uniform/varying broadcast is unhandled.
- **`CompositeConstruct`(80) general mid-expression** (`%59 = vec4(s0,s1,s2,s3)` from four
  distinct scalar ids). xsc only constructs at the **output root** via `emit_construct`; a
  general construct feeding a `Store` to the output is structurally the same here (it *is* the
  root), but note the inputs are `CompositeExtract` of an FAdd result plus a loaded scalar —
  `scalar_origin` must resolve a `CompositeExtract` of a *computed* (non-local-variable) value,
  which it currently does not (it handles CompositeExtract, Load-via-AccessChain, and
  Dot/Constant, but `%56..%58` extract from `%51`, an FAdd SSA value).

Already-handled ops present (control): `VectorShuffle`(79) identity `.rgb`, `VectorTimesScalar`(142),
`FAdd`(129), `AccessChain`(65)+`Load`(61), `CompositeExtract`(81).

### Proposed Xenos lowering
- **Normalize(69), Length(66), Pow(26):** transcendental/geometric — decomposition deferred,
  **see `docs/builtin-lowering.md`**. For the record: `length` = `sqrt(dot(v,v))`
  (`kDp3`+scalar `sqrt`/recip-sqrt), `normalize` = `v * rsqrt(dot(v,v))`, and `pow(a,b)` =
  `exp2(b * log2(a))` — all built from the Xenos scalar transcendental unit (`logc`/`expc`/
  `rsqc` etc.), owned by the parallel builtin agent.
- **Floor(8):** maps to the Xenos `kFloor` vector ALU op (a native vector opcode in
  `AluVectorOpcode`) — one instruction, no constant pool. This one is cheap and could land in
  xsc proper independently of the transcendental work; record it on the worklist as a simple
  1:1 ALU op.
- **Non-constant broadcast CompositeConstruct (`vec3(u_gamma)`):** lower as a read of the
  scalar's register through the broadcast swizzle `.xxx` (the existing `_BCAST_X` delta) — no
  new instruction, it folds into Pow's source select. Generalizes the current
  constant-only broadcast to any scalar operand.
- **General CompositeConstruct at root (`vec4(rgb, fl.x)`):** the existing `emit_construct`
  already lays out per-component runs into masked exports; the only gap is that
  `scalar_origin` must learn to resolve `CompositeExtract(computed_vec, i)` by emitting/locating
  the producing value in a register and routing component `i` through the swizzle. This is an
  extension of `scalar_origin`, not a new ALU op.

---

## 3. clamprange.frag — arbitrary-bounds clamp

### GLSL source
```glsl
#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(binding = 0) uniform sampler2D u_tex;
layout(location = 0) out vec4 o_color;

void main()
{
    vec4 tex = texture(u_tex, v_uv);
    o_color = clamp(tex * v_color, 0.2, 0.8);
}
```

### SPIR-V function body
```
%9  = Variable [7]
%14 = Load [13]
%18 = Load [17]
%19 = ImageSampleImplicitLod [14, 18]
Store [9, 19]
%22 = Load [9]
%25 = Load [24]
%26 = FMul [22, 25]                              ; tex * v_color
%29 = CompositeConstruct [27, 27, 27, 27]        ; vec4(0.2) — lo
%30 = CompositeConstruct [28, 28, 28, 28]        ; vec4(0.8) — hi
%31 = ExtInst [1, 43, 26, 29, 30]                ; FClamp(x, lo, hi)
Store [21, 31]
Return []
```

### New ops (not yet handled by xsc)
- **`ExtInst` GLSL.std.450 `#43` = `FClamp` with non-[0,1] bounds.** The op itself *is* in
  the dispatch, but `_extinst`'s FClamp branch raises `v1 clamp supports only [0,1]` unless
  `lo == 0.0 and hi == 1.0`. Here `%27 = 0.2` and `%28 = 0.8` (each broadcast through a
  `CompositeConstruct` that `_const_scalar` already unwraps), so the bounds resolve fine — only
  the [0,1] guard rejects it.

No other new ops: `FMul`(133) and the broadcast `CompositeConstruct`(80) of `OpConstant` are
handled.

### Proposed Xenos lowering
`clamp(x, lo, hi)` = `min(max(x, lo), hi)`. Two ALU ops using the existing `kMax`/`kMin`:

```
max  rT, x, c_lo     ; kMax, x and the pooled-literal lo register
min  oC0, rT, c_hi   ; kMin, result and the pooled-literal hi register, fused to export
```

`lo` (0.2) and `hi` (0.8) each become **pooled literal constants** via the existing
`_literal_reg((v,v,v,v))` path (the host uploads them per the manifest, exactly like the
current `mix` 0.5 case). Note the existing one-constant-per-instruction rule in `_emit`
(`>1 constant operand` raises) is satisfied because `x` is a register and each of `kMax`/`kMin`
touches only one literal register. Implementation: in `_extinst`'s FClamp branch, when bounds
are not (0,1), pool `lo`/`hi` as `vec4` literals and emit `kMax` then `kMin` (the second fused
to the output export when this is the root), instead of raising.

---

## 4. absign.frag — abs + sign + min/max on a uniform

### GLSL source
```glsl
#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(binding = 0) uniform sampler2D u_tex;
layout(location = 0) uniform vec4 u_offset;
layout(location = 0) out vec4 o_color;

void main()
{
    vec4 tex = texture(u_tex, v_uv);
    vec4 mag = abs(u_offset);
    vec4 dir = sign(u_offset);
    vec4 lo = min(tex, v_color);
    vec4 hi = max(tex + mag * dir, lo);
    o_color = hi;
}
```

### SPIR-V function body
```
%9  = Variable [7]
%20 = Variable [7]
%25 = Variable [7]
%28 = Variable [7]
%34 = Variable [7]
%14 = Load [13]
%18 = Load [17]
%19 = ImageSampleImplicitLod [14, 18]
Store [9, 19]
%23 = Load [22]                                  ; u_offset
%24 = ExtInst [1, 4, 23]                          ; FAbs(u_offset)
Store [20, 24]
%26 = Load [22]                                  ; u_offset
%27 = ExtInst [1, 6, 26]                          ; FSign(u_offset)
Store [25, 27]
%29 = Load [9]
%32 = Load [31]
%33 = ExtInst [1, 37, 29, 32]                     ; FMin(tex, v_color)
Store [28, 33]
%35 = Load [9]
%36 = Load [20]
%37 = Load [25]
%38 = FMul [36, 37]                               ; mag * dir
%39 = FAdd [35, 38]                               ; tex + (mag*dir)
%40 = Load [28]
%41 = ExtInst [1, 40, 39, 40]                     ; FMax(..., lo)
Store [34, 41]
%44 = Load [34]
Store [43, 44]
Return []
```

### New ops (not yet handled by xsc)
- **`ExtInst` GLSL.std.450 `#4` = `FAbs`** (`abs(u_offset)`).
- **`ExtInst` GLSL.std.450 `#6` = `FSign`** (`sign(u_offset)`).

Already-handled ops present (control / sanity): `ExtInst #37 = FMin` and `#40 = FMax`
(both in `_extinst`), `FMul`(133), `FAdd`(129). Note the final `Store [34,...]` then
`Load [34]` then `Store [43,...]` is a function-local `hi` forwarded to the output — the
store-forwarding case xsc already supports.

### Proposed Xenos lowering
- **FAbs(4):** the Xenos ALU source modifier set includes a per-source **absolute-value**
  input modifier (the `src_abs` / input-modifier field on ALU sources). `abs(x)` therefore
  folds **free** into the consuming op's source select — read `u_offset` with the abs modifier
  set, no extra instruction. If a standalone value is needed (here it feeds the `FMul`), set
  abs on that `FMul`'s source. Fallback if the abs input-modifier is not wired in the builder:
  `max(x, -x)` via `kMax` with `src2_negate` (the negate field already exists in `_src`).
- **FSign(6):** no single Xenos op. Lower as the difference of two set-greater-than results,
  i.e. `sign(x) = (x>0) - (x<0)`. On Xenos that is the `kSetgtFloat`-style vector predicates:
  `gt = kSetgt(x, 0); lt = kSetgt(0, x); sign = gt - lt` (kSetgt returns 1.0/0.0 per
  component). `0.0` is a pooled literal (or, once 0.0/1.0 source-swizzle folding lands, free).
  Three ALU ops; record as a small fixed sequence on the worklist. (If a `kSign`/`kCndGe`
  trick is available in the ucode vocabulary it can collapse this — flag for implementer to
  check `AluVectorOpcode`.)
- **FMin(37)/FMax(40):** already lowered to `kMin`/`kMax` — present here only as controls.

---

## Consolidated worklist

Every NEW op observed across the four shaders, with its proposed lowering. "free" = folds
into an existing instruction's source-select/modifier with no added ALU op.

| # | New op (SPIR-V / GLSL.std.450) | Appears in | Proposed Xenos lowering |
|---|---|---|---|
| 1 | `VectorShuffle`(79), non-identity, single source | sw_swizzle | **free** — fold index list into `src_swizzle` delta `((idx[p]-p)&3)<<(2p)`; return 3-tuple `(opnd,neg,swz)` from `operand()` instead of raising |
| 2 | `ExtInst #8 Floor` | mathfns | 1:1 → native `kFloor` vector ALU op (cheap, can land independently) |
| 3 | `ExtInst #69 Normalize` | mathfns | `v * rsqrt(dot(v,v))` — **defer to `docs/builtin-lowering.md`** |
| 4 | `ExtInst #66 Length` | mathfns | `sqrt(dot(v,v))` — **defer to `docs/builtin-lowering.md`** |
| 5 | `ExtInst #26 Pow` | mathfns | `exp2(b*log2(a))` — **defer to `docs/builtin-lowering.md`** |
| 6 | `CompositeConstruct`(80) broadcast of a **non-constant** scalar | mathfns | **free** — read scalar reg through `.xxx` broadcast swizzle (`_BCAST_X` delta); generalize the constant-only broadcast |
| 7 | `CompositeConstruct`(80) **general** `vec4(s0,s1,s2,s3)` w/ `CompositeExtract` of a computed value | mathfns | extend `emit_construct`/`scalar_origin` to resolve `CompositeExtract(computed_vec,i)` into a reg+swizzle; no new ALU op |
| 8 | `ExtInst #43 FClamp` with **non-[0,1]** bounds | clamprange | `min(max(x,lo),hi)` via `kMax`+`kMin`; `lo`/`hi` as pooled `vec4` literals (manifest upload), second op fused to export |
| 9 | `ExtInst #4 FAbs` | absign | **free** — per-source `abs` input modifier; fallback `max(x,-x)` via `kMax`+negate |
| 10 | `ExtInst #6 FSign` | absign | `(x>0)-(x<0)` via two `kSetgt` + `kAdd`-neg; `0.0` pooled literal — ~3 ALU ops (impl: check `AluVectorOpcode` for a cheaper `kSign`/`kCnd`) |

Notes for the implementer:
- Items **1, 2, 6, 8, 9** are self-contained and do not depend on the transcendental work;
  they can land first. Item **7** is a `scalar_origin` extension. Items **3/4/5** wait on
  `docs/builtin-lowering.md`.
- `FMin`(37)/`FMax`(40) were included in absign/(via min/max) purely as already-handled
  controls — no action needed.
- Verified GLSL.std.450 numbers against the SPIR-V actually emitted by glslang:
  4=FAbs, 6=FSign, 8=Floor, 26=Pow, 37=FMin, 40=FMax, 43=FClamp, 66=Length, 69=Normalize
  (and 46=FMix from the existing baseline). No `UNKNOWN op` was encountered.
