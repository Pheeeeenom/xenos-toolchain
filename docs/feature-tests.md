# Xenos ISA feature micro-tests

Three pixel shaders exercising ISA features the toolchain can encode but has
never proven on hardware. Authored by `examples/author_features.py`, which is
byte-deterministic (re-runs produce identical .bin/.hlsl). All three pass the
XenosRecomp oracle (`XenosRecomp.exe <bin> <hlsl> shader_common.h`, exit 0)
and their HLSL bodies match the intended semantics.

All three use the same interpolator contract as `xost_ps1` (r0 = TEXCOORD0,
r1 = COLOR0), so the existing textured vertex shader `xost_vs2.bin` drives
them unchanged.

| shader | feature | ucode slots | container |
|---|---|---|---|
| `feat_cjmp.bin` | bool constant + conditional jump CF | 5 | 320 B |
| `feat_coissue.bin` | vector+scalar co-issue in one ALU slot | 4 | 220 B |
| `feat_pred.bin` | setp + ALU-level predication (stretch) | 6 | 332 B |

---

## 1. feat_cjmp — boolean constant + conditional jump

**Purpose.** Prove `kCondJmp` control flow steered by a D3D boolean constant.
First open-toolchain shader with non-straight-line CF. CF is hand-built via
`enc.build_cf_cond_jmp` + `pack_program` (the Program builder cannot express
jumps); shape is modeled directly on corpus `ps_fb0.bin`.

**Constants.** `prim` = c0 (float4), `env` = c1 (float4),
`use_prim` = RS_BOOL register 0. Pixel-shader bools live at 128+ in the
unified D3D bool space, so RS_BOOL reg 0 appears in ucode as **b128**
(confirmed against ps_fb0's `use_tex`).

**Disassembly** (`python xdis/xdis.py examples/out/feat_cjmp.bin`):

```
/*    0.0 */       cjmp !b128, L3
/*    0.1 */       alloc colors
/*    1.0 */       exece
/*    3   */          max oC0, c0, c0
                label L3
/*    1.1 */       alloc colors
/*    2.0 */       exece
/*    4   */          max oC0, c1, c1
/*    2.1 */       cnop
```

**XenosRecomp HLSL** (exit 0; excerpt of main body):

```hlsl
if (b128 != 0)
{
    oC0.xyzw = max(prim.xyzw, prim.xyzw);
    ... return;
}
oC0.xyzw = max(env.xyzw, env.xyzw);
... return;
```

Exactly the expected `if (b128) prim else env` structure.

**Field choices that needed care.**
- cjmp word copied field-for-field from ps_fb0 cf0 `(0x4, 0xb200)`, which the
  script first re-builds byte-identically as an inversion proof:
  `address=4, direction=0, bool_address=128, condition=False,
  addressing_mode=0, is_unconditional=False, is_predicated=False`.
  Only the target was changed (4 → 3) for our shorter arm A.
- **Polarity:** `condition=False` means *jump when the bool is false*
  (disasm `cjmp !b128`). So the fall-through arm (A) runs when b128 is TRUE
  — PRIM must therefore be arm A.
- **Address semantics:** the 13-bit `address` is an absolute CF entry index
  (label `L3` = cf entry 3, the second `alloc`). `direction=0` with
  `addressing_mode=0` is what the XDK emits for this forward jump; whether
  `direction` is ever nonzero for forward jumps is untested
  (TODO(ambiguity) — no backward-jump corpus sample exists).
- Each arm needs its own `alloc colors` + `exece`; the jump target is the
  arm-B `alloc`, mirroring ps_fb0 exactly.

**Console test.** Draw the spinning cube with `xost_vs2` + `feat_cjmp`, set
`prim`/`env` to two loud colors (e.g. red/blue), and toggle
`Xe_SetPixelShaderConstantB(0, v)` every ~60 frames.
- PASS: the whole cube flips solid PRIM-color ↔ solid ENV-color with the
  toggle (true → PRIM).
- FAIL modes: stuck on one color (bool constant not reaching the SQ, or
  wrong b-register base); inverted mapping (condition polarity wrong);
  garbage/hang (cjmp address or direction field wrong — the sequencer
  jumping outside the CF program).

## 2. feat_coissue — scalar co-issue

**Purpose.** Prove that one 96-bit ALU slot can retire a vector op and a
scalar op to different write masks of the same export:
`mul oC0.xyz_` (vector pipe) + `maxs oC0.___w` (scalar pipe). The scalar
fields are copied **verbatim** from corpus `cube_ps.bin` slot 4
(`scalar_opcode=kMaxs(5)`, `scalar_write_mask=0x8`, src3 swizzle 27 = `.ww`),
with sources remapped to our contract r2 = texel, r1 = shade.

**Disassembly:**

```
/*    0.0 */       exec
/*    2   */          tfetch2D r2, r0.xy, tf0
/*    0.1 */       alloc colors
/*    1.0 */       exece
/*    3   */          mul oC0.xyz_, r2.xyzz, r1.xyzz
              +       maxs oC0.___w, r2.ww
/*    1.1 */       cnop
```

The `+` continuation line is the co-issue marker.

**XenosRecomp HLSL** (exit 0; excerpt):

```hlsl
r2.xyzw = tfetch2D(tex0_..., r0.xy, float2(0, 0)).xyzw;
oC0.xyz = r2.xyz * r1.xyz;
ps = max(r2.w, r2.w);
oC0.w = ps;
```

rgb from the vector pipe, w from the scalar pipe — two independent writes
from one instruction, as required.

**Field choices that needed care.**
- The scalar operand is **src3** (`src3_reg=2`, swizzle 27 = wwww, printed
  `.ww` because scalar ops read two components); vector ops use src1/src2.
- Vector and scalar write masks must be disjoint subsets of the same export
  (`0x7` + `0x8`); both target `vector_dest=0`/`scalar_dest=0` = oC0 with
  `is_export=True`.
- src1/src2 swizzle 192 (`xyzz`) kept from the cube_ps template — harmless
  under the xyz mask and keeps the instruction as close to the golden sample
  as possible.

**Console test.** Textured cube (alpha-gradient texture, e.g. alpha ramping
left→right within the texture), `shade` from vertex color, with alpha
blending ON against a mid-grey background.
- PASS: rgb shows texel*shade modulation AND transparency follows the
  texture's alpha ramp (proves the scalar pipe delivered texel alpha to
  oC0.w).
- FAIL modes: opaque or uniformly-blended cube (scalar co-issue write to
  oC0.w never landed — w would be whatever the vector pipe left, likely
  garbage from the unwritten lane); rgb wrong but alpha right (vector mask
  clobbered by scalar fields).

## 3. feat_pred — setp + predicated ALUs (stretch — built)

**Purpose.** Prove the predicate register path: a `setp_ne` scalar op sets
p0 from float constant c2.w, then two ALUs in the *same* exec block execute
under `(p0)` / `(!p0)` predication, selecting prim vs env into r3.
No corpus sample contains setp — the operand convention (scalar pipe reads
src3, both write masks 0) is decoder/xenia-derived and oracle-validated only.
That residual uncertainty is exactly what the console test settles.
`TODO(ambiguity)`: whether real hardware lets predicated ALUs *within the
same exec* observe a p0 written earlier in that exec (XenosRecomp models it
sequentially; the XDK compiler's own placement habits are unverified). The
exec uses `is_predicate_clean=True` like every corpus exec — believed to
mean "reset p0 at exec start", which is correct here since setp runs first.

**Constants.** `prim` = c0, `env` = c1, `pred_src` = c2 (predicate driven by
`pred_src.w != 0`, set via `Xe_SetPixelShaderConstantF(2, ...)`).

**Disassembly:**

```
/*    0.0 */       exec
/*    2   */          setp_ne r0._, c2.w
/*    3   */     (p0) max r3, c0, c0
/*    4   */    (!p0) max r3, c1, c1
/*    0.1 */       alloc colors
/*    1.0 */       exece
/*    5   */          max oC0, r3, r3
/*    1.1 */       cnop
```

**XenosRecomp HLSL** (exit 0; excerpt):

```hlsl
p0 = pred_src.w != 0.0;
ps = p0 ? 0.0 : 1.0;
if (p0)
{
    r3.xyzw = max(prim.xyzw, prim.xyzw);
}
if (!p0)
{
    r3.xyzw = max(env.xyzw, env.xyzw);
}
oC0.xyzw = max(r3.xyzw, r3.xyzw);
```

**Field choices that needed care.**
- setp_ne occupies the scalar pipe of an otherwise idle instruction: vector
  pipe is template kMax with `vector_write_mask=0`, `scalar_write_mask=0`
  (only p0 is updated; disasm shows the dead `r0._` destination).
- Predication is per-ALU (`is_predicated=True` + `predicate_condition`),
  not CF-level — no `kCondExecPred` needed, so the PixelProgram builder
  could express the whole shader.
- Both predicated movs write the temp r3; the final unconditional `exece`
  exports it. This avoids the open question of what happens when a
  `kCondExecPredEnd` is skipped and the CF program runs off the end.

**Console test.** Same harness as feat_cjmp but toggle
`Xe_SetPixelShaderConstantF(2, {0,0,0, 0.0 or 1.0})`.
- PASS: cube flips PRIM (w=1.0) ↔ ENV (w=0.0).
- FAIL modes: always ENV (setp never fired / p0 not visible to later ALUs
  in the same exec — would mean predicated uses must move to a later exec);
  always PRIM or flicker (predicate_condition polarity or
  is_predicate_clean semantics wrong); both-arms-execute artifacts can't
  occur here since both write the full r3 mask — last writer (!p0 env)
  would win, showing as "always ENV".

## Validation summary

For each shader: (a) clean `xdis` disassembly above; (b)
`XenosRecomp.exe <bin> <hlsl> reference\XenosRecomp\XenosRecomp\shader_common.h`
exit 0 with HLSL matching intended semantics; (c) two consecutive runs of
`python examples/author_features.py` produce identical MD5s for all six
outputs. The script also re-proves the ps_fb0 cjmp CF word
(`build_cf_cond_jmp(...) == (0x4, 0xb200)`) before every build.

## Hardware results

**Boolean constants and real loops: PASS on console (2026-10-10).** Proven
through the xsdk homebrew SDK (the sibling repo this toolchain is being
merged with), not libxenon, by its `tests/probe_constants.c` and
`tests/test_constants.c`. The shaders were packed with `xasm.encode` and
cross-checked byte-for-byte against `xasm.textasm`'s assembly of the same
program. Each step drew a quad, resolved it, and read its pixels back:

- `cexec b128` (pixel shader boolean 0 in D3D's split) skipped its block
  with the boolean clear and ran it with the boolean set.
- `loop i0` / `endloop i0` ran 0, 1 and 3 passes as loop constant 0 said,
  set per draw. With 0 passes, LOOP_START skipped straight past LOOP_END.
- `mad oPos, r1, c0, c1`: two float constants in one ALU instruction.
- Six quads a frame for 60 frames, each placed and coloured only by its
  vertex, pixel, boolean and loop constants, all right on every frame.

**The constant-write rule found on the way.** Constants must go to the GPU
the way libxenon's `Xe_pUploadALUConstants` / `Xe_pUploadIntegerConstants`
send them. Floats go 16 constants at a time, and all 40 boolean + loop dwords
(0x4900-0x4927) go in one packet, each packet padded with type-2 NOPs so its
values start on a 16-byte ring boundary (`Xe_pAlign`). Writing only what
changed hung the GPU on the first draw, even when aligned: three float
constants, one boolean dword and one loop dword. Which of those partial
writes is the culprit isn't isolated yet. Anything driving the GPU without
libxenon must keep whole-block uploads.

**Still unproven:** `cjmp` (feat_cjmp), scalar co-issue (feat_coissue) and
predication (feat_pred).
