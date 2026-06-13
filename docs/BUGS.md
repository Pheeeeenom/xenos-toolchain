# Known issues / cleanup (xost)

All compiler bugs from the parallel GLES2 batch are fixed and **hardware-verified
on the RGH console**. One optional cleanup remains.

## Phase C loops — landed in code, AWAITING HARDWARE (gallery G29/G30)
Two new ISA primitives, both decode-valid (XenosRecomp) + round-trip-asserted,
**not yet confirmed on console**:

- **aL-relative constant addressing** `c[base + aL]` (C1). `build_alu` gained
  per-source `srcN_const_rel` flags that fold onto the two hardware addressing
  bits via the decoder's `src_const_is_addressed(i)` rule (aL, not a0). Proof
  slot **G29 ALCONST**: `for(aL=0;aL<4;aL++) r2 += c[0+aL]` over c0..c3.
  Expected on screen: **light grey** (sum of four distinct constants). Pure
  **red** = aL addressing fell back to a fixed c0 (broken). A non-red, non-grey
  hue (e.g. cyan-ish) = aL relative works but the aL sequence is off-by-one.

- **Data-dependent `while`-break** via a predicated LOOP_END (C2).
  `program.py end_loop(predicated_break=, break_condition=)` threads the
  `is_predicated_break`/`condition` bits; a body `setp_*` sets p0. Proof slot
  **G30 WHILEBRK**: loop capped at 16, breaks when the accumulator reaches .25.
  Expected: **25% grey** (broke at pass 4). **White** = the break was ignored
  (ran all 16) — the predicate-clean bug below, or a condition-polarity error.

### Predicate-clean bug (found by adversarial review, FIXED)
`program.py` hardcoded `is_predicate_clean=True` on every exec. An exec that
runs a `setp_*` must be `is_predicate_clean=False`, or the sequencer treats p0
as clean and does not wait for the write before the predicated LOOP_END reads
it — so the break never fires (white instead of grey). Fixed: `_build_ucode`
now sets `is_predicate_clean=False` for any exec containing a predicate-writing
scalar op (`_PRED_WRITE_SCALAR_OPS`). Existing setp-free shaders stay
predicate-clean → byte-identical. Locked by `tests/test_program_loop.py`.

The GLSL frontend for these (generalizing `lower_fragment_real_loop` to lower
`arr[i]` and `while(cond) break` from glslang SPIR-V) is the next increment —
deferred until the primitives are console-confirmed.

## Resolved (hardware-verified)
- **vec*mat / mat*mat transpose** — fixed (mad-chain for `v*M`; per-row `vec*mat`
  for the product matrix).
- **cube-fetch encoding** — fixed (`cube .zzxy/.yxz` → `rcp(abs(MA))` →
  `mad …,1.5` → `tfetchCube .yxw`, per xenia).
- **GLSLLOOP interpolator leak (G28)** — fixed: the real-loop PS declares 2
  interpolators (matching the cube VS) with the accumulator in r2.
- **samplerCube on hardware (G27)** — works: 6 distinct faces render by
  direction (green/cyan/yellow/… sides).

## Optional future cleanup — a real libxenon cube-texture API
The cubemap currently works via an **app-side fetch-constant poke** in
`app/glslgallery/source/main.c`: after `Xe_SetTexture`, promote texture unit 0's
fetch dword5 dimension `2D(1)→Cube(3)` (set bit 10) and dword2 `stack_depth=5`.
The `Xe_CreateTexture(64, 64*6)` allocation already has the correct 16384-byte
per-face slice stride, so no layout change was needed. This is a pragmatic
in-place patch.

A cleaner long-term fix is a real libxenon cube API (`is_cube` on `XenosSurface`
+ a cube branch in `TEXTURE_FETCH` setting the dimension/stack fields) — specced
in this file's history. Only worth the libxenon rebuild if cube textures become
common; until then the app-side poke is fine and self-contained.
