# Known issues / cleanup (xost)

All compiler bugs from the parallel GLES2 batch are fixed and **hardware-verified
on the RGH console**. One optional cleanup remains.

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
