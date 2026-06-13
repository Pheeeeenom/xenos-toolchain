# Next steps — parallel roadmap (fan out with subagents)

> Read [HANDOFF.md](HANDOFF.md) first (invariants + build/verify workflow).
> This is a **batch of independent tasks meant to be run in parallel.** Spawn
> one subagent per task. The more that runs concurrently, the better — but
> follow the integration protocol below, because most tasks touch `xsc/lower.py`
> and naive parallel edits will collide.

## Parallelization protocol (do this, not a free-for-all)

1. **One subagent per task, each on its own git branch / worktree.** Each agent
   takes a task end-to-end: implementation + a `xsc/testdata/` shader + an
   oracle-validated test in `test_xsc_lower.py` + (if the task adds a new GPU
   op/fetch) a `glslgallery` slot.
2. **Integrate sequentially, not all at once.** Merge one branch at a time. After
   **every** merge run `python tests/test_xsc_lower.py` and confirm
   **`modulate.frag` is still byte-identical** — that's the canary that a merge
   broke shared lowering. Then run the full suite.
3. **Conflicts cluster in two spots** and are mechanical to resolve:
   `ValueLowerer.operand`'s `elif op == …` dispatch chain, and the op tables in
   `lower_fragment`. Resolve by keeping all the new branches.
4. **Do task E (loops) last / on its own.** It has the broadest `lower.py`
   footprint and will fight every other branch. Land the small ones first, then
   E against the integrated result.
5. **Hardware-gated tasks** (marked HW below) add a gallery slot but are NOT done
   until verified on the maintainer's console — the XenosRecomp oracle validates
   *decode, not execution*. Build one combined `glslgallery.elf32` after
   integration; the maintainer flashes it once and reports per-slot. Mark such
   features "console-pending" in the commit until confirmed.

## Tasks (independent unless noted)

### A — Vertex literal constants  *(no HW gate; unblocks lighting VS)*
`lower_vertex` builds `ValueLowerer(..., allow_literals=False)`, so any VS
literal (`vec3(0,0,1)`, `0.5`) raises "vertex literal constants need a host
manifest". `vs_lighting`'s *math already works* post-refactor — this is the one
wire missing. Flip to `allow_literals=True`, return the `lit_manifest` from
`lower_vertex` (mirror `lower_fragment`), declare the literals as VS constants in
the `p.finish` constant list, and have `lower_ex`'s vertex branch return
`(blob, manifest)` instead of `(blob, [])`. Host uploads them via
`Xe_SetVertexShaderConstantF`.
- **Accept:** `vs_lighting.vert` (normalize/dot/max + a `vec3` literal) compiles
  + oracle-validates; manifest returned; modulate byte-identical.

### B — Cubemap / texture dimensions  *(HW gate)*
xsc has one fetch template (`tfetch3`, 2D); `samplerCube` silently miscompiles to
a 2D fetch. Detect the sampler `Dim` from `OpTypeImage` (2=2D? confirm: 1=2D,
3=Cube) in `resolve.py`, and in the `lower_fragment` tfetch `gather` emit a
**cube** fetch (3-component direction coord) for cube samplers. Investigate the
Xenos cube-fetch encoding first (don't guess — check `xdis/xenos_ucode.py`
TextureFetch fields + how xenia/XenosRecomp represent cube fetches). At minimum,
turn the current *silent miscompile* into either a correct cube fetch or a clean
`LowerError`.
- **Accept:** `fs_cubemap` emits a cube fetch (not 2D), oracle-validates; gallery
  slot with a cubemap texture (console-pending).

### C — Matrix math beyond `mat*vec`  *(no HW gate)*
Only `OpMatrixTimesVector` (145) is handled. Add `OpVectorTimesMatrix` (144,
`v*M` = dot of v with each column) and `OpMatrixTimesMatrix` (146, = 4×`mat*vec`,
result is a matrix held in 4 regs). Scope: at least `vec*mat`, and `mat*mat`
whose result immediately multiplies a vector. New handlers in `ValueLowerer`
(dispatch in `operand`, like the existing `emit_mat_vec`).
- **Accept:** testdata shaders for `vec*mat` and `mat*mat`; oracle-validated.

### D — Integer scalars used as float values  *(no HW gate; lower priority)*
The Xenos ALU is float-centric. Generalize the loop-only `ConvertSToF/FToS`
handling so plain `int` used in float context (`float(i)`, `int` counters
feeding float math) works outside loops too, with `int` arithmetic constant-
folded or float-represented where safe. Keep it tight — GLES2 fragment shaders
rarely need real integers; the goal is to stop rejecting common `int`→`float`
idioms, not a full integer ALU.
- **Accept:** a shader mixing an `int` expression into float output compiles +
  oracle-validates.

### E — Phase B: data-dependent loops in the frontend  *(HW gate; do LAST/own)*
The real-loop mechanism is **already proven on hardware** (Phase A — see
HANDOFF.md "Loop control flow"): `program.py` `begin_loop/end_loop`, trip count
from loop constant `integer_constants[8+id]`. Today `unroll.py` *bails* on a
uniform-bounded loop (`UnrollError: condition not (induction var vs constant)`).
Route that case to a real-loop lowering instead: allocate a persistent GPR for
each loop-carried variable, emit the pre-loop init, wrap the body in
`begin_loop/end_loop`, and bind the uniform bound to a loop constant (extend the
manifest with a loop-constant entry the host uploads). The value-graph/mem2reg
model does NOT fit a loop-carried register — this needs small register-machine
codegen for the accumulator. **Start minimal:** `for(i=0;i<u_count;i++) sum+=k;`
(no `aL` use) → `sum = u_count*k`.  Then `aL`-relative addressing is Phase C.
- This is the big one. One agent, no internal parallelism, integrate after A–D.
- **Accept:** a uniform-bounded loop compiles to a real LOOP_START/END shader;
  gallery slot where changing the uniform changes the output (console-pending).

### F — Optimizer: 0/1 source-select fold  *(behavioral; isolate)*
Fold `0.0`/`1.0` literal operands into the Xenos source-select swizzle (free,
saves constant-pool registers) instead of allocating a pooled constant. This
**changes microcode** for shaders using 0/1 literals, so it will move oracle
needles and possibly the literal manifest — update the affected tests
deliberately. Keep it off the byte-identical path (`modulate` has no 0/1 literal,
so it must stay byte-identical regardless). Lower priority; do it on its own
branch and integrate last with eyes open.
- **Accept:** a 0/1-using shader drops the pooled const; suite updated + green.

## Conventions (every task, non-negotiable)

- **Byte-identity canary:** `python tests/test_xsc_lower.py`, `modulate` stays
  byte-identical, after every change and every merge.
- **Hardware is the gate** for B and E (new GPU op/fetch). Oracle-green ≠ correct
  — see the ALLOC / interpolator / kill bugs in HANDOFF.md.
- **Test style:** match `test_xsc_lower.py` — `compile → oracle_hlsl → need[]`
  needles checking real microcode structure → `PASS/FAIL` + byte counts. One
  testdata shader per feature. Lines ≤ 79 cols.
- **Keep `ValueLowerer` unified** — add to the shared class, don't fork per-stage
  copies.
- **Commit identity:** `xost <xost@local>` (local git config already set), no
  co-author trailer. Author every commit by the project identity.

## Suggested wave plan

- **Wave 1 (parallel):** A, B, C, D, F — five branches at once.
- **Integrate** Wave 1 sequentially (canary after each), build one gallery elf32
  for the HW-gated slot(s), maintainer flashes once.
- **Wave 2:** E (loops) against the integrated trunk.
