# Next task — fragment control flow: `discard` + if/else-into-a-local

> Living task brief. Read [HANDOFF.md](HANDOFF.md) first (invariants +
> build/verify workflow); this builds on it. The `ValueLowerer` unification is
> the foundation — keep it intact, don't re-introduce per-stage copies.

## Why

Two of the highest-frequency real GLES2 fragment patterns still fail. Both
surface as the same error today (`if/else arm does not assign o_color`), because
the region lowering in `lower_fragment` (`_region` / `_block` / `_emit_region`,
~lines 914–1026 of `xsc/lower.py`) only handles arms that store the **output**
(`o_color`) directly:

```glsl
if (cond) { c = A; } else { c = B; }  o_color = c;   // assigns a LOCAL  -> fails
if (c.a < 0.5) discard;  o_color = c;                 // discard arm     -> fails
```

## Tasks, in order

### 1 — if/else assigning a local (frontend only, no new hardware)

Generalize the region model so the merge variable isn't hard-coded to `out_var`.
When both arms assign the same local, lower the if/else to a branch-free `cnd*`
select that **produces that local's value into a register** (the machinery
already exists — `_region_value` / `_emit_region` build chained selects), then
register that result so the existing `operand()` resolution feeds the post-merge
expression (`o_color = f(c)`) normally.

- **Accept:** a new `fs_branch_local` testdata shader compiles and oracle-
  validates to a `select(...)`; nested if/else still works; **`modulate` stays
  byte-identical**.

### 2 — `discard` (new hardware op — MUST hardware-verify)

`discard` → SPIR-V `OpKill` (opcode 252); `if(cond) discard;` is a
`SelectionMerge` whose arm is `OpKill`. The Xenos kill is a **vector ALU op**,
not control flow:

- `AluVectorOpcode.kKillGt=25 / kKillGe=26 / kKillEq=24 / kKillNe=27`. Semantics
  (per XenosRecomp): `kKillGt(a,b)` kills the pixel if **any** component of
  `a > b`. So `if(c.a < 0.5) discard` → `kKillGt(0.5, c.a)`.
- Recognize the `SelectionMerge → arm contains OpKill → merge` pattern, map the
  comparison to the matching `kKill*` op (mind operand order / `<` vs `>`), emit
  it, then continue to the normal output.
- **CRITICAL:** XenosRecomp validates *decode, not execution* — it will accept a
  wrong kill. This **must** be verified on silicon. Add a gallery slot (e.g.
  discard where `v_uv.x < 0.5` → half the cube's pixels vanish), build the
  elf32, hand it to the maintainer to flash. Place it **last** in the gallery
  (see the ALLOC / interpolator bugs in HANDOFF.md for why oracle-pass ≠
  correct).
- **Accept:** `fs_discard` compiles + oracle-validates; the gallery slot shows
  pixels actually discarded on the console.

### 3 — VS crash guard (trivial robustness)

`vs_wave` (`p.y = p.y + sin(...)`) currently **crashes** the lowerer with a
`struct.pack 'I' format requires 0 <= number` error instead of failing cleanly
(root: the VS path is `allow_literals=False`, so a literal hits an unencodable
value). You don't need to make it *work* — just guard it to raise a clear
`LowerError`. A crash in the lowerer erodes trust.

## Conventions (non-negotiable)

- **Byte-identity canary:** after every change, `python tests/test_xsc_lower.py`
  and confirm `modulate.frag` is still byte-identical. Task 1 touches shared
  lowering — this is how you catch a regression.
- **Test style:** match `test_xsc_lower.py` exactly — `compile → oracle_hlsl →
  need[]` needles that check *real microcode structure* → `PASS/FAIL` with byte
  counts. One testdata shader per feature.
- **Hardware is the gate for `discard`.** Software-green is necessary, not
  sufficient.
- **Commit identity:** `xost <xost@local>` (local git config already set), no
  co-author trailer.

## Out of scope (the task after this)

VS literal constants — wire the fragment's literal-manifest mechanism into the
VS path so `vec3(0,0,1)` / `0.5` work in vertex shaders (this is what blocks
`vs_lighting`, whose math already works post-refactor). Tackle after these three
land.
