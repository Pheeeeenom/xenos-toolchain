#!/usr/bin/env python3
"""N64-style color-combiner -> Xenos pixel-shader generator.

The emulator-enabling artifact: compiles parametric RDP combiner state

    out = (A - B) * C + D     (RGB and Alpha pipelines, independently)

directly to Xenos microcode containers via xasm.program.PixelProgram —
the same role GLideN64-class plugins fill by generating GLSL, with no
shader compiler involved at any point.

Register/constant contract (pairs with xost_vs2's interpolator layout):
    r0 = TEXCOORD0          c0 = PRIM   (app-set, Xe_SetPixelShaderConstantF)
    r1 = SHADE              c1 = ENV
    r2 = TEXEL0 (tfetch)    c2 = (1,1,1,1) for ONE
    r3 = scratch
Constant indices are fixed regardless of which are referenced, so a host
can set all three unconditionally.

The generator folds algebra so trivial modes cost one instruction:
A==B or C==0 -> D; B==0 drops the subtract; A==0 folds to a negated
operand; C==1 drops the multiply; D==0 fuses to mul; the general case is
a single mad. Encoder constraint honored: at most ONE distinct constant
register per ALU instruction (the ucode const_0/const_1 encoding is
shared across operands; we stay in the simple regime and split through
scratch when a step would need two).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import xdis.container_emit as ce  # noqa: E402
import xdis.xenos_ucode as xu  # noqa: E402
from xasm.program import PixelProgram, swizzle4, mask4  # noqa: E402

V = xu.AluVectorOpcode
_SCRATCH = 3


class Src(Enum):
    ZERO = "zero"
    ONE = "one"
    TEXEL0 = "texel0"
    SHADE = "shade"
    PRIM = "prim"
    ENV = "env"


_TEMPS = {Src.TEXEL0: 2, Src.SHADE: 1}
_CONSTS = {Src.PRIM: 0, Src.ENV: 1, Src.ONE: 2}


@dataclass(frozen=True)
class Op:
    """A resolved ALU operand."""
    reg: int
    is_temp: bool
    negate: bool = False

    @staticmethod
    def of(src: Src) -> "Op":
        if src in _TEMPS:
            return Op(_TEMPS[src], True)
        return Op(_CONSTS[src], False)

    def negated(self) -> "Op":
        return Op(self.reg, self.is_temp, not self.negate)

    @property
    def is_const(self) -> bool:
        return not self.is_temp


SCRATCH_OP = Op(_SCRATCH, True)


@dataclass
class CombinerCycle:
    """(A-B)*C+D selects; alpha selects default to the color selects."""
    a: Src
    b: Src
    c: Src
    d: Src
    aa: Src | None = None
    ab: Src | None = None
    ac: Src | None = None
    ad: Src | None = None

    def color_sel(self):
        return (self.a, self.b, self.c, self.d)

    def alpha_sel(self):
        return (self.aa if self.aa is not None else self.a,
                self.ab if self.ab is not None else self.b,
                self.ac if self.ac is not None else self.c,
                self.ad if self.ad is not None else self.d)


class Generator:
    """Stateless once constructed; reuse across many combiner words."""

    def __init__(self, alu_template: dict, tfetch_template: dict):
        """Templates are proven field dicts (corpus-derived; see
        examples/author_combiners.py)."""
        self._base = dict(alu_template, is_export=False)
        self._tfetch = tfetch_template

    # ---- low-level emission ----

    def _emit(self, p: PixelProgram, opcode, dest: int, export: bool,
              wmask: int, ops: dict, alpha: bool):
        n_consts = len({o.reg for o in ops.values() if o.is_const})
        assert n_consts <= 1, "split before emitting: >1 const register"
        f = dict(self._base, vector_opcode=opcode, vector_dest=dest,
                 is_export=export, vector_write_mask=wmask)
        swiz = swizzle4("wwww") if alpha else 0
        for n in (1, 2, 3):
            o = ops.get(n)
            if o is None:
                o = Op(0, True)  # unused slot: benign r0 reference
            f[f"src{n}_reg"] = o.reg
            f[f"src{n}_is_temp"] = o.is_temp
            f[f"src{n}_negate"] = o.negate
            f[f"src{n}_swizzle"] = swiz if ops.get(n) else 0
        p.alu(**f)

    def _mov_to_scratch(self, p, op: Op, alpha: bool):
        self._emit(p, V.kMax, _SCRATCH, False, 0xF, {1: op, 2: op}, alpha)
        return SCRATCH_OP

    # ---- one (A-B)*C+D pipeline ----

    def _pipeline(self, p: PixelProgram, sel, alpha: bool, wmask: int):
        a, b, c, d = sel

        # difference term
        if a == b:
            diff = None                      # (A-B) == 0
        elif b == Src.ZERO:
            diff = None if a == Src.ZERO else Op.of(a)
        elif a == Src.ZERO:
            diff = Op.of(b).negated()        # -(B)
        else:
            oa, ob = Op.of(a), Op.of(b).negated()
            if oa.is_const and ob.is_const and oa.reg != ob.reg:
                oa = self._mov_to_scratch(p, oa, alpha)
            self._emit(p, V.kAdd, _SCRATCH, False, 0xF,
                       {1: oa, 2: ob}, alpha)
            diff = SCRATCH_OP

        # product & sum
        if diff is None or c == Src.ZERO:
            if d == Src.ZERO:
                # true zero with no constant: SHADE - SHADE
                z = Op.of(Src.SHADE)
                self._emit(p, V.kAdd, 0, True, wmask,
                           {1: z, 2: z.negated()}, alpha)
            else:
                od = Op.of(d)
                self._emit(p, V.kMax, 0, True, wmask,
                           {1: od, 2: od}, alpha)
            return

        if c == Src.ONE:
            if d == Src.ZERO:
                self._emit(p, V.kMax, 0, True, wmask,
                           {1: diff, 2: diff}, alpha)
            else:
                oc, od = diff, Op.of(d)
                if oc.is_const and od.is_const and oc.reg != od.reg:
                    oc = self._mov_to_scratch(p, oc, alpha)
                self._emit(p, V.kAdd, 0, True, wmask,
                           {1: oc, 2: od}, alpha)
            return

        ops = {1: diff, 2: Op.of(c)}
        if d != Src.ZERO:
            ops[3] = Op.of(d)
        n_consts = len({o.reg for o in ops.values() if o.is_const})
        if n_consts > 1:
            # split: scratch = diff * C, then oC0 = scratch + D
            o1, o2 = ops[1], ops[2]
            if o1.is_const and o2.is_const:
                o1 = self._mov_to_scratch(p, o1, alpha)
            self._emit(p, V.kMul, _SCRATCH, False, 0xF,
                       {1: o1, 2: o2}, alpha)
            if 3 in ops:
                self._emit(p, V.kAdd, 0, True, wmask,
                           {1: SCRATCH_OP, 2: ops[3]}, alpha)
            else:
                self._emit(p, V.kMax, 0, True, wmask,
                           {1: SCRATCH_OP, 2: SCRATCH_OP}, alpha)
        else:
            opcode = V.kMad if 3 in ops else V.kMul
            self._emit(p, opcode, 0, True, wmask, ops, alpha)

    # ---- whole shader ----

    def generate(self, cycle: CombinerCycle) -> bytes:
        """One-cycle combiner -> pixel-shader container bytes."""
        color, alpha = cycle.color_sel(), cycle.alpha_sel()
        used = set(color) | set(alpha)

        p = PixelProgram()
        p.input(0, ce.USAGE_TEXCOORD)
        p.input(1, ce.USAGE_COLOR)
        if Src.TEXEL0 in used:
            p.tfetch(self._tfetch, dest=2, src=0)
        p.alloc_colors()

        if color == alpha:
            self._pipeline(p, color, alpha=False, wmask=0xF)
        else:
            self._pipeline(p, color, alpha=False, wmask=mask4("xyz"))
            self._pipeline(p, alpha, alpha=True, wmask=mask4("w"))

        constants = []
        if Src.TEXEL0 in used:
            constants.append(ce.Constant("tex0", ce.RS_SAMPLER, 0, 1,
                                         ce.CLASS_OBJECT, 0xC,
                                         rows=1, columns=1))
        for s, name in ((Src.PRIM, "prim"), (Src.ENV, "env"),
                        (Src.ONE, "one")):
            if s in used:
                constants.append(ce.Constant(
                    name, ce.RS_FLOAT4, _CONSTS[s], 1,
                    ce.CLASS_VECTOR, ce.TYPE_FLOAT, rows=1, columns=4))
        return p.finish(constants=constants)
