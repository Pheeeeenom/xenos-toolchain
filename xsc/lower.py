#!/usr/bin/env python3
"""xsc lowering — SPIR-V function body -> Xenos shader container.

v1 scope: branch-free vertex and (next) fragment shaders. The body is
lowered by evaluating each output's value expression through the SSA def
chain and emitting Program-builder calls, scheduled into the Xenos phase
structure (vfetch -> alloc position -> transform -> alloc interpolators ->
interpolator exports for VS).

Instruction templates (proven ALU/vfetch field skeletons) come from the
golden corpus, same as the author_* scripts.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import xdis.container as container  # noqa: E402
import xdis.container_emit as ce  # noqa: E402
import xdis.xenos_ucode as xu  # noqa: E402
import xsc.spirv as spirv  # noqa: E402
from xsc.unroll import unroll  # noqa: E402
from xsc.resolve import resolve_interface, resolve_types  # noqa: E402
from xasm.program import VertexProgram, PixelProgram  # noqa: E402
from author_passthrough import alu_fields, vf_fields, corpus_records  # noqa: E402
from author_textured import tf_fields  # noqa: E402

OPOS = int(xu.ExportRegister.kVSPosition)  # 62
_V = xu.AluVectorOpcode
_S = xu.AluScalarOpcode


class LowerError(Exception):
    pass


def _templates():
    vs0 = corpus_records("vs0")
    ps0 = corpus_records("ps_fb0")
    return {
        "vf": vf_fields(xu.VertexFetchInstruction(*vs0["vfetch3"].raw)),
        "dp4": alu_fields(xu.AluInstruction(*vs0["alu7"].raw)),   # dp4 oPos.x
        "mov": alu_fields(xu.AluInstruction(*vs0["alu13"].raw)),  # max o2,r,r
        "tf": tf_fields(xu.TextureFetchInstruction(*ps0["tfetch3"].raw)),
    }


def _store_map(m: spirv.Module):
    """target-id -> value-id for every OpStore in the body."""
    out = {}
    for ins in m.instrs:
        if ins.opcode == 62:  # Store target value
            out[ins.operands[0]] = ins.operands[1]
    return out


def _access_chain_base(m: spirv.Module, ptr_id: int):
    """If ptr_id is an AccessChain, return (base_var_id, [index consts])."""
    ins = m.by_result.get(ptr_id)
    if ins is None or ins.opcode != 65:  # AccessChain
        return None
    base = ins.operands[0]
    idx_consts = []
    for w in ins.operands[1:]:
        c = m.by_result.get(w)
        if c is not None and c.opcode == 43:  # Constant
            idx_consts.append(c.operands[0])
    return base, idx_consts


class ValueLowerer:
    """Shared SPIR-V value lowering for vertex and fragment stages.

    Stage setup still lives in lower_vertex/lower_fragment (vfetches, tfetch
    gather, exports, constants). This context owns the single source of truth
    for ALU source resolution and value-expression emission.
    """

    BCAST_X = sum(((0 - p) & 3) << (2 * p) for p in range(4))

    def __init__(self, *, m, si, program, templates, types, stores, first_reg,
                 inputs, local_store, stage, sample_reg=None,
                 literal_base=0, allow_literals=True):
        self.m = m
        self.si = si
        self.p = program
        self.T = templates
        self.types = types
        self.stores = stores
        self.next_reg = first_reg
        self.inputs = inputs
        self.local_store = local_store
        self.stage = stage
        self.sample_reg = sample_reg if sample_reg is not None else {}
        self.allow_literals = allow_literals
        self.lit_base = literal_base
        self.lit_pool = {}
        self.lit_manifest = []
        self.op_cache = {}
        self.local_values = {}

    def fresh(self):
        r = self.next_reg
        self.next_reg += 1
        return r

    def vec_size(self, ssa):
        t = self.types.get(self.m.by_result[ssa].result_type)
        return t.count if t and t.kind == "vector" else 1

    def scalar_const(self, ssa):
        bits = self.m.by_result[ssa].operands[0] & 0xFFFFFFFF
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    def literal_reg(self, value4):
        if not self.allow_literals:
            raise LowerError("vertex literal constants need a host manifest")
        key = tuple(round(x, 9) for x in value4)
        if key in self.lit_pool:
            return self.lit_pool[key]
        reg = self.lit_base
        self.lit_base += 1
        self.lit_pool[key] = reg
        self.lit_manifest.append((reg, tuple(value4)))
        return reg

    def const_base(self, ssa_id: int) -> tuple:
        """Resolve a Load of a uniform -> (constant base reg, reg count)."""
        ins = self.m.by_result.get(ssa_id)
        if ins is None or ins.opcode != 61:
            raise LowerError("matrix operand must be a uniform Load")
        role = self.si.roles.get(ins.operands[0])
        if not role or role[0] != "constant":
            raise LowerError("expected a uniform constant")
        return role[1]

    def src(self, fields, n, opnd, negate=False, swizzle=0):
        kind, val = opnd
        fields[f"src{n}_reg"] = val
        fields[f"src{n}_is_temp"] = (kind == "reg")
        fields[f"src{n}_swizzle"] = swizzle
        fields[f"src{n}_negate"] = negate

    def emit(self, opcode, ops, dst, export, mask=0xF, clamp=False):
        n_const = len({v[0][1] for v in ops.values() if v[0][0] == "const"})
        if n_const > 1:
            raise LowerError(">1 constant operand (v1 cannot split here)")
        f = dict(self.T["mov"], vector_opcode=opcode, vector_dest=dst,
                 is_export=export, vector_write_mask=mask,
                 vector_clamp=clamp)
        for n, v in ops.items():
            opnd, neg = v[0], v[1]
            swz = v[2] if len(v) > 2 else 0
            self.src(f, n, opnd, neg, swz)
        self.p.alu(**f)

    def emit_mat_vec(self, mat_ssa, vec_ssa, dst=None, export_position=False):
        if self.stage != "vertex":
            raise LowerError("matrix-vector lowering is vertex-only")
        mat_base, mat_regs = self.const_base(mat_ssa)
        if mat_regs != 4:
            raise LowerError("vertex matrix multiply currently expects mat4")
        vec_op = self.operand(vec_ssa)
        if vec_op[0] != "reg":
            raise LowerError("matrix vector operand must be in a register")
        target = OPOS if export_position else (
            dst if dst is not None else self.fresh())
        for i in range(4):
            self.p.alu(**dict(self.T["dp4"],
                              vector_dest=target,
                              is_export=export_position,
                              vector_write_mask=(1 << i),
                              src1_reg=vec_op[1], src1_is_temp=True,
                              src2_reg=mat_base + i, src2_is_temp=False))
        return target

    def operand(self, ssa_id: int):
        """Resolve an SSA value to ('reg', n) or ('const', reg), emitting
        ops for computed values (memoized so a shared value emits once)."""
        if ssa_id in self.op_cache:
            return self.op_cache[ssa_id]
        ins = self.m.by_result[ssa_id]
        op = ins.opcode
        if op == 61:                              # Load
            src = ins.operands[0]
            ptr = self.m.by_result.get(src)
            if ptr is not None and ptr.opcode == 65:
                o, comp = self.scalar_origin(ssa_id)
                bsw = 0 if comp == "B" else self.bsw(comp)
                r = self.fresh()
                self.emit(_V.kMax, {1: (o, False, bsw),
                                    2: (o, False, bsw)}, r, False)
                res = ("reg", r)
            elif src in self.inputs:
                res = ("reg", self.inputs[src])
            elif src in self.local_values:
                res = self.local_values[src]
            elif src in self.local_store:
                res = self.operand(self.local_store[src])
            else:
                role = self.si.roles.get(src)
                if role and role[0] == "constant":
                    res = ("const", role[1][0])
                else:
                    raise LowerError(f"{self.stage}: unresolved Load %{src}")
        elif op == 87:
            if ssa_id not in self.sample_reg:
                raise LowerError(
                    "ImageSample was not gathered before lowering")
            res = ("reg", self.sample_reg[ssa_id])
        elif op == 79:                            # VectorShuffle
            idxs = list(ins.operands[2:])
            if idxs == list(range(len(idxs))):
                res = self.operand(ins.operands[0])
            elif all(i < 4 for i in idxs):
                base = self.operand(ins.operands[0])
                swz = self.swz(idxs)
                r = self.fresh()
                self.emit(_V.kMax, {1: (base, False, swz),
                                    2: (base, False, swz)}, r, False)
                res = ("reg", r)
            else:
                raise LowerError(f"two-source swizzle {idxs} (v1)")
        elif op == 145:                           # MatrixTimesVector
            res = ("reg", self.emit_mat_vec(ins.operands[0], ins.operands[1]))
        elif op in (129, 131, 133):               # FAdd FSub FMul
            res = ("reg", self.binop(op, ins.operands[0], ins.operands[1]))
        elif op == 142:                           # VectorTimesScalar
            res = ("reg", self.vts(ins.operands[0], ins.operands[1]))
        elif op == 148:                           # OpDot
            res = ("reg", self.dot(ins.operands[0], ins.operands[1]))
        elif op == 136:                           # OpFDiv
            res = ("reg", self.fdiv(ins.operands[0], ins.operands[1]))
        elif op == 141:                           # OpFMod
            res = ("reg", self.fmod(ins.operands[0], ins.operands[1]))
        elif op == 43:                            # OpConstant
            v = self.scalar_const(ssa_id)
            res = ("const", self.literal_reg((v, v, v, v)))
        elif op == 44:                            # OpConstantComposite
            comps = [self.scalar_const(c) for c in ins.operands]
            comps = (comps + [0.0, 0.0, 0.0, 0.0])[:4]
            res = ("const", self.literal_reg(tuple(comps)))
        elif op == 80:                            # CompositeConstruct
            parts = ins.operands
            if len(set(parts)) == 1:
                if self.m.by_result[parts[0]].opcode == 43:
                    v = self.scalar_const(parts[0])
                    res = ("const", self.literal_reg((v, v, v, v)))
                else:
                    res = self.operand(parts[0])
            else:
                r = self.fresh()
                self.emit_construct(ins, r, export=False)
                res = ("reg", r)
        elif op == 12:                            # ExtInst
            res = ("reg", self.extinst(ins))
        else:
            raise LowerError(f"v1 {self.stage}: unhandled op {ins.name}")
        self.op_cache[ssa_id] = res
        return res

    def const_scalar(self, ssa):
        """Scalar constant, possibly wrapped in a broadcast construct."""
        ins = self.m.by_result[ssa]
        if ins.opcode == 43:
            return self.scalar_const(ssa)
        if ins.opcode == 80 and len(set(ins.operands)) == 1 and \
                self.m.by_result[ins.operands[0]].opcode == 43:
            return self.scalar_const(ins.operands[0])
        raise LowerError(f"%{ssa} is not a constant scalar")

    def const_val(self, ssa_id):
        c = self.m.by_result.get(ssa_id)
        if c is None or c.opcode != 43:
            raise LowerError(f"%{ssa_id} is not an int constant")
        return c.operands[0]

    def scalar_origin(self, ssa_id):
        """Resolve a scalar SSA to (vector_operand, component). Component is an
        int, or "B" when the value is broadcast across all components."""
        ins = self.m.by_result[ssa_id]
        if ins.opcode == 81:                      # CompositeExtract(v, idx)
            return self.operand(ins.operands[0]), ins.operands[1]
        if ins.opcode == 61:                      # Load
            src = ins.operands[0]
            ptr = self.m.by_result.get(src)
            if ptr is not None and ptr.opcode == 65:
                base, comp = ptr.operands[0], self.const_val(ptr.operands[1])
                if base in self.local_values:
                    return self.local_values[base], comp
                if base in self.local_store:
                    return self.operand(self.local_store[base]), comp
                if base in self.inputs:
                    return ("reg", self.inputs[base]), comp
                role = self.si.roles.get(base)
                if role and role[0] == "constant":
                    return ("const", role[1][0]), comp
            if src in self.local_values:
                return self.local_values[src], "B"
            if src in self.local_store:
                return self.operand(self.local_store[src]), "B"
            role = self.si.roles.get(src)
            if role and role[0] == "constant":
                return ("const", role[1][0]), 0
            if src in self.inputs:
                return ("reg", self.inputs[src]), 0
        if ins.opcode in (148, 43):
            return self.operand(ssa_id), "B"
        if ins.opcode in (129, 131, 133, 136, 141, 12):
            return self.operand(ssa_id), "B"
        raise LowerError(f"cannot resolve scalar %{ssa_id}")

    def emit_construct(self, ins, dst_reg, export=True):
        origins = [self.scalar_origin(part) for part in ins.operands]
        n = len(origins)
        j = 0
        while j < n:
            op_j = origins[j][0]
            k = j
            while k < n and origins[k][0] == op_j and \
                    origins[k][1] in (k, "B"):
                k += 1
            if k > j:
                mask = sum(1 << t for t in range(j, k))
                self.emit(_V.kMax, {1: (op_j, False), 2: (op_j, False)},
                          dst_reg, export, mask=mask)
                j = k
            else:
                op, comp = origins[j]
                swz = 0 if comp == "B" else ((comp - j) & 3) << (2 * j)
                self.emit(_V.kMax, {1: (op, False, swz),
                                    2: (op, False, swz)},
                          dst_reg, export, mask=(1 << j))
                j += 1

    def binop(self, op, a, b, dst=None, export=False, clamp=False):
        oa, ob = self.operand(a), self.operand(b)
        if dst is None:
            dst = self.fresh()
        if op == 133:
            self.emit(_V.kMul, {1: (oa, False), 2: (ob, False)}, dst,
                      export, clamp=clamp)
        elif op == 129:
            self.emit(_V.kAdd, {1: (oa, False), 2: (ob, False)}, dst,
                      export, clamp=clamp)
        else:
            self.emit(_V.kAdd, {1: (oa, False), 2: (ob, True)}, dst,
                      export, clamp=clamp)
        return dst

    def emit_clamped(self, ssa, dst, export):
        vi = self.m.by_result[ssa]
        if vi.opcode in (129, 131, 133):
            return self.binop(vi.opcode, vi.operands[0], vi.operands[1],
                              dst=dst, export=export, clamp=True)
        r = self.operand(ssa)
        self.emit(_V.kMax, {1: (r, False), 2: (r, False)}, dst, export,
                  clamp=True)
        return dst

    def vts(self, vec, scal, dst=None, export=False):
        ov, os = self.operand(vec), self.operand(scal)
        if dst is None:
            dst = self.fresh()
        self.emit(_V.kMul, {1: (ov, False),
                            2: (os, False, self.BCAST_X)}, dst, export)
        return dst

    def dot(self, a, b, dst=None, export=False):
        oa, ob = self.operand(a), self.operand(b)
        opcode = _V.kDp3 if self.vec_size(a) == 3 else _V.kDp4
        if dst is None:
            dst = self.fresh()
        self.emit(opcode, {1: (oa, False), 2: (ob, False)}, dst, export,
                  mask=0xF)
        return dst

    def emit_scalar(self, sop, dst, src_op, src_comp=3, mask=0xF):
        kind, reg = src_op
        swz = ((src_comp - 3) & 3) << 6
        f = dict(self.T["mov"], vector_opcode=_V.kMax, vector_dest=0,
                 vector_write_mask=0, vector_clamp=False,
                 scalar_opcode=sop, scalar_dest=dst, scalar_write_mask=mask,
                 scalar_clamp=False, is_export=False,
                 src3_reg=reg, src3_is_temp=(kind == "reg"),
                 src3_swizzle=swz, src3_negate=False)
        self.p.alu(**f)
        return dst

    def dotself(self, ov, n):
        d = self.fresh()
        self.emit(_V.kDp3 if n == 3 else _V.kDp4,
                  {1: (ov, False), 2: (ov, False)}, d, False, mask=0xF)
        return d

    def fdiv(self, a, b, dst=None, export=False):
        oa, ob = self.operand(a), self.operand(b)
        bi = self.m.by_result.get(b)
        scalar_b = (bi is not None and bi.opcode == 80 and
                    len(set(bi.operands)) == 1 and
                    self.m.by_result[bi.operands[0]].opcode != 43)
        if dst is None:
            dst = self.fresh()
        n = max(self.vec_size(a), self.vec_size(b), 1)
        if scalar_b or n == 1:
            rb = self.emit_scalar(_S.kRcp, self.fresh(), ob, src_comp=0)
            self.emit(_V.kMul, {1: (oa, False), 2: (("reg", rb), False)},
                      dst, export)
        else:
            for i in range(n):
                rb = self.emit_scalar(_S.kRcp, self.fresh(), ob, src_comp=i)
                self.emit(_V.kMul, {1: (oa, False),
                                    2: (("reg", rb), False)},
                          dst, export, mask=(1 << i))
        return dst

    def to_reg(self, opnd):
        if opnd[0] == "reg":
            return opnd
        r = self.fresh()
        self.emit(_V.kMax, {1: (opnd, False), 2: (opnd, False)}, r, False)
        return ("reg", r)

    @staticmethod
    def swz(idxs):
        return sum(((idxs[p] - p) & 3) << (2 * p) for p in range(len(idxs)))

    @staticmethod
    def bsw(c):
        return sum(((c - p) & 3) << (2 * p) for p in range(4))

    def recip(self, opnd, n):
        if n == 1:
            return self.emit_scalar(_S.kRcp, self.fresh(), opnd, src_comp=0)
        r = self.fresh()
        for i in range(n):
            ri = self.emit_scalar(_S.kRcp, self.fresh(), opnd, src_comp=i)
            self.emit(_V.kMax, {1: (("reg", ri), False),
                                2: (("reg", ri), False)},
                      r, False, mask=(1 << i))
        return r

    def fmod(self, a, b, dst=None, export=False):
        oa, ob = self.operand(a), self.operand(b)
        q = self.fdiv(a, b)
        fq = self.fresh()
        self.emit(_V.kFloor, {1: (("reg", q), False),
                              2: (("reg", q), False)}, fq, False)
        if dst is None:
            dst = self.fresh()
        if oa[0] == "const" and ob[0] == "const":
            oa = self.to_reg(oa)
        self.emit(_V.kMad, {1: (ob, True), 2: (("reg", fq), False),
                            3: (oa, False)}, dst, export)
        return dst

    def extinst(self, ins, dst=None, export=False):
        inst = ins.operands[1]
        args = ins.operands[2:]
        if dst is None:
            dst = self.fresh()
        if inst == 46:                            # FMix(x,y,a)=x+(y-x)*a
            ox, oy, oa = (self.operand(args[0]), self.operand(args[1]),
                          self.operand(args[2]))
            tmp = self.fresh()
            self.emit(_V.kAdd, {1: (oy, False), 2: (ox, True)}, tmp, False)
            self.emit(_V.kMad, {1: (("reg", tmp), False), 2: (oa, False),
                                3: (ox, False)}, dst, export)
            return dst
        if inst in (40, 37):                      # FMax / FMin
            self.emit(_V.kMax if inst == 40 else _V.kMin,
                      {1: (self.operand(args[0]), False),
                       2: (self.operand(args[1]), False)}, dst, export)
            return dst

        def scalar_result(sop, src_op, src_comp=3):
            if export:
                r = self.emit_scalar(sop, self.fresh(), src_op, src_comp)
                self.emit(_V.kMax, {1: (("reg", r), False),
                                    2: (("reg", r), False)}, dst, True)
            else:
                self.emit_scalar(sop, dst, src_op, src_comp)
            return dst

        if inst == 8:                             # Floor
            ox = self.operand(args[0])
            self.emit(_V.kFloor, {1: (ox, False), 2: (ox, False)},
                      dst, export)
            return dst
        if inst == 69:                            # Normalize
            ov = self.operand(args[0])
            d = self.dotself(ov, self.vec_size(args[0]))
            s = self.emit_scalar(_S.kRsq, self.fresh(), ("reg", d))
            self.emit(_V.kMul, {1: (ov, False), 2: (("reg", s), False)},
                      dst, export)
            return dst
        if inst == 66:                            # Length
            ov = self.operand(args[0])
            d = self.dotself(ov, self.vec_size(args[0]))
            return scalar_result(_S.kSqrt, ("reg", d))
        if inst == 67:                            # Distance
            oa, ob = self.operand(args[0]), self.operand(args[1])
            diff = self.fresh()
            self.emit(_V.kAdd, {1: (oa, False), 2: (ob, True)}, diff, False)
            d = self.dotself(("reg", diff), self.vec_size(args[0]))
            return scalar_result(_S.kSqrt, ("reg", d))
        if inst == 31:
            return scalar_result(_S.kSqrt, self.operand(args[0]))
        if inst == 32:
            return scalar_result(_S.kRsq, self.operand(args[0]))
        if inst == 29:
            return scalar_result(_S.kExp, self.operand(args[0]))
        if inst == 30:
            return scalar_result(_S.kLog, self.operand(args[0]))

        def sbcast(c):
            return sum(((c - pp) & 3) << (2 * pp) for pp in range(4))

        def scalar_src(ssa):
            o, c = self.scalar_origin(ssa)
            return o, (0 if c == "B" else sbcast(c)), (3 if c == "B" else c)

        def bcast(ssa):
            o, sw, _c = scalar_src(ssa)
            r = self.fresh()
            self.emit(_V.kMax, {1: (o, False, sw), 2: (o, False, sw)},
                      r, False)
            return ("reg", r)

        def horner(xop, coeffs):
            p = self.to_reg(("const", self.literal_reg((coeffs[0],) * 4)))
            pr = p[1]
            for ck in coeffs[1:]:
                ckop = ("const", self.literal_reg((ck,) * 4))
                self.emit(_V.kMad, {1: (xop, False),
                                    2: (("reg", pr), False),
                                    3: (ckop, False)}, pr, False)
            return ("reg", pr)

        def asin_reg(ssa):
            xr = bcast(ssa)
            ax = self.fresh()
            self.emit(_V.kMax, {1: (xr, False), 2: (xr, True)}, ax, False)
            p = horner(("reg", ax),
                       [-0.0187293, 0.0742610, -0.2121144, 1.5707288])
            one = ("const", self.literal_reg((1.0,) * 4))
            om = self.fresh()
            self.emit(_V.kAdd, {1: (one, False),
                                2: (("reg", ax), True)}, om, False)
            s = self.emit_scalar(_S.kSqrt, self.fresh(), ("reg", om))
            halfpi = ("const", self.literal_reg((1.5707963,) * 4))
            inner = self.fresh()
            self.emit(_V.kMad, {1: (("reg", s), True), 2: (p, False),
                                3: (halfpi, False)}, inner, False)
            zero = ("const", self.literal_reg((0.0,) * 4))
            gt = self.fresh()
            self.emit(_V.kSgt, {1: (xr, False), 2: (zero, False)}, gt, False)
            lt = self.fresh()
            self.emit(_V.kSgt, {1: (zero, False), 2: (xr, False)}, lt, False)
            sgn = self.fresh()
            self.emit(_V.kAdd, {1: (("reg", gt), False),
                                2: (("reg", lt), True)}, sgn, False)
            out = self.fresh()
            self.emit(_V.kMul, {1: (("reg", sgn), False),
                                2: (("reg", inner), False)}, out, False)
            return out

        def atan_reg(ssa):
            xr = bcast(ssa)
            x2 = self.fresh()
            self.emit(_V.kMul, {1: (xr, False), 2: (xr, False)}, x2, False)
            p = horner(("reg", x2), [0.0208351, -0.0851330, 0.1801410,
                                     -0.3302995, 0.9998660])
            out = self.fresh()
            self.emit(_V.kMul, {1: (xr, False), 2: (p, False)}, out, False)
            return out

        if inst == 27:                            # Exp
            o, sw, _c = scalar_src(args[0])
            ck = ("const", self.literal_reg((1.4426950408,) * 4))
            mm = self.fresh()
            self.emit(_V.kMul, {1: (o, False, sw), 2: (ck, False)},
                      mm, False)
            return scalar_result(_S.kExp, ("reg", mm))
        if inst == 28:                            # Log
            o, _sw, c = scalar_src(args[0])
            lg = self.emit_scalar(_S.kLog, self.fresh(), o, src_comp=c)
            ck = ("const", self.literal_reg((0.6931471805,) * 4))
            self.emit(_V.kMul, {1: (("reg", lg), False), 2: (ck, False)},
                      dst, export)
            return dst
        if inst in (13, 14):                      # Sin / Cos
            o, _sw, c = scalar_src(args[0])
            return scalar_result(_S.kSin if inst == 13 else _S.kCos, o, c)
        if inst == 71:                            # Reflect
            oi, on = self.operand(args[0]), self.operand(args[1])
            d = self.dotself(on, self.vec_size(args[0])) \
                if args[0] == args[1] else None
            if d is None:
                d = self.fresh()
                self.emit(_V.kDp3 if self.vec_size(args[0]) == 3 else _V.kDp4,
                          {1: (on, False), 2: (oi, False)}, d, False,
                          mask=0xF)
            d2 = self.fresh()
            self.emit(_V.kAdd, {1: (("reg", d), False),
                                2: (("reg", d), False)}, d2, False)
            self.emit(_V.kMad, {1: (("reg", d2), True), 2: (on, False),
                                3: (oi, False)}, dst, export)
            return dst
        if inst == 26:                            # Pow
            ox, oy = self.operand(args[0]), self.operand(args[1])
            n = max(self.vec_size(args[0]), 1)
            for i in range(n):
                li = self.emit_scalar(_S.kLog, self.fresh(), ox, src_comp=i)
                mi = self.fresh()
                self.emit(_V.kMul, {1: (("reg", li), False),
                                    2: (oy, False, self.BCAST_X)}, mi, False)
                ri = self.emit_scalar(_S.kExp, self.fresh(), ("reg", mi))
                self.emit(_V.kMax, {1: (("reg", ri), False),
                                    2: (("reg", ri), False)}, dst, export,
                          mask=(1 << i))
            return dst
        if inst == 4:                             # FAbs
            ox = self.operand(args[0])
            self.emit(_V.kMax, {1: (ox, False), 2: (ox, True)}, dst, export)
            return dst
        if inst == 6:                             # FSign
            ox = self.operand(args[0])
            if ox[0] == "const":
                xr = self.fresh()
                self.emit(_V.kMax, {1: (ox, False), 2: (ox, False)},
                          xr, False)
                ox = ("reg", xr)
            zero = ("const", self.literal_reg((0.0, 0.0, 0.0, 0.0)))
            gt = self.fresh()
            self.emit(_V.kSgt, {1: (ox, False), 2: (zero, False)}, gt, False)
            lt = self.fresh()
            self.emit(_V.kSgt, {1: (zero, False), 2: (ox, False)}, lt, False)
            self.emit(_V.kAdd, {1: (("reg", gt), False),
                                2: (("reg", lt), True)}, dst, export)
            return dst
        if inst == 15:                            # Tan
            o, _sw, c = scalar_src(args[0])
            s = self.emit_scalar(_S.kSin, self.fresh(), o, src_comp=c)
            co = self.emit_scalar(_S.kCos, self.fresh(), o, src_comp=c)
            rc = self.emit_scalar(_S.kRcp, self.fresh(), ("reg", co))
            self.emit(_V.kMul, {1: (("reg", s), False),
                                2: (("reg", rc), False)}, dst, export)
            return dst
        if inst == 10:                            # Fract
            ox = self.operand(args[0])
            self.emit(_V.kFrc, {1: (ox, False), 2: (ox, False)}, dst, export)
            return dst
        if inst == 3:                             # Trunc
            ox = self.operand(args[0])
            self.emit(_V.kTrunc, {1: (ox, False), 2: (ox, False)}, dst, export)
            return dst
        if inst == 9:                             # Ceil
            ox = self.operand(args[0])
            tf = self.fresh()
            self.emit(_V.kFloor, {1: (ox, True), 2: (ox, True)}, tf, False)
            self.emit(_V.kMax, {1: (("reg", tf), True),
                                2: (("reg", tf), True)}, dst, export)
            return dst
        if inst in (1, 2):                        # Round/RoundEven
            ox = self.to_reg(self.operand(args[0]))
            half = ("const", self.literal_reg((0.5, 0.5, 0.5, 0.5)))
            s = self.fresh()
            self.emit(_V.kAdd, {1: (ox, False), 2: (half, False)}, s, False)
            self.emit(_V.kFloor, {1: (("reg", s), False),
                                  2: (("reg", s), False)}, dst, export)
            return dst
        if inst in (11, 12):                      # Radians/Degrees
            ox = self.to_reg(self.operand(args[0]))
            k = 0.0174532925 if inst == 11 else 57.2957795
            ck = ("const", self.literal_reg((k, k, k, k)))
            self.emit(_V.kMul, {1: (ox, False), 2: (ck, False)}, dst, export)
            return dst
        if inst == 48:                            # Step
            oe = self.operand(args[0])
            ox = self.to_reg(self.operand(args[1]))
            self.emit(_V.kSge, {1: (ox, False), 2: (oe, False)}, dst, export)
            return dst
        if inst == 49:                            # SmoothStep
            oe0, oe1, ox = (self.operand(args[0]), self.operand(args[1]),
                            self.operand(args[2]))
            e0r = self.to_reg(oe0)
            n = max(self.vec_size(args[2]), 1)
            num = self.fresh()
            self.emit(_V.kAdd, {1: (ox, False), 2: (e0r, True)}, num, False)
            den = self.fresh()
            self.emit(_V.kAdd, {1: (oe1, False), 2: (e0r, True)}, den, False)
            rden = self.recip(("reg", den), n)
            tt = self.fresh()
            self.emit(_V.kMul, {1: (("reg", num), False),
                                2: (("reg", rden), False)}, tt, False,
                      clamp=True)
            two_t = self.fresh()
            self.emit(_V.kAdd, {1: (("reg", tt), False),
                                2: (("reg", tt), False)}, two_t, False)
            three = ("const", self.literal_reg((3.0, 3.0, 3.0, 3.0)))
            h = self.fresh()
            self.emit(_V.kAdd, {1: (three, False),
                                2: (("reg", two_t), True)}, h, False)
            t2 = self.fresh()
            self.emit(_V.kMul, {1: (("reg", tt), False),
                                2: (("reg", tt), False)}, t2, False)
            self.emit(_V.kMul, {1: (("reg", t2), False),
                                2: (("reg", h), False)}, dst, export)
            return dst
        if inst == 68:                            # Cross
            oa, ob = self.operand(args[0]), self.operand(args[1])
            yzx, zxy = self.swz([1, 2, 0]), self.swz([2, 0, 1])
            tc = self.fresh()
            self.emit(_V.kMul, {1: (oa, False, zxy),
                                2: (ob, False, yzx)}, tc, False)
            self.emit(_V.kMad, {1: (oa, False, yzx),
                                2: (ob, False, zxy),
                                3: (("reg", tc), True)}, dst, export)
            return dst
        if inst == 43:                            # FClamp
            lo, hi = self.const_scalar(args[1]), self.const_scalar(args[2])
            if lo == 0.0 and hi == 1.0:
                return self.emit_clamped(args[0], dst, export)
            ox = self.operand(args[0])
            lo_op = ("const", self.literal_reg((lo, lo, lo, lo)))
            hi_op = ("const", self.literal_reg((hi, hi, hi, hi)))
            t = self.fresh()
            self.emit(_V.kMax, {1: (ox, False), 2: (lo_op, False)}, t, False)
            self.emit(_V.kMin, {1: (("reg", t), False),
                                2: (hi_op, False)}, dst, export)
            return dst
        if inst == 16:                            # Asin
            r = asin_reg(args[0])
            self.emit(_V.kMax, {1: (("reg", r), False),
                                2: (("reg", r), False)}, dst, export)
            return dst
        if inst == 17:                            # Acos
            r = asin_reg(args[0])
            halfpi = ("const", self.literal_reg((1.5707963, 1.5707963,
                                                 1.5707963, 1.5707963)))
            self.emit(_V.kAdd, {1: (halfpi, False),
                                2: (("reg", r), True)}, dst, export)
            return dst
        if inst == 18:                            # Atan
            r = atan_reg(args[0])
            self.emit(_V.kMax, {1: (("reg", r), False),
                                2: (("reg", r), False)}, dst, export)
            return dst
        raise LowerError(f"v1 {self.stage}: unhandled GLSL.std.450 #{inst}")

    def emit_value(self, ssa, dst, export, export_position=False):
        """Emit a value expression to `dst` (export-fused when export)."""
        vi = self.m.by_result[ssa]
        if vi.opcode == 80:
            self.emit_construct(vi, dst, export)
        elif vi.opcode == 145:
            self.emit_mat_vec(vi.operands[0], vi.operands[1], dst=dst,
                              export_position=export_position)
        elif vi.opcode in (129, 131, 133):
            self.binop(vi.opcode, vi.operands[0], vi.operands[1], dst=dst,
                       export=export)
        elif vi.opcode == 142:
            self.vts(vi.operands[0], vi.operands[1], dst=dst, export=export)
        elif vi.opcode == 148:
            self.dot(vi.operands[0], vi.operands[1], dst=dst, export=export)
        elif vi.opcode == 136:
            self.fdiv(vi.operands[0], vi.operands[1], dst=dst, export=export)
        elif vi.opcode == 141:
            self.fmod(vi.operands[0], vi.operands[1], dst=dst, export=export)
        elif vi.opcode == 12:
            self.extinst(vi, dst=dst, export=export)
        else:
            r = self.operand(ssa)
            if r[0] == "const":
                raise LowerError(f"{self.stage} export of constants (v1)")
            self.emit(_V.kMax, {1: (r, False), 2: (r, False)}, dst, export)


def lower_vertex(m: spirv.Module) -> bytes:
    si = resolve_interface(m)
    if si.stage != "vertex":
        raise LowerError("lower_vertex called on non-vertex module")
    T = _templates()
    types = resolve_types(m)
    stores = _store_map(m)
    p = VertexProgram()

    # --- phase 1: vfetch every input, assign registers in fetch order ---
    input_reg = {}
    reg = 0
    for v, usage, uidx in si.vfetch:
        p.fetch(usage, T["vf"], dest=reg, usage_index=uidx)
        input_reg[v.vid] = reg
        reg += 1

    local_vars = {ins.result_id for ins in m.instrs
                  if ins.opcode == 59 and ins.operands[0] == 7}
    local_store = {tgt: val for tgt, val in stores.items()
                   if tgt in local_vars}
    ctx = ValueLowerer(m=m, si=si, program=p, templates=T, types=types,
                       stores=stores, first_reg=reg, inputs=input_reg,
                       local_store=local_store, stage="vertex",
                       allow_literals=False)

    # --- phase 2: position ---
    p.alloc_position()
    pos_value = None
    for tgt, val in stores.items():
        ac = _access_chain_base(m, tgt)
        if tgt == si.position_out or (ac and ac[0] == si.position_out):
            pos_value = val
    if pos_value is None:
        raise LowerError("no write to gl_Position")
    ctx.emit_value(pos_value, OPOS, True, export_position=True)

    # --- phase 3: interpolator exports ---
    p.alloc_interpolators()
    for v, o, usage in si.interp_out:
        if v.vid not in stores:
            raise LowerError(f"interpolator {v.name} never written")
        r = ctx.operand(stores[v.vid])
        if r[0] == "const":
            raise LowerError("vertex interpolator export of constants (v1)")
        p.export(o, usage, T["mov"], src1_reg=r[1], src2_reg=r[1])

    constants = []
    for cs in si.constants:
        if cs.regs > 1:
            constants.append(ce.Constant(cs.name, ce.RS_FLOAT4, cs.reg,
                                         cs.regs, ce.CLASS_MATRIX_COLS,
                                         ce.TYPE_FLOAT, rows=4, columns=4))
        else:
            constants.append(ce.Constant(cs.name, ce.RS_FLOAT4, cs.reg, 1,
                                         ce.CLASS_VECTOR, ce.TYPE_FLOAT,
                                         rows=1, columns=4))
    return p.finish(constants=constants)


def lower_fragment(m: spirv.Module) -> bytes:
    si = resolve_interface(m)
    if si.stage != "pixel":
        raise LowerError("lower_fragment called on non-pixel module")
    T = _templates()
    types = resolve_types(m)
    stores = _store_map(m)
    p = PixelProgram()

    interp_reg = {}
    maxr = -1
    for v, r, usage in si.interp_in:
        p.input(r, usage)
        interp_reg[v.vid] = r
        maxr = max(maxr, r)
    sample_reg = {}       # ImageSample ssa-id -> register (phase 1)
    used_samplers = set()

    # Literal pool starts after uniform-declared constants.
    lit_base = max((cs.reg + cs.regs for cs in si.constants), default=0)

    # Function-local variables (vec4 tmp = ...): store-forward in branch-free.
    local_vars = {ins.result_id for ins in m.instrs
                  if ins.opcode == 59 and ins.operands[0] == 7}
    local_store = {tgt: val for tgt, val in stores.items()
                   if tgt in local_vars}

    ctx = ValueLowerer(m=m, si=si, program=p, templates=T, types=types,
                       stores=stores, first_reg=maxr + 1,
                       inputs=interp_reg, local_store=local_store,
                       stage="fragment", sample_reg=sample_reg,
                       literal_base=lit_base, allow_literals=True)

    # phase 1 (before alloc_colors): emit a tfetch for every ImageSample
    # reachable from the output expression (walk the whole DAG).
    seen = set()

    def gather(ssa_id: int):
        if ssa_id in seen:
            return
        seen.add(ssa_id)
        ins = m.by_result.get(ssa_id)
        if ins is None:
            return
        if ins.opcode == 87:                      # ImageSample sampler coord
            samp = m.by_result.get(ins.operands[0])
            srole = si.roles.get(samp.operands[0]) if samp else None
            if not srole or srole[0] != "sampler":
                raise LowerError("ImageSample needs a sampler")
            coord = ins.operands[1]
            cins = m.by_result.get(coord)
            if not cins or cins.opcode != 61 or \
                    cins.operands[0] not in interp_reg:
                raise LowerError("texture coord must be an interpolator")
            dst = ctx.fresh()
            p.tfetch(dict(T["tf"], fetch_constant_index=srole[1]),
                     dest=dst, src=interp_reg[cins.operands[0]])
            sample_reg[ssa_id] = dst
            used_samplers.add(srole[1])
            return
        if ins.opcode == 61 and ins.operands[0] == merged_local:
            return
        if ins.opcode == 61 and ins.operands[0] in local_store:
            gather(local_store[ins.operands[0]])  # Load of a local
            return
        if ins.opcode == 65:                      # AccessChain: recurse base
            if ins.operands[0] == merged_local:
                return
            if ins.operands[0] in local_store:
                gather(local_store[ins.operands[0]])
            return
        # generic: recurse into any operand that is an SSA result
        skip = 2 if ins.opcode == 12 else 0       # ExtInst: set + inst num
        for w in ins.operands[skip:]:
            if w in m.by_result:
                gather(w)

    out_var = si.color_out[0][0].vid

    # control flow: structured if/else, possibly NESTED, that assigns one
    # common variable in every arm lowers branch-free to chained Xenos cnd*
    # selects (no jumps). The common variable can be o_color directly, or a
    # Function-local later loaded by the post-merge output expression.
    def _block(label):
        """Basic block starting at `label`: instrs after the Label through the
        terminator (Branch / BranchConditional / Kill / Return)."""
        out, inb = [], False
        for ins in m.instrs:
            if ins.opcode == 248 and ins.result_id == label:
                inb = True
                continue
            if inb:
                out.append(ins)
                if ins.opcode in (249, 250, 252, 253, 254):
                    break
        return out

    def _leaf_store_region(blk, target):
        stores_here = [(i.operands[0], i.operands[1])
                       for i in blk if i.opcode == 62]
        if target is not None:
            vals = [v for t, v in stores_here if t == target]
            if not vals:
                name = "o_color" if target == out_var else f"%{target}"
                raise LowerError(f"if/else arm does not assign {name}")
            return ("val", vals[-1]), target

        # v1: accept arms whose meaningful assignment is either the output or
        # a Function-local. Ignore stores to unrelated storage classes.
        candidates = [(t, v) for t, v in stores_here
                      if t == out_var or t in local_vars]
        targets = []
        for t, _v in candidates:
            if t not in targets:
                targets.append(t)
        if len(targets) != 1:
            raise LowerError("if/else arm does not assign o_color")
        target = targets[0]
        vals = [v for t, v in candidates if t == target]
        return ("val", vals[-1]), target

    def _region(label, target=None):
        """A structured region -> ('val', ssa) leaf, or
        ('sel', cmp_instr, true_region, false_region) for a nested if/else,
        plus the variable all leaves assign."""
        blk = _block(label)
        brc = next((i for i in blk if i.opcode == 250), None)
        if brc is not None:                       # nested selection header
            cond, t_lbl, f_lbl = brc.operands
            treg, t_target = _region(t_lbl, target)
            freg, f_target = _region(f_lbl, target or t_target)
            if t_target != f_target:
                raise LowerError("if/else arms assign different variables")
            return ("sel", m.by_result[cond], treg, freg), t_target
        return _leaf_store_region(blk, target)

    def _selection_region(brc):
        cond, t_lbl, f_lbl = brc.operands
        treg, target = _region(t_lbl)
        freg, f_target = _region(f_lbl, target)
        if target != f_target:
            raise LowerError("if/else arms assign different variables")
        return ("sel", m.by_result[cond], treg, freg), target

    def _gather_region(region):
        if region[0] == "val":
            gather(region[1])
        else:
            gather(region[1].operands[0])
            gather(region[1].operands[1])
            _gather_region(region[2])
            _gather_region(region[3])

    def _has_kill(label):
        return any(i.opcode == 252 for i in _block(label))

    top_brc = next((i for i in m.instrs if i.opcode == 250), None)
    region = None
    region_target = None
    merged_local = None
    kill_cmp = None
    kill_on_true = True
    out_val = None
    if top_brc is not None:
        cond_ssa, t_lbl, f_lbl = top_brc.operands
        t_kill, f_kill = _has_kill(t_lbl), _has_kill(f_lbl)
        if t_kill or f_kill:
            if t_kill and f_kill:
                raise LowerError("if/discard kills both arms")
            kill_cmp = m.by_result[cond_ssa]
            kill_on_true = t_kill
            gather(kill_cmp.operands[0])
            gather(kill_cmp.operands[1])
            if out_var not in stores:
                raise LowerError("o_color never written")
            out_val = stores[out_var]
            gather(out_val)
        else:
            region, region_target = _selection_region(top_brc)
            _gather_region(region)
            if region_target == out_var:
                pass
            elif region_target in local_vars:
                merged_local = region_target
                if out_var not in stores:
                    raise LowerError("o_color never written")
                out_val = stores[out_var]
                gather(out_val)
            else:
                raise LowerError("if/else arms assign unsupported variable")
    else:
        if out_var not in stores:
            raise LowerError("o_color never written")
        out_val = stores[out_var]
        gather(out_val)

    # phase 2 (after alloc_colors): arithmetic, operand-typed.
    p.alloc_colors()

    # comparison opcode -> (cnd vector op, swap operands, swap branches)
    _CMP = {
        186: (_V.kCndGt, False, False),   # a >  b : cndgt(a-b)
        184: (_V.kCndGt, True, False),    # a <  b : cndgt(b-a)
        190: (_V.kCndGe, False, False),   # a >= b : cndge(a-b)
        188: (_V.kCndGe, True, False),    # a <= b : cndge(b-a)
        180: (_V.kCndEq, False, False),   # a == b : cndeq(a-b)
        182: (_V.kCndEq, False, True),    # a != b : cndeq(a-b), swap arms
    }
    # comparison opcode -> (kill vector op, swap operands) for true/false arm
    _KILL_TRUE = {
        186: (_V.kKillGt, False),          # a >  b
        184: (_V.kKillGt, True),           # a <  b -> b > a
        190: (_V.kKillGe, False),          # a >= b
        188: (_V.kKillGe, True),           # a <= b -> b >= a
        180: (_V.kKillEq, False),          # a == b
        182: (_V.kKillNe, False),          # a != b
    }
    _KILL_FALSE = {
        186: (_V.kKillGe, True),           # !(a > b) -> b >= a
        184: (_V.kKillGe, False),          # !(a < b) -> a >= b
        190: (_V.kKillGt, True),           # !(a >= b) -> b > a
        188: (_V.kKillGt, False),          # !(a <= b) -> a > b
        180: (_V.kKillNe, False),          # !(a == b) -> a != b
        182: (_V.kKillEq, False),          # !(a != b) -> a == b
    }

    def _bsw(c):
        return 0 if c == "B" else ctx.bsw(c)

    def _emit_kill(cmp, on_true):
        table = _KILL_TRUE if on_true else _KILL_FALSE
        if cmp.opcode not in table:
            raise LowerError(f"v1: unsupported discard comparison {cmp.name}")
        killop, swap_ops = table[cmp.opcode]
        oa, ca = ctx.scalar_origin(cmp.operands[0])
        ob, cb = ctx.scalar_origin(cmp.operands[1])
        if swap_ops:
            (oa, ca), (ob, cb) = (ob, cb), (oa, ca)
        ctx.emit(killop, {1: (oa, False, _bsw(ca)),
                          2: (ob, False, _bsw(cb))},
                 0, False, mask=0)

    def _region_value(region):
        """Operand holding a region's value: a leaf resolves through operand()
        directly (no extra mov); a nested select / general construct spills to
        a temp."""
        if region[0] == "sel":
            r = ctx.fresh()
            _emit_region(region, r, False)
            return ("reg", r)
        try:
            return ctx.operand(region[1])
        except LowerError:                        # e.g. general construct
            r = ctx.fresh()
            ctx.emit_value(region[1], r, False)
            return ("reg", r)

    def _emit_region(region, dst, export):
        """Emit a region tree: a leaf value, or chained cnd* selects for
        nested if/else."""
        if region[0] == "val":
            ctx.emit_value(region[1], dst, export)
            return
        _, cmp, treg, freg = region
        if cmp.opcode not in _CMP:
            raise LowerError(f"v1: unsupported comparison {cmp.name}")
        cndop, swap_ops, swap_br = _CMP[cmp.opcode]
        oa, ca = ctx.scalar_origin(cmp.operands[0])
        ob, cb = ctx.scalar_origin(cmp.operands[1])
        if swap_ops:
            (oa, ca), (ob, cb) = (ob, cb), (oa, ca)
        if swap_br:
            treg, freg = freg, treg
        d = ctx.fresh()                           # d = a - b
        ctx.emit(_V.kAdd, {1: (oa, False, _bsw(ca)),
                           2: (ob, True, _bsw(cb))}, d, False)
        tv = _region_value(treg)
        fv = _region_value(freg)
        if tv[0] == "const" and fv[0] == "const":
            fv = ctx.to_reg(fv)                   # cnd allows <=1 const/op
        ctx.emit(cndop, {1: (("reg", d), False), 2: (tv, False),
                         3: (fv, False)}, dst, export)

    if kill_cmp is not None:
        _emit_kill(kill_cmp, kill_on_true)

    if region is not None and region_target == out_var:
        _emit_region(region, 0, True)
    elif region is not None:
        r = ctx.fresh()
        _emit_region(region, r, False)
        ctx.local_values[merged_local] = ("reg", r)
        ctx.emit_value(out_val, 0, True)
    else:
        ctx.emit_value(out_val, 0, True)

    constants = []
    for slot in sorted(used_samplers):
        constants.append(ce.Constant(f"tex{slot}", ce.RS_SAMPLER, slot, 1,
                                     ce.CLASS_OBJECT, 0xC, rows=1, columns=1))
    # fragment uniform constants (e.g. u_blend vec4)
    for cs in si.constants:
        constants.append(ce.Constant(cs.name, ce.RS_FLOAT4, cs.reg, cs.regs,
                                     ce.CLASS_VECTOR, ce.TYPE_FLOAT,
                                     rows=1, columns=4))
    # compiler-allocated literal constants (host uploads per lit_manifest)
    for reg, _val in ctx.lit_manifest:
        constants.append(ce.Constant(f"_lit{reg}", ce.RS_FLOAT4, reg, 1,
                                     ce.CLASS_VECTOR, ce.TYPE_FLOAT,
                                     rows=1, columns=4))
    return p.finish(constants=constants), ctx.lit_manifest


def lower_ex(m: spirv.Module):
    """Returns (container_bytes, literal_manifest). The manifest is
    [(constant_register, (f0,f1,f2,f3))] the host must upload for any GLSL
    literals the shader used (empty for vertex / literal-free shaders)."""
    unroll(m)  # expand constant-trip loops to straight-line SPIR-V (no-op if none)
    stage = "vertex" if m.entry_points[0][0] == 0 else "pixel"
    if stage == "vertex":
        return lower_vertex(m), []
    return lower_fragment(m)


def lower(m: spirv.Module) -> bytes:
    return lower_ex(m)[0]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("spv")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()
    m = spirv.Module.parse(Path(args.spv).read_bytes())
    blob = lower(m)
    if args.out:
        Path(args.out).write_bytes(blob)
        print(f"wrote {args.out} ({len(blob)} B)")
    c = container.Container(blob)
    print(xu.disassemble(c.ucode_dwords, c.shader_type))


if __name__ == "__main__":
    main()
