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


def lower_vertex(m: spirv.Module) -> bytes:
    si = resolve_interface(m)
    if si.stage != "vertex":
        raise LowerError("lower_vertex called on non-vertex module")
    T = _templates()
    stores = _store_map(m)
    p = VertexProgram()

    # --- phase 1: vfetch every input, assign registers in fetch order ---
    input_reg = {}
    reg = 0
    for v, usage, uidx in si.vfetch:
        p.fetch(usage, T["vf"], dest=reg, usage_index=uidx)
        input_reg[v.vid] = reg
        reg += 1

    def value_reg(ssa_id: int) -> int:
        """Resolve an SSA value that must already live in a register
        (v1: a Load of a vfetch input)."""
        ins = m.by_result.get(ssa_id)
        if ins is None or ins.opcode != 61:  # Load
            raise LowerError(f"v1 expects a Load, got %{ssa_id}")
        src_var = ins.operands[0]
        if src_var not in input_reg:
            raise LowerError(f"value %{ssa_id} not a vertex input")
        return input_reg[src_var]

    def const_base(ssa_id: int) -> tuple:
        """Resolve a Load of a uniform -> (constant base reg, reg count)."""
        ins = m.by_result.get(ssa_id)
        if ins is None or ins.opcode != 61:
            raise LowerError("matrix operand must be a uniform Load")
        role = si.roles.get(ins.operands[0])
        if not role or role[0] != "constant":
            raise LowerError("expected a uniform constant")
        return role[1]

    # --- phase 2: position ---
    p.alloc_position()
    pos_value = None
    for tgt, val in stores.items():
        ac = _access_chain_base(m, tgt)
        if ac and ac[0] == si.position_out:
            pos_value = val
    if pos_value is None:
        raise LowerError("no write to gl_Position")
    pv = m.by_result.get(pos_value)
    if pv is not None and pv.opcode == 145:  # MatrixTimesVector
        mat_base, mat_regs = const_base(pv.operands[0])
        vec_reg = value_reg(pv.operands[1])
        if mat_regs != 4:
            raise LowerError("position matrix must be mat4")
        for i in range(4):
            p.alu(**dict(T["dp4"], vector_dest=OPOS,
                         vector_write_mask=(1 << i),
                         src1_reg=vec_reg, src1_is_temp=True,
                         src2_reg=mat_base + i, src2_is_temp=False))
    else:
        raise LowerError("v1 position must be mat4 * vec4")

    # --- phase 3: interpolator exports (passthrough loads) ---
    p.alloc_interpolators()
    for v, o, usage in si.interp_out:
        if v.vid not in stores:
            raise LowerError(f"interpolator {v.name} never written")
        r = value_reg(stores[v.vid])
        p.export(o, usage, T["mov"], src1_reg=r, src2_reg=r)

    constants = [ce.Constant("u_mvp", ce.RS_FLOAT4, 0, 4,
                             ce.CLASS_MATRIX_COLS, ce.TYPE_FLOAT,
                             rows=4, columns=4)]
    return p.finish(constants=constants)


def lower_fragment(m: spirv.Module) -> bytes:
    si = resolve_interface(m)
    if si.stage != "pixel":
        raise LowerError("lower_fragment called on non-pixel module")
    T = _templates()
    types = resolve_types(m)
    stores = _store_map(m)
    p = PixelProgram()

    def vec_size(ssa):
        t = types.get(m.by_result[ssa].result_type)
        return t.count if t and t.kind == "vector" else 1

    interp_reg = {}
    maxr = -1
    for v, r, usage in si.interp_in:
        p.input(r, usage)
        interp_reg[v.vid] = r
        maxr = max(maxr, r)
    next_reg = [maxr + 1]
    sample_reg = {}       # ImageSample ssa-id -> register (phase 1)
    used_samplers = set()

    # literal constant pool: GLSL literals (0.5, vec4(...)) have no ALU
    # immediate on Xenos, so each lands in a constant register the host must
    # upload (the manifest). Pool starts after uniform-declared constants.
    lit_base = [max((cs.reg + cs.regs for cs in si.constants), default=0)]
    lit_pool = {}         # value4 -> constant register
    lit_manifest = []     # [(reg, (f0,f1,f2,f3))]

    def _scalar_const(ssa):
        bits = m.by_result[ssa].operands[0]
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    def _literal_reg(value4):
        key = tuple(round(x, 9) for x in value4)
        if key in lit_pool:
            return lit_pool[key]
        reg = lit_base[0]
        lit_base[0] += 1
        lit_pool[key] = reg
        lit_manifest.append((reg, tuple(value4)))
        return reg

    # Function-local variables (vec4 tmp = ...): store-forward in branch-free
    local_store = {ins.operands[0]: None for ins in m.instrs
                   if ins.opcode == 59 and ins.operands[0] == 7}
    # actually OpVariable storageclass is operands[0]; collect locals
    local_vars = {ins.result_id for ins in m.instrs
                  if ins.opcode == 59 and ins.operands[0] == 7}
    local_store = {tgt: val for tgt, val in stores.items()
                   if tgt in local_vars}

    def fresh():
        r = next_reg[0]
        next_reg[0] += 1
        return r

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
            dst = fresh()
            p.tfetch(dict(T["tf"], fetch_constant_index=srole[1]),
                     dest=dst, src=interp_reg[cins.operands[0]])
            sample_reg[ssa_id] = dst
            used_samplers.add(srole[1])
            return
        if ins.opcode == 61 and ins.operands[0] in local_store:
            gather(local_store[ins.operands[0]])  # Load of a local
            return
        if ins.opcode == 65:                      # AccessChain: recurse base
            if ins.operands[0] in local_store:
                gather(local_store[ins.operands[0]])
            return
        # generic: recurse into any operand that is an SSA result
        skip = 2 if ins.opcode == 12 else 0       # ExtInst: set + inst num
        for w in ins.operands[skip:]:
            if w in m.by_result:
                gather(w)

    out_var = si.color_out[0][0].vid

    # control flow: a single structured if/else that assigns o_color in both
    # arms lowers branch-free to a Xenos cnd* select (no jumps in v1).
    brc = next((i for i in m.instrs if i.opcode == 250), None)
    cf = None
    if brc is not None:
        cond_ssa, t_lbl, f_lbl = brc.operands

        def _branch_value(label):
            inb, val = False, None
            for ins in m.instrs:
                if ins.opcode == 248 and ins.result_id == label:
                    inb = True
                    continue
                if inb:
                    if ins.opcode in (249, 250, 253):
                        break
                    if ins.opcode == 62 and ins.operands[0] == out_var:
                        val = ins.operands[1]
            return val
        cmp = m.by_result[cond_ssa]
        cf_t, cf_f = _branch_value(t_lbl), _branch_value(f_lbl)
        if cf_t is None or cf_f is None:
            raise LowerError("if/else must assign o_color in both arms")
        cf = (cmp, cf_t, cf_f)
        for s in (cf_t, cf_f, cmp.operands[0], cmp.operands[1]):
            gather(s)
    else:
        if out_var not in stores:
            raise LowerError("o_color never written")
        out_val = stores[out_var]
        gather(out_val)

    # phase 2 (after alloc_colors): arithmetic, operand-typed.
    p.alloc_colors()

    op_cache = {}

    def operand(ssa_id: int):
        """Resolve an SSA value to ('reg', n) or ('const', reg), emitting
        ops for computed values (memoized so a shared value emits once)."""
        if ssa_id in op_cache:
            return op_cache[ssa_id]
        ins = m.by_result[ssa_id]
        op = ins.opcode
        if op == 61:                              # Load
            src = ins.operands[0]
            if src in interp_reg:
                res = ("reg", interp_reg[src])
            elif src in local_store:
                res = operand(local_store[src])
            else:
                role = si.roles.get(src)
                if role and role[0] == "constant":
                    res = ("const", role[1][0])
                else:
                    raise LowerError(f"unresolved Load %{src}")
        elif op == 87:
            res = ("reg", sample_reg[ssa_id])
        elif op == 79:                            # VectorShuffle
            idxs = list(ins.operands[2:])
            if idxs == list(range(len(idxs))):    # identity prefix (.rgb/.xy)
                res = operand(ins.operands[0])
            elif all(i < 4 for i in idxs):        # single-source reorder
                base = operand(ins.operands[0])
                swz = sum(((idxs[pp] - pp) & 3) << (2 * pp)
                          for pp in range(len(idxs)))
                r = fresh()
                _emit(_V.kMax, {1: (base, False, swz), 2: (base, False, swz)},
                      r, False)
                res = ("reg", r)
            else:
                raise LowerError(f"two-source swizzle {idxs} (v1)")
        elif op in (129, 131, 133):               # FAdd FSub FMul
            res = ("reg", _binop(op, ins.operands[0], ins.operands[1]))
        elif op == 142:                           # VectorTimesScalar
            res = ("reg", _vts(ins.operands[0], ins.operands[1]))
        elif op == 148:                           # OpDot
            res = ("reg", _dot(ins.operands[0], ins.operands[1]))
        elif op == 43:                            # OpConstant (scalar lit)
            v = _scalar_const(ssa_id)
            res = ("const", _literal_reg((v, v, v, v)))
        elif op == 44:                            # OpConstantComposite
            comps = [_scalar_const(c) for c in ins.operands]
            comps = (comps + [0.0, 0.0, 0.0, 0.0])[:4]
            res = ("const", _literal_reg(tuple(comps)))
        elif op == 80:                            # CompositeConstruct value
            parts = ins.operands
            if len(set(parts)) == 1:              # broadcast of one scalar
                if m.by_result[parts[0]].opcode == 43:    # literal scalar
                    v = _scalar_const(parts[0])
                    res = ("const", _literal_reg((v, v, v, v)))
                else:                             # e.g. vec3(u_gamma)
                    res = operand(parts[0])       # consumer broadcasts (.xxxx)
            else:
                raise LowerError("v1: only broadcast CompositeConstruct "
                                 "usable as a value")
        elif op == 12:                            # ExtInst
            res = ("reg", _extinst(ins))
        else:
            raise LowerError(f"v1 fragment: unhandled op {ins.name}")
        op_cache[ssa_id] = res
        return res

    def _src(fields, n, opnd, negate=False, swizzle=0):
        kind, val = opnd
        fields[f"src{n}_reg"] = val
        fields[f"src{n}_is_temp"] = (kind == "reg")
        fields[f"src{n}_swizzle"] = swizzle
        fields[f"src{n}_negate"] = negate

    def _emit(opcode, ops, dst, export, mask=0xF, clamp=False):
        n_const = len({v[0][1] for v in ops.values() if v[0][0] == "const"})
        if n_const > 1:
            raise LowerError(">1 constant operand (v1 cannot split here)")
        f = dict(T["mov"], vector_opcode=opcode, vector_dest=dst,
                 is_export=export, vector_write_mask=mask,
                 vector_clamp=clamp)
        for n, v in ops.items():
            opnd, neg = v[0], v[1]
            swz = v[2] if len(v) > 2 else 0
            _src(f, n, opnd, neg, swz)
        p.alu(**f)

    def _const_scalar(ssa):
        """A scalar constant value, possibly wrapped in a broadcast
        CompositeConstruct; returns the float or raises."""
        ins = m.by_result[ssa]
        if ins.opcode == 43:
            return _scalar_const(ssa)
        if ins.opcode == 80 and len(set(ins.operands)) == 1 and \
                m.by_result[ins.operands[0]].opcode == 43:
            return _scalar_const(ins.operands[0])
        raise LowerError(f"%{ssa} is not a constant scalar")

    def _const_val(ssa_id):
        c = m.by_result.get(ssa_id)
        if c is None or c.opcode != 43:
            raise LowerError(f"%{ssa_id} is not an int constant")
        return c.operands[0]

    def scalar_origin(ssa_id):
        """Resolve a scalar SSA to (vector_operand, component). The component
        is an int, or "B" when the value is broadcast across all components
        (a dp3/dp4 result or a scalar constant)."""
        ins = m.by_result[ssa_id]
        if ins.opcode == 81:                      # CompositeExtract(v, idx)
            return operand(ins.operands[0]), ins.operands[1]
        if ins.opcode == 61:                      # Load
            src = ins.operands[0]
            ptr = m.by_result.get(src)
            if ptr is not None and ptr.opcode == 65:  # AccessChain base idx
                base, comp = ptr.operands[0], _const_val(ptr.operands[1])
                if base in local_store:
                    return operand(local_store[base]), comp
                if base in interp_reg:
                    return ("reg", interp_reg[base]), comp
                role = si.roles.get(base)         # uniform component access
                if role and role[0] == "constant":
                    return ("const", role[1][0]), comp
            if src in local_store:                # local holding a scalar val
                return operand(local_store[src]), "B"
        if ins.opcode in (148, 43):               # Dot / OpConstant scalar
            return operand(ssa_id), "B"
        raise LowerError(f"cannot resolve scalar %{ssa_id}")

    def emit_construct(ins, export_reg):
        origins = [scalar_origin(part) for part in ins.operands]
        n = len(origins)
        j = 0
        while j < n:
            op_j = origins[j][0]
            k = j                                 # run: same reg, identity
            while k < n and origins[k][0] == op_j and \
                    origins[k][1] in (k, "B"):    # broadcast fits any slot
                k += 1
            if k > j:
                mask = sum(1 << t for t in range(j, k))
                _emit(_V.kMax, {1: (op_j, False), 2: (op_j, False)},
                      export_reg, True, mask=mask)
                j = k
            else:                                 # singleton: route comp->j
                op, comp = origins[j]
                swz = 0 if comp == "B" else ((comp - j) & 3) << (2 * j)
                _emit(_V.kMax, {1: (op, False, swz), 2: (op, False, swz)},
                      export_reg, True, mask=(1 << j))
                j += 1

    def _binop(op, a, b, dst=None, export=False, clamp=False):
        oa, ob = operand(a), operand(b)
        if dst is None:
            dst = fresh()
        if op == 133:                             # FMul
            _emit(_V.kMul, {1: (oa, False), 2: (ob, False)}, dst, export,
                  clamp=clamp)
        elif op == 129:                           # FAdd
            _emit(_V.kAdd, {1: (oa, False), 2: (ob, False)}, dst, export,
                  clamp=clamp)
        else:                                     # FSub: a + (-b)
            _emit(_V.kAdd, {1: (oa, False), 2: (ob, True)}, dst, export,
                  clamp=clamp)
        return dst

    def _emit_clamped(ssa, dst, export):
        """Lower a value with the saturate (clamp-to-[0,1]) modifier on its
        final instruction."""
        vi = m.by_result[ssa]
        if vi.opcode in (129, 131, 133):
            return _binop(vi.opcode, vi.operands[0], vi.operands[1],
                          dst=dst, export=export, clamp=True)
        r = operand(ssa)
        _emit(_V.kMax, {1: (r, False), 2: (r, False)}, dst, export,
              clamp=True)
        return dst

    # broadcast swizzle: every component reads source component x (.xxxx)
    _BCAST_X = sum(((0 - p) & 3) << (2 * p) for p in range(4))

    def _vts(vec, scal, dst=None, export=False):
        """VectorTimesScalar: vec * scalar, scalar broadcast from its .x.
        (host uploads a scalar uniform in the register's x component)."""
        ov, os = operand(vec), operand(scal)
        if dst is None:
            dst = fresh()
        _emit(_V.kMul, {1: (ov, False), 2: (os, False, _BCAST_X)},
              dst, export)
        return dst

    def _dot(a, b, dst=None, export=False):
        """dot(a,b): dp3 (vec3) or dp4 (vec4); result broadcast to all
        components (write mask 0xF) so downstream reads any component."""
        oa, ob = operand(a), operand(b)
        opcode = _V.kDp3 if vec_size(a) == 3 else _V.kDp4
        if dst is None:
            dst = fresh()
        _emit(opcode, {1: (oa, False), 2: (ob, False)}, dst, export, mask=0xF)
        return dst

    def _emit_scalar(sop, dst, src_op, src_comp=3, mask=0xF):
        """A standalone scalar-pipe op (rsq/sqrt/exp/log/...): the vector half
        is a nop (write mask 0); the scalar half reads ONE component of src3
        and broadcasts its result to `dst`. Returns dst."""
        kind, reg = src_op
        swz = ((src_comp - 3) & 3) << 6           # scalar reads src3 pos-3
        f = dict(T["mov"], vector_opcode=_V.kMax, vector_dest=0,
                 vector_write_mask=0, vector_clamp=False,
                 scalar_opcode=sop, scalar_dest=dst, scalar_write_mask=mask,
                 scalar_clamp=False, is_export=False,
                 src3_reg=reg, src3_is_temp=(kind == "reg"),
                 src3_swizzle=swz, src3_negate=False)
        p.alu(**f)
        return dst

    def _dotself(ov, n):                          # dot(v, v) -> broadcast reg
        d = fresh()
        _emit(_V.kDp3 if n == 3 else _V.kDp4, {1: (ov, False), 2: (ov, False)},
              d, False, mask=0xF)
        return d

    def _extinst(ins, dst=None, export=False):
        inst = ins.operands[1]
        args = ins.operands[2:]
        if dst is None:
            dst = fresh()
        if inst == 46:                            # FMix(x,y,a)=x+(y-x)*a
            ox, oy, oa = (operand(args[0]), operand(args[1]),
                          operand(args[2]))
            tmp = fresh()
            _emit(_V.kAdd, {1: (oy, False), 2: (ox, True)}, tmp, False)
            _emit(_V.kMad, {1: (("reg", tmp), False), 2: (oa, False),
                            3: (ox, False)}, dst, export)
            return dst
        if inst in (40, 37):                      # FMax / FMin
            _emit(_V.kMax if inst == 40 else _V.kMin,
                  {1: (operand(args[0]), False),
                   2: (operand(args[1]), False)}, dst, export)
            return dst
        def _scalar_result(sop, src_op, src_comp=3):
            if export:
                r = _emit_scalar(sop, fresh(), src_op, src_comp)
                _emit(_V.kMax, {1: (("reg", r), False),
                                2: (("reg", r), False)}, dst, True)
            else:
                _emit_scalar(sop, dst, src_op, src_comp)
            return dst

        if inst == 8:                             # Floor
            _emit(_V.kFloor, {1: (operand(args[0]), False),
                              2: (operand(args[0]), False)}, dst, export)
            return dst
        if inst == 69:                            # Normalize(v)= v*rsq(dot)
            ov = operand(args[0])
            d = _dotself(ov, vec_size(args[0]))
            s = _emit_scalar(_S.kRsq, fresh(), ("reg", d))
            _emit(_V.kMul, {1: (ov, False), 2: (("reg", s), False)},
                  dst, export)
            return dst
        if inst == 66:                            # Length(v) = sqrt(dot(v,v))
            ov = operand(args[0])
            d = _dotself(ov, vec_size(args[0]))
            return _scalar_result(_S.kSqrt, ("reg", d))
        if inst == 67:                            # Distance(a,b)=length(a-b)
            oa, ob = operand(args[0]), operand(args[1])
            diff = fresh()
            _emit(_V.kAdd, {1: (oa, False), 2: (ob, True)}, diff, False)
            d = _dotself(("reg", diff), vec_size(args[0]))
            return _scalar_result(_S.kSqrt, ("reg", d))
        if inst == 31:                            # Sqrt (scalar/broadcast arg)
            return _scalar_result(_S.kSqrt, operand(args[0]))
        if inst == 32:                            # InverseSqrt
            return _scalar_result(_S.kRsq, operand(args[0]))
        if inst == 29:                            # Exp2
            return _scalar_result(_S.kExp, operand(args[0]))
        if inst == 30:                            # Log2
            return _scalar_result(_S.kLog, operand(args[0]))
        def _sbcast(c):                           # broadcast component c
            return sum(((c - pp) & 3) << (2 * pp) for pp in range(4))

        def _scalar_src(ssa):                     # (operand, swizzle) of scalar
            o, c = scalar_origin(ssa)
            return o, (0 if c == "B" else _sbcast(c)), (3 if c == "B" else c)

        if inst == 27:                            # Exp = exp2(x*log2 e)
            o, sw, c = _scalar_src(args[0])
            ck = ("const", _literal_reg((1.4426950408,) * 4))
            mm = fresh()
            _emit(_V.kMul, {1: (o, False, sw), 2: (ck, False)}, mm, False)
            return _scalar_result(_S.kExp, ("reg", mm))
        if inst == 28:                            # Log = log2(x)*ln2
            o, sw, c = _scalar_src(args[0])
            lg = _emit_scalar(_S.kLog, fresh(), o, src_comp=c)
            ck = ("const", _literal_reg((0.6931471805,) * 4))
            _emit(_V.kMul, {1: (("reg", lg), False), 2: (ck, False)},
                  dst, export)
            return dst
        if inst in (13, 14):                      # Sin / Cos (range-limited)
            o, sw, c = _scalar_src(args[0])
            return _scalar_result(_S.kSin if inst == 13 else _S.kCos, o, c)
        if inst == 71:                            # Reflect=i-2*dot(n,i)*n
            oi, on = operand(args[0]), operand(args[1])
            d = _dotself(on, vec_size(args[0])) if args[0] == args[1] else None
            if d is None:
                d = fresh()
                _emit(_V.kDp3 if vec_size(args[0]) == 3 else _V.kDp4,
                      {1: (on, False), 2: (oi, False)}, d, False, mask=0xF)
            d2 = fresh()
            _emit(_V.kAdd, {1: (("reg", d), False), 2: (("reg", d), False)},
                  d2, False)
            _emit(_V.kMad, {1: (("reg", d2), True), 2: (on, False),
                            3: (oi, False)}, dst, export)
            return dst
        if inst == 26:                            # Pow(x,y)=exp2(y*log2 x)
            ox, oy = operand(args[0]), operand(args[1])
            n = max(vec_size(args[0]), 1)
            for i in range(n):                    # per component (scalar pipe)
                li = _emit_scalar(_S.kLog, fresh(), ox, src_comp=i)
                mi = fresh()
                _emit(_V.kMul, {1: (("reg", li), False),
                                2: (oy, False, _BCAST_X)}, mi, False)
                ri = _emit_scalar(_S.kExp, fresh(), ("reg", mi))
                _emit(_V.kMax, {1: (("reg", ri), False),
                                2: (("reg", ri), False)}, dst, export,
                      mask=(1 << i))
            return dst
        if inst == 4:                             # FAbs -> max(x, -x)
            ox = operand(args[0])
            _emit(_V.kMax, {1: (ox, False), 2: (ox, True)}, dst, export)
            return dst
        if inst == 43:                            # FClamp(x, lo, hi)
            lo, hi = _const_scalar(args[1]), _const_scalar(args[2])
            if lo == 0.0 and hi == 1.0:           # saturate -> free modifier
                return _emit_clamped(args[0], dst, export)
            # general clamp = min(max(x, lo), hi); lo/hi pooled literals
            ox = operand(args[0])
            lo_op = ("const", _literal_reg((lo, lo, lo, lo)))
            hi_op = ("const", _literal_reg((hi, hi, hi, hi)))
            t = fresh()
            _emit(_V.kMax, {1: (ox, False), 2: (lo_op, False)}, t, False)
            _emit(_V.kMin, {1: (("reg", t), False), 2: (hi_op, False)},
                  dst, export)
            return dst
        raise LowerError(f"v1 fragment: unhandled GLSL.std.450 #{inst}")

    # comparison opcode -> (cnd vector op, swap operands, swap branches)
    _CMP = {
        186: (_V.kCndGt, False, False),   # a >  b : cndgt(a-b)
        184: (_V.kCndGt, True, False),    # a <  b : cndgt(b-a)
        190: (_V.kCndGe, False, False),   # a >= b : cndge(a-b)
        188: (_V.kCndGe, True, False),    # a <= b : cndge(b-a)
        180: (_V.kCndEq, False, False),   # a == b : cndeq(a-b)
        182: (_V.kCndEq, False, True),    # a != b : cndeq(a-b), swap arms
    }

    def _bsw(c):
        return 0 if c == "B" else \
            sum(((c - p) & 3) << (2 * p) for p in range(4))

    if cf is not None:
        # if(a CMP b) oC0=T else oC0=F  ->  d=(a-b); oC0 = cnd(d>0?T:F)
        cmp, cf_t, cf_f = cf
        if cmp.opcode not in _CMP:
            raise LowerError(f"v1: unsupported comparison {cmp.name}")
        cndop, swap_ops, swap_br = _CMP[cmp.opcode]
        oa, ca = scalar_origin(cmp.operands[0])
        ob, cb = scalar_origin(cmp.operands[1])
        if swap_ops:
            (oa, ca), (ob, cb) = (ob, cb), (oa, ca)
        d = fresh()
        _emit(_V.kAdd, {1: (oa, False, _bsw(ca)), 2: (ob, True, _bsw(cb))},
              d, False)
        tv, fv = operand(cf_t), operand(cf_f)
        if swap_br:
            tv, fv = fv, tv
        _emit(cndop, {1: (("reg", d), False), 2: (tv, False),
                      3: (fv, False)}, 0, True)
    else:
        # emit the output expression, final op fused to the oC0 export
        root_ins = m.by_result[out_val]
        if root_ins.opcode == 80:                 # CompositeConstruct
            emit_construct(root_ins, 0)
        elif root_ins.opcode in (129, 131, 133):
            _binop(root_ins.opcode, root_ins.operands[0],
                   root_ins.operands[1], dst=0, export=True)
        elif root_ins.opcode == 142:
            _vts(root_ins.operands[0], root_ins.operands[1], dst=0,
                 export=True)
        elif root_ins.opcode == 12:
            _extinst(root_ins, dst=0, export=True)
        else:                                     # plain value -> oC0
            r = operand(out_val)
            _emit(_V.kMax, {1: (r, False), 2: (r, False)}, 0, True)

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
    for reg, _val in lit_manifest:
        constants.append(ce.Constant(f"_lit{reg}", ce.RS_FLOAT4, reg, 1,
                                     ce.CLASS_VECTOR, ce.TYPE_FLOAT,
                                     rows=1, columns=4))
    return p.finish(constants=constants), lit_manifest


def lower_ex(m: spirv.Module):
    """Returns (container_bytes, literal_manifest). The manifest is
    [(constant_register, (f0,f1,f2,f3))] the host must upload for any GLSL
    literals the shader used (empty for vertex / literal-free shaders)."""
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
