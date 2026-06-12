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
from xsc.resolve import resolve_interface  # noqa: E402
from xasm.program import VertexProgram, PixelProgram  # noqa: E402
from author_passthrough import alu_fields, vf_fields, corpus_records  # noqa: E402
from author_textured import tf_fields  # noqa: E402

OPOS = int(xu.ExportRegister.kVSPosition)  # 62
_V = xu.AluVectorOpcode


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
    stores = _store_map(m)
    p = PixelProgram()

    interp_reg = {}
    maxr = -1
    for v, r, usage in si.interp_in:
        p.input(r, usage)
        interp_reg[v.vid] = r
        maxr = max(maxr, r)
    next_reg = [maxr + 1]
    sample_reg = {}       # ImageSample ssa-id -> register (phase 1)
    used_samplers = set()

    def load_reg(ssa_id: int) -> int:
        ins = m.by_result.get(ssa_id)
        if ins is None or ins.opcode != 61 or ins.operands[0] not in \
                interp_reg:
            raise LowerError(f"expected an interpolator Load (%{ssa_id})")
        return interp_reg[ins.operands[0]]

    out_var = si.color_out[0][0].vid
    if out_var not in stores:
        raise LowerError("o_color never written")
    out_val = stores[out_var]

    # phase 1 (before alloc_colors): emit a tfetch for every ImageSample
    def gather(ssa_id: int):
        ins = m.by_result.get(ssa_id)
        if ins is None:
            return
        if ins.opcode == 87:  # ImageSampleImplicitLod sampler coord
            samp = m.by_result.get(ins.operands[0])
            srole = si.roles.get(samp.operands[0]) if samp else None
            if not srole or srole[0] != "sampler":
                raise LowerError("ImageSample needs a sampler")
            dst = next_reg[0]
            next_reg[0] += 1
            p.tfetch(dict(T["tf"], fetch_constant_index=srole[1]),
                     dest=dst, src=load_reg(ins.operands[1]))
            sample_reg[ssa_id] = dst
            used_samplers.add(srole[1])
        elif ins.opcode == 133:  # FMul
            gather(ins.operands[0])
            gather(ins.operands[1])
    gather(out_val)

    # phase 2 (after alloc_colors): arithmetic
    p.alloc_colors()

    def evalv(ssa_id: int, target: int, export: bool) -> int:
        ins = m.by_result[ssa_id]
        if ins.opcode == 61:                      # Load interpolator
            r = interp_reg[ins.operands[0]]
        elif ins.opcode == 87:                    # pre-fetched sample
            r = sample_reg[ssa_id]
        elif ins.opcode == 133:                   # FMul
            ra = evalv(ins.operands[0], -1, False)
            rb = evalv(ins.operands[1], -1, False)
            dst = target if export else next_reg[0]
            if not export:
                next_reg[0] += 1
            p.alu(**dict(T["mov"], vector_opcode=_V.kMul,
                         vector_dest=dst, is_export=export,
                         src1_reg=ra, src1_is_temp=True,
                         src2_reg=rb, src2_is_temp=True))
            return dst
        else:
            raise LowerError(f"v1 fragment: unhandled op {ins.name}")
        if export:                                # plain value -> oC0 move
            p.alu(**dict(T["mov"], vector_opcode=_V.kMax,
                         vector_dest=target, is_export=True,
                         src1_reg=r, src2_reg=r))
            return target
        return r

    evalv(out_val, 0, export=True)

    constants = []
    for slot in sorted(used_samplers):
        constants.append(ce.Constant(f"tex{slot}", ce.RS_SAMPLER, slot, 1,
                                     ce.CLASS_OBJECT, 0xC, rows=1, columns=1))
    return p.finish(constants=constants)


def lower(m: spirv.Module) -> bytes:
    stage = "vertex" if m.entry_points[0][0] == 0 else "pixel"
    return lower_vertex(m) if stage == "vertex" else lower_fragment(m)


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
