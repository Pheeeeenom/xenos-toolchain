#!/usr/bin/env python3
"""Authors xost_vs1: a matrix-transform vertex shader (position * mvp from
c0-c3, color passthrough) for the spinning-cube demo.

Program (slots):
  3: vfetch position -> r2      (corpus vs0 slot 3, verbatim)
  4: vfetch color    -> r0      (corpus vs0 slot 6, verbatim)
  5-8: dp4 oPos.{x,y,z,w}, r2.wzxy, c{0-3}.wzxy   (corpus slots 7-10, verbatim)
  9: max o0, r0, r0             (corpus slot 13, vdest->0)

Semantics note: oPos[i] = dot4(pos, c[i]) — the app must upload the MVP
matrix such that row i of the column-vector transform lands in constant i.
"""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import xdis.container as container  # noqa: E402
import xdis.container_emit as ce  # noqa: E402
import xdis.xenos_ucode as xu  # noqa: E402
from xasm import encode as enc  # noqa: E402

# reuse the proven extraction helpers
sys.path.insert(0, str(ROOT / "examples"))
from author_passthrough import (  # noqa: E402
    vf_fields, alu_fields, corpus_records, pack_program, sq_program_cntl)

K = xu.ControlFlowOpcode
OUT = ROOT / "examples" / "out"


def main():
    vs = corpus_records("vs0")

    slots = [
        enc.build_vfetch_full(**vf_fields(
            xu.VertexFetchInstruction(*vs["vfetch3"].raw))),   # 3: pos->r2
        enc.build_vfetch_full(**vf_fields(
            xu.VertexFetchInstruction(*vs["vfetch6"].raw))),   # 4: col->r0
    ]
    for slot in (7, 8, 9, 10):  # corpus dp4 oPos.{x,y,z,w} block
        f = alu_fields(xu.AluInstruction(*vs[f"alu{slot}"].raw))
        built = enc.build_alu(**f)
        assert tuple(built) == tuple(vs[f"alu{slot}"].raw), \
            f"dp4 inversion mismatch at corpus slot {slot}"
        slots.append(built)                                    # 5..8
    ocol = dict(alu_fields(xu.AluInstruction(*vs["alu13"].raw)),
                vector_dest=0)
    slots.append(enc.build_alu(**ocol))                        # 9: o0<-r0

    cf = [
        enc.build_cf_exec(opcode=K.kExec, address=3, count=2, is_yield=False,
                          sequence=0b0101, vc_hi=15, vc_lo=0,
                          is_predicate_clean=True, addressing_mode=0),
        enc.build_cf_alloc(opcode=K.kAlloc, size=0, is_unserialized=False,
                           alloc_type=1),
        enc.build_cf_exec(opcode=K.kExec, address=5, count=4, is_yield=False,
                          sequence=0, vc_hi=0, vc_lo=0,
                          is_predicate_clean=True, addressing_mode=0),
        enc.build_cf_alloc(opcode=K.kAlloc, size=0, is_unserialized=False,
                           alloc_type=2),
        enc.build_cf_exec(opcode=K.kExecEnd, address=9, count=1,
                          is_yield=False, sequence=0, vc_hi=0, vc_lo=0,
                          is_predicate_clean=True, addressing_mode=0),
        (0, 0),  # cnop
    ]
    ucode = pack_program(cf, slots, first_slot=3)

    blob = ce.build_container(
        "vertex", ucode,
        constants=[ce.Constant("mvp", ce.RS_FLOAT4, 0, 4,
                               ce.CLASS_MATRIX_COLS, ce.TYPE_FLOAT,
                               rows=4, columns=4)],
        program_control=sq_program_cntl(vs_num_reg=2, vs_resource=1,
                                        vs_export_count=0),
        context_misc=0,
        interpolators=[ce.Interpolator(0, ce.USAGE_COLOR, 0, flags=0x0F)],
        vertex_elements=[
            ce.VertexElement(3, ce.USAGE_POSITION, 0, flags=0x1),
            ce.VertexElement(4, ce.USAGE_COLOR, 0, flags=0x3),
        ],
        vs_tail_count=1,
        export_slots=[9],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "xost_vs1.bin").write_bytes(blob)
    print(f"wrote xost_vs1.bin ({len(blob)} B)")

    c = container.Container(blob)
    print(xu.disassemble(c.ucode_dwords, c.shader_type))


if __name__ == "__main__":
    main()
