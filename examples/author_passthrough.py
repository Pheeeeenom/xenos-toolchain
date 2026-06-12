#!/usr/bin/env python3
"""Authors the first fully-open Xenos shader pair: a clip-space passthrough
vertex shader (position + color) and a color-passthrough pixel shader.

Every instruction is built with xasm.encode builders; every container byte
comes from container_emit. Field conventions are extracted from the golden
corpus via the verified decoders, and the extraction→builder path is proven
by reassembling corpus instructions byte-identically first.

Output: examples/out/xost_vs0.bin, examples/out/xost_ps0.bin
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

OUT = ROOT / "examples" / "out"

# --- field extraction (decoder accessors -> builder kwargs) -----------------

VF_FIELDS = [
    "opcode", "src", "src_swizzle", "is_src_relative", "dest",
    "dest_swizzle", "is_dest_relative", "must_be_one", "const_index",
    "const_index_sel", "prefetch_count", "data_format", "exp_adjust",
    "is_signed", "is_normalized", "signed_rf_mode", "is_index_rounded",
    "stride", "offset", "is_predicated", "predicate_condition",
]

ALU_FIELDS = [
    "vector_opcode", "scalar_opcode", "vector_dest",
    "is_vector_dest_relative", "vector_write_mask", "vector_clamp",
    "scalar_dest", "is_scalar_dest_relative", "scalar_write_mask",
    "scalar_clamp", "is_export", "abs_constants", "is_predicated",
    "predicate_condition", "is_const_0_addressed", "is_const_1_addressed",
    "is_const_address_register_relative",
]


def vf_fields(v: xu.VertexFetchInstruction) -> dict:
    return {k: getattr(v, k)() for k in VF_FIELDS}


def alu_fields(a: xu.AluInstruction) -> dict:
    d = {k: getattr(a, k)() for k in ALU_FIELDS}
    for i in (1, 2, 3):
        d[f"src{i}_reg"] = a.src_reg(i)
        d[f"src{i}_is_temp"] = a.src_is_temp(i)
        d[f"src{i}_swizzle"] = a.src_swizzle(i)
        d[f"src{i}_negate"] = a.src_negate(i)
    return d


def corpus_records(name: str):
    blob = (ROOT / "corpus" / f"{name}.bin").read_bytes()
    c = container.Container(blob)
    return {r.kind + str(r.offset): r
            for r in xu.parse_program(c.ucode_dwords, c.shader_type)}


# --- prove extraction->builder inversion on real corpus instructions --------

def prove_inversion(vs, ps):
    checks = [
        ("vs vfetch@3", enc.build_vfetch_full(
            **vf_fields(xu.VertexFetchInstruction(*vs["vfetch3"].raw))),
         vs["vfetch3"].raw),
        ("vs max@13", enc.build_alu(
            **alu_fields(xu.AluInstruction(*vs["alu13"].raw))),
         vs["alu13"].raw),
        ("ps max@4", enc.build_alu(
            **alu_fields(xu.AluInstruction(*ps["alu4"].raw))),
         ps["alu4"].raw),
    ]
    for label, built, raw in checks:
        assert tuple(built) == tuple(raw), \
            f"{label}: builder {tuple(map(hex, built))} != raw " \
            f"{tuple(map(hex, raw))}"
    print("inversion proof: corpus instructions rebuilt byte-identically")


# --- SQ_PROGRAM_CNTL (xenia registers.h layout) ------------------------------

def sq_program_cntl(vs_num_reg=0, ps_num_reg=0, vs_resource=0, ps_resource=0,
                    param_gen=0, vs_export_count=0, vs_export_mode=0,
                    ps_export_mode=0) -> int:
    return (vs_num_reg | (ps_num_reg << 8) | (vs_resource << 16) |
            (ps_resource << 17) | (param_gen << 18) |
            (vs_export_count << 20) | (vs_export_mode << 24) |
            (ps_export_mode << 27))


# --- author ------------------------------------------------------------------

K = xu.ControlFlowOpcode


def author_vs(vs) -> bytes:
    """exec{vfetch pos->r2, vfetch col->r0}; alloc position;
    exec{max oPos<-r2}; alloc interpolators; exece{max o0<-r0}; cnop"""
    pos_vf = vf_fields(xu.VertexFetchInstruction(*vs["vfetch3"].raw))  # ->r2
    col_vf = vf_fields(xu.VertexFetchInstruction(*vs["vfetch6"].raw))  # ->r0
    mov = alu_fields(xu.AluInstruction(*vs["alu13"].raw))  # max oN<-r0 .xyzw

    opos = dict(mov, vector_dest=62, src1_reg=2, src2_reg=2)  # oPos <- r2
    ocol = dict(mov, vector_dest=0)                           # o0   <- r0

    slots = [
        enc.build_vfetch_full(**pos_vf),   # slot 3
        enc.build_vfetch_full(**col_vf),   # slot 4
        enc.build_alu(**opos),             # slot 5
        enc.build_alu(**ocol),             # slot 6
    ]
    cf = [
        enc.build_cf_exec(opcode=K.kExec, address=3, count=2, is_yield=False,
                          sequence=0b0101, vc_hi=0b11, vc_lo=0,
                          is_predicate_clean=True, addressing_mode=0),
        enc.build_cf_alloc(opcode=K.kAlloc, size=0, is_unserialized=False,
                           alloc_type=1),
        enc.build_cf_exec(opcode=K.kExec, address=5, count=1, is_yield=False,
                          sequence=0, vc_hi=0, vc_lo=0,
                          is_predicate_clean=True, addressing_mode=0),
        enc.build_cf_alloc(opcode=K.kAlloc, size=0, is_unserialized=False,
                           alloc_type=2),
        enc.build_cf_exec(opcode=K.kExecEnd, address=6, count=1,
                          is_yield=False, sequence=0, vc_hi=0, vc_lo=0,
                          is_predicate_clean=True, addressing_mode=0),
        (0, 0),  # cnop
    ]
    return pack_program(cf, slots, first_slot=3)


def author_ps(ps) -> bytes:
    """alloc colors; exece{max oC0 <- r0}"""
    mov = alu_fields(xu.AluInstruction(*ps["alu4"].raw))  # max oC0 <- r0
    slots = [enc.build_alu(**mov)]  # slot 1
    cf = [
        enc.build_cf_alloc(opcode=K.kAlloc, size=0, is_unserialized=False,
                           alloc_type=2),
        enc.build_cf_exec(opcode=K.kExecEnd, address=1, count=1,
                          is_yield=False, sequence=0, vc_hi=0, vc_lo=0,
                          is_predicate_clean=True, addressing_mode=0),
    ]
    return pack_program(cf, slots, first_slot=1)


def pack_program(cf_pairs, slots, first_slot: int) -> bytes:
    if len(cf_pairs) % 2:
        cf_pairs.append((0, 0))
    dwords = []
    for i in range(0, len(cf_pairs), 2):
        (a0, a1), (b0, b1) = cf_pairs[i], cf_pairs[i + 1]
        dwords.extend(enc.pack_cf_pair((a0, a1), (b0, b1)))
    assert len(dwords) // 3 == first_slot, \
        f"CF occupies {len(dwords)//3} slots, expected {first_slot}"
    for triple in slots:
        dwords.extend(triple)
    return struct.pack(f">{len(dwords)}I", *dwords)


def main():
    vs, ps = corpus_records("vs0"), corpus_records("ps_fb0")
    prove_inversion(vs, ps)

    vs_ucode = author_vs(vs)
    ps_ucode = author_ps(ps)

    vs_bin = ce.build_container(
        "vertex", vs_ucode,
        constants=[],
        program_control=sq_program_cntl(vs_num_reg=2, vs_resource=1,
                                        vs_export_count=0),
        context_misc=0,
        interpolators=[ce.Interpolator(0, ce.USAGE_COLOR, 0, flags=0x0F)],
        vertex_elements=[
            ce.VertexElement(3, ce.USAGE_POSITION, 0, flags=0x1),
            ce.VertexElement(4, ce.USAGE_COLOR, 0, flags=0x3),
        ],
        vs_tail_count=1,
        export_slots=[6],
    )
    ps_bin = ce.build_container(
        "pixel", ps_ucode,
        constants=[],
        program_control=sq_program_cntl(ps_num_reg=0, ps_export_mode=2),
        context_misc=0x8,
        field10=0x20000,
        interpolators=[ce.Interpolator(0, ce.USAGE_COLOR, 0, flags=0xF)],
        ps_field18=(1 << 16) | 7,  # corpus pattern: (interp_count<<16)|7
        ps_outputs=0x1,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "xost_vs0.bin").write_bytes(vs_bin)
    (OUT / "xost_ps0.bin").write_bytes(ps_bin)
    print(f"wrote xost_vs0.bin ({len(vs_bin)} B), "
          f"xost_ps0.bin ({len(ps_bin)} B)")
    for name, blob in (("xost_vs0", vs_bin), ("xost_ps0", ps_bin)):
        c = container.Container(blob)
        print(f"--- {name} ({c.shader_type}, {c.sh_size} B ucode)")
        print(xu.disassemble(c.ucode_dwords, c.shader_type))


if __name__ == "__main__":
    main()
