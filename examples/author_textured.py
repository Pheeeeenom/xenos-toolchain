#!/usr/bin/env python3
"""Authors the textured pipeline via the Program builder API:

  xost_vs2 (vertex): pos*mvp from c0-c3; exports o0=TEXCOORD0, o1=COLOR0
  xost_ps1 (pixel):  tfetch2D tf0 at r0.xy, multiplied by vertex color r1

First customer of xasm.program — instruction field templates extracted from
the golden corpus via the verified decoders.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import xdis.container as container  # noqa: E402
import xdis.container_emit as ce  # noqa: E402
import xdis.xenos_ucode as xu  # noqa: E402
from xasm.program import VertexProgram, PixelProgram  # noqa: E402
from author_passthrough import vf_fields, alu_fields, corpus_records  # noqa: E402

OUT = ROOT / "examples" / "out"

TF_FIELDS = [
    "opcode", "src", "src_swizzle", "is_src_relative", "dest",
    "dest_swizzle", "is_dest_relative", "fetch_constant_index",
    "fetch_valid_only", "unnormalized_coordinates", "mag_filter",
    "min_filter", "mip_filter", "aniso_filter", "arbitrary_filter",
    "vol_mag_filter", "vol_min_filter", "use_computed_lod",
    "use_register_lod", "use_register_gradients", "sample_location",
    "lod_bias", "dimension", "offset_x", "offset_y", "offset_z",
    "is_predicated", "predicate_condition",
]


def tf_fields(t: xu.TextureFetchInstruction) -> dict:
    return {k: getattr(t, k)() for k in TF_FIELDS}


def main():
    vs0 = corpus_records("vs0")
    ps0 = corpus_records("ps_fb0")

    vf_t = vf_fields(xu.VertexFetchInstruction(*vs0["vfetch3"].raw))
    mov_t = alu_fields(xu.AluInstruction(*vs0["alu13"].raw))  # max rX,rX
    dp4_t = [alu_fields(xu.AluInstruction(*vs0[f"alu{s}"].raw))
             for s in (7, 8, 9, 10)]
    tf_t = tf_fields(xu.TextureFetchInstruction(*ps0["tfetch3"].raw))

    # ---- vertex: pos -> r2, color -> r0, texcoord -> r1 ----
    v = VertexProgram()
    v.fetch(ce.USAGE_POSITION, vf_t, dest=2)
    v.fetch(ce.USAGE_COLOR, vf_t, dest=0)
    v.fetch(ce.USAGE_TEXCOORD, vf_t, dest=1)
    v.alloc_position()
    for f in dp4_t:
        v.alu(**f)  # dp4 oPos.{x,y,z,w}, r2.wzxy, c{0-3}.wzxy
    v.alloc_interpolators()
    v.export(0, ce.USAGE_TEXCOORD, mov_t, src1_reg=1, src2_reg=1)
    v.export(1, ce.USAGE_COLOR, mov_t, src1_reg=0, src2_reg=0)
    vs_bin = v.finish(constants=[
        ce.Constant("mvp", ce.RS_FLOAT4, 0, 4, ce.CLASS_MATRIX_COLS,
                    ce.TYPE_FLOAT, rows=4, columns=4)])

    # ---- pixel: oC0 = tex2D(tf0, r0.xy) * r1 ----
    p = PixelProgram()
    p.input(0, ce.USAGE_TEXCOORD)
    p.input(1, ce.USAGE_COLOR)
    p.tfetch(tf_t, dest=2, src=0)
    p.alloc_colors()
    p.alu(**dict(mov_t,
                 vector_opcode=xu.AluVectorOpcode.kMul,
                 vector_dest=0,
                 src1_reg=2, src2_reg=1))
    ps_bin = p.finish(constants=[
        ce.Constant("tex0", ce.RS_SAMPLER, 0, 1, ce.CLASS_OBJECT, 0xC,
                    rows=1, columns=1)])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "xost_vs2.bin").write_bytes(vs_bin)
    (OUT / "xost_ps1.bin").write_bytes(ps_bin)
    print(f"wrote xost_vs2.bin ({len(vs_bin)} B), "
          f"xost_ps1.bin ({len(ps_bin)} B)\n")
    for name, blob in (("xost_vs2", vs_bin), ("xost_ps1", ps_bin)):
        c = container.Container(blob)
        print(f"--- {name}")
        print(xu.disassemble(c.ucode_dwords, c.shader_type))


if __name__ == "__main__":
    main()
