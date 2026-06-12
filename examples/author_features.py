#!/usr/bin/env python3
"""Authors micro test-shaders for two Xenos ISA features the toolchain can
encode but has never proven on hardware:

  feat_cjmp.bin    boolean constant (b128) + conditional-jump control flow.
                   Modeled byte-for-byte on the corpus ps_fb0.bin CF shape:
                   cjmp !b128 -> L; arm A exports PRIM (c0); label L; arm B
                   exports ENV (c1). Console test: toggling
                   Xe_SetPixelShaderConstantB(0) must flip the whole cube
                   between the two solid colors.

  feat_coissue.bin scalar co-issue: one ALU instruction carrying both a
                   vector op (mul oC0.xyz_ = texel.rgb * shade.rgb) and a
                   scalar op (maxs oC0.___w = texel.a). Scalar fields are
                   copied verbatim from corpus cube_ps.bin slot 4, sources
                   remapped to r2=texel / r1=shade.

CF for feat_cjmp is hand-built (the Program builder cannot express jumps);
feat_coissue uses PixelProgram. All instruction field templates come from
the golden corpus via the verified decoders, and the cjmp builder is proven
by reassembling ps_fb0's own cjmp word byte-identically first.

Outputs examples/out/feat_*.bin (+ .hlsl when --recomp / XenosRecomp found).
"""
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import xdis.container as container  # noqa: E402
import xdis.container_emit as ce  # noqa: E402
import xdis.xenos_ucode as xu  # noqa: E402
from xasm import encode as enc  # noqa: E402
from xasm.program import PixelProgram  # noqa: E402
from author_passthrough import alu_fields, corpus_records, pack_program  # noqa: E402
from author_textured import tf_fields  # noqa: E402

OUT = ROOT / "examples" / "out"
K = xu.ControlFlowOpcode

RECOMP_EXE = ROOT / "reference" / "XenosRecomp" / "build" / "XenosRecomp" / \
    "XenosRecomp.exe"
RECOMP_HDR = ROOT / "reference" / "XenosRecomp" / "XenosRecomp" / \
    "shader_common.h"

# b register namespace: D3D unified bool space puts pixel-shader bools at
# 128+; corpus ps_fb0's 'use_tex' (RS_BOOL register 0) appears in the ucode
# as bool_address 128.  Xe_SetPixelShaderConstantB(0, v) drives b128.
PS_BOOL_BASE = 128


# --- proof: rebuild ps_fb0's cjmp CF word byte-identically -------------------

def prove_cjmp_inversion(ps):
    """ps_fb0 cf0 is 'cjmp !b128, L4' = (0x4, 0xb200).  Rebuild it from
    first principles with build_cf_cond_jmp before trusting the builder."""
    raw = (0x4, 0xb200)
    built = enc.build_cf_cond_jmp(
        opcode=K.kCondJmp, address=4, is_unconditional=False,
        is_predicated=False, direction=0, bool_address=PS_BOOL_BASE,
        condition=False, addressing_mode=0)
    assert tuple(built) == raw, \
        f"cjmp inversion: built {tuple(map(hex, built))} != raw {raw}"
    print("cjmp inversion proof: ps_fb0 CF word rebuilt byte-identically")


# --- feat_cjmp ---------------------------------------------------------------

def author_cjmp(ps) -> bytes:
    """cjmp !b128, L3
       alloc colors; exece{ max oC0, c0, c0 }   # arm A: PRIM (b128 true)
       label L3:
       alloc colors; exece{ max oC0, c1, c1 }   # arm B: ENV  (b128 false)
    """
    # template: ps_fb0 slot 4 'max oC0, r0, r0' (is_export, mask xyzw);
    # flip both vector sources from temp r0 to constant cN.
    mov_t = alu_fields(xu.AluInstruction(*ps["alu4"].raw))

    def mov_const(cidx):
        return enc.build_alu(**dict(
            mov_t,
            src1_reg=cidx, src1_is_temp=False,
            src2_reg=cidx, src2_is_temp=False))

    slots = [
        mov_const(0),   # slot 3: oC0 <- c0 (prim)
        mov_const(1),   # slot 4: oC0 <- c1 (env)
    ]

    # CF layout (6 entries -> 3 CF slots, so first instruction slot is 3):
    #   cf0 cjmp  cf1 alloc  cf2 exece(slot3)  cf3 alloc(=L3)
    #   cf4 exece(slot4)  cf5 cnop
    # cjmp fields copied from ps_fb0 cf0 (direction=0 forward,
    # bool_address=128, condition=False => jump taken when b128 is FALSE,
    # addressing_mode=0) with only the target CF index changed: 4 -> 3.
    cf = [
        enc.build_cf_cond_jmp(opcode=K.kCondJmp, address=3,
                              is_unconditional=False, is_predicated=False,
                              direction=0, bool_address=PS_BOOL_BASE,
                              condition=False, addressing_mode=0),
        enc.build_cf_alloc(opcode=K.kAlloc, size=0, is_unserialized=False,
                           alloc_type=2),
        enc.build_cf_exec(opcode=K.kExecEnd, address=3, count=1,
                          is_yield=False, sequence=0, vc_hi=0, vc_lo=0,
                          is_predicate_clean=True, addressing_mode=0),
        enc.build_cf_alloc(opcode=K.kAlloc, size=0, is_unserialized=False,
                           alloc_type=2),                       # label L3
        enc.build_cf_exec(opcode=K.kExecEnd, address=4, count=1,
                          is_yield=False, sequence=0, vc_hi=0, vc_lo=0,
                          is_predicate_clean=True, addressing_mode=0),
        (0, 0),  # cnop
    ]
    ucode = pack_program(cf, slots, first_slot=3)

    # container: mirror PixelProgram.finish() conventions; interpolators
    # texcoord->r0 (nibble 0x3), color->r1 (nibble 0xF); max reg index 1.
    max_reg = 1
    return ce.build_container(
        "pixel", ucode,
        constants=[
            ce.Constant("prim", ce.RS_FLOAT4, 0, 1, ce.CLASS_VECTOR,
                        ce.TYPE_FLOAT, rows=1, columns=4),
            ce.Constant("env", ce.RS_FLOAT4, 1, 1, ce.CLASS_VECTOR,
                        ce.TYPE_FLOAT, rows=1, columns=4),
            # mirrors corpus 'use_tex': RS_BOOL reg 0, scalar bool 1x1
            ce.Constant("use_prim", ce.RS_BOOL, 0, 1, ce.CLASS_SCALAR,
                        ce.TYPE_BOOL, rows=1, columns=1),
        ],
        program_control=(max_reg << 8) | (2 << 27),
        context_misc=0x8,
        field10=0x20000,
        interpolators=[
            ce.Interpolator(0, ce.USAGE_TEXCOORD, 0, flags=0x3),
            ce.Interpolator(1, ce.USAGE_COLOR, 0, flags=0xF),
        ],
        ps_field18=(2 << 16) | 7,
        ps_outputs=0x1)


# --- feat_coissue ------------------------------------------------------------

def author_coissue(ps, cube) -> bytes:
    """tfetch2D r2 <- tex0 at r0.xy; then ONE co-issued ALU instruction:
         mul  oC0.xyz_, r2.xyzz, r1.xyzz   (vector pipe: rgb = texel*shade)
       + maxs oC0.___w, r2.ww              (scalar pipe: a = texel alpha)
    """
    tf_t = tf_fields(xu.TextureFetchInstruction(*ps["tfetch3"].raw))
    # cube_ps slot 4 is the golden co-issue instruction:
    #   mul oC0.xyz_, r0.xxxx, r1.xyzz  +  maxs oC0.___w, r1.ww
    # Keep every scalar-pipe field verbatim (scalar_opcode=kMaxs,
    # scalar_write_mask=w, src3 swizzle .ww); remap sources to our register
    # contract: r2 = texel, r1 = shade.
    co_t = alu_fields(xu.AluInstruction(*cube["alu4"].raw))
    co = dict(co_t,
              src1_reg=2, src1_swizzle=co_t["src2_swizzle"],  # r2.xyzz texel
              src2_reg=1,                                     # r1.xyzz shade
              src3_reg=2)                                     # r2.ww  texel.a

    p = PixelProgram()
    p.input(0, ce.USAGE_TEXCOORD)
    p.input(1, ce.USAGE_COLOR)
    p.tfetch(tf_t, dest=2, src=0)
    p.alloc_colors()
    p.alu(**co)
    return p.finish(constants=[
        ce.Constant("tex0", ce.RS_SAMPLER, 0, 1, ce.CLASS_OBJECT,
                    ce.TYPE_SAMPLER, rows=1, columns=1)])


# --- feat_pred (stretch) -----------------------------------------------------

def author_pred(ps) -> bytes:
    """ALU-level predication (no corpus reference for setp -- decoder-derived):
       exec { setp_ne p0 <- c2.w        # p0 = (pred_src.w != 0)
              (p0)  max r3, c0, c0      # predicated: r3 = prim
              (!p0) max r3, c1, c1 }    # predicated: r3 = env
       alloc colors; exece { max oC0, r3, r3 }
    TODO(ambiguity): no golden-corpus setp instruction exists; the setp_ne
    operand convention (scalar pipe reads src3, both write masks 0) is taken
    from the decoder/xenia model and validated against XenosRecomp only --
    this is exactly what the console test must confirm.
    """
    mov_t = alu_fields(xu.AluInstruction(*ps["alu4"].raw))

    p = PixelProgram()
    p.input(0, ce.USAGE_TEXCOORD)
    p.input(1, ce.USAGE_COLOR)
    # setp_ne: scalar pipe only; vector pipe is the template kMax with write
    # mask 0 (no destination written anywhere; only p0 is set).
    p.alu(**dict(mov_t,
                 is_export=False, vector_dest=0, vector_write_mask=0,
                 scalar_opcode=xu.AluScalarOpcode.kSetpNe,
                 scalar_dest=0, scalar_write_mask=0,
                 src3_reg=2, src3_is_temp=False, src3_swizzle=27))  # c2.ww
    p.alu(**dict(mov_t, is_export=False, vector_dest=3,
                 src1_reg=0, src1_is_temp=False,
                 src2_reg=0, src2_is_temp=False,
                 is_predicated=True, predicate_condition=True))   # (p0) prim
    p.alu(**dict(mov_t, is_export=False, vector_dest=3,
                 src1_reg=1, src1_is_temp=False,
                 src2_reg=1, src2_is_temp=False,
                 is_predicated=True, predicate_condition=False))  # (!p0) env
    p.alloc_colors()
    p.alu(**dict(mov_t, src1_reg=3, src2_reg=3))  # exece max oC0, r3, r3
    return p.finish(constants=[
        ce.Constant("prim", ce.RS_FLOAT4, 0, 1, ce.CLASS_VECTOR,
                    ce.TYPE_FLOAT, rows=1, columns=4),
        ce.Constant("env", ce.RS_FLOAT4, 1, 1, ce.CLASS_VECTOR,
                    ce.TYPE_FLOAT, rows=1, columns=4),
        ce.Constant("pred_src", ce.RS_FLOAT4, 2, 1, ce.CLASS_VECTOR,
                    ce.TYPE_FLOAT, rows=1, columns=4),
    ])


# --- validation --------------------------------------------------------------

def recomp(bin_path: Path) -> str:
    """Run XenosRecomp over the container; return the HLSL text. Raises on
    nonzero exit."""
    hlsl_path = bin_path.with_suffix(".hlsl")
    r = subprocess.run(
        [str(RECOMP_EXE), str(bin_path), str(hlsl_path), str(RECOMP_HDR)],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"XenosRecomp failed on {bin_path.name} (exit {r.returncode}):\n"
            f"{r.stdout}{r.stderr}")
    return hlsl_path.read_text()


def main():
    ps = corpus_records("ps_fb0")
    cube = corpus_records("cube_ps")
    prove_cjmp_inversion(ps)

    shaders = [
        ("feat_cjmp", author_cjmp(ps)),
        ("feat_coissue", author_coissue(ps, cube)),
        ("feat_pred", author_pred(ps)),
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, blob in shaders:
        path = OUT / f"{name}.bin"
        path.write_bytes(blob)
        c = container.Container(blob)
        print(f"\n=== {name}.bin ({len(blob)} B container, "
              f"{c.sh_size // 12} ucode slots) ===")
        print(xu.disassemble(c.ucode_dwords, c.shader_type))

    if RECOMP_EXE.exists():
        for name, _ in shaders:
            text = recomp(OUT / f"{name}.bin")
            print(f"--- XenosRecomp OK: {name}.hlsl ({len(text)} B)")
    else:
        print(f"XenosRecomp not found at {RECOMP_EXE}; skipped .hlsl")


if __name__ == "__main__":
    main()
