#!/usr/bin/env python3
"""Build the glslgallery shader header.

The gallery compiles GLSL fragment shaders from source, embeds each pixel
shader plus its literal-constant manifest, and also embeds the GLSL vertex
shaders used by selected slots. The app uploads the manifest before drawing.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import xsc.spirv as spirv  # noqa: E402
from xsc.lower import lower_ex, _templates  # noqa: E402
from xasm.program import PixelProgram  # noqa: E402
import xdis.xenos_ucode as xu  # noqa: E402
import xdis.container_emit as ce  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "source" / "xost_shaders.h"
GLSLANG = ROOT / "tools" / "glslang" / "bin" / "glslang.exe"
TD = ROOT / "xsc" / "testdata"

# (label, fragment GLSL source stem, vertex shader array, vertex mode) in
# gallery order. Keep frontier paths late so a hang cannot hide earlier slots.
# Extra hardware-gated proofs are appended after LOOPHW in main().
GALLERY = [
    ("MODULATE", "modulate.frag", "glsl_vs", "VS_MODE_MVP"),
    ("TINT", "tint.frag", "glsl_vs", "VS_MODE_MVP"),
    ("GRAY", "gray.frag", "glsl_vs", "VS_MODE_MVP"),
    ("GLOW", "sat.frag", "glsl_vs", "VS_MODE_MVP"),
    ("BLEND50", "blend50.frag", "glsl_vs", "VS_MODE_MVP"),
    ("IFELSE", "cf_if.frag", "glsl_vs", "VS_MODE_MVP"),
    ("SWIZ", "sw_swizzle.frag", "glsl_vs", "VS_MODE_MVP"),
    ("CLAMP", "clamprange.frag", "glsl_vs", "VS_MODE_MVP"),
    ("FLRABS", "floorabs.frag", "glsl_vs", "VS_MODE_MVP"),
    ("GAMMA", "gamma.frag", "glsl_vs", "VS_MODE_MVP"),
    # Fragment verification slots.
    ("SMOOTH", "verify_smooth.frag", "glsl_vs", "VS_MODE_MVP"),
    ("MOD", "verify_mod.frag", "glsl_vs", "VS_MODE_MVP"),
    ("FRACT", "verify_fract.frag", "glsl_vs", "VS_MODE_MVP"),
    ("TRUNC", "verify_trunc.frag", "glsl_vs", "VS_MODE_MVP"),
    ("ROUND", "verify_round.frag", "glsl_vs", "VS_MODE_MVP"),
    ("CROSS", "verify_cross.frag", "glsl_vs", "VS_MODE_MVP"),
    ("NEST", "verify_nest.frag", "glsl_vs", "VS_MODE_MVP"),
    # Crash suspects / hardware-frontier slots.
    ("SQRT", "verify_sqrt.frag", "glsl_vs", "VS_MODE_MVP"),
    ("SIGN", "verify_sign.frag", "glsl_vs", "VS_MODE_MVP"),
    ("ATAN", "verify_atan.frag", "glsl_vs", "VS_MODE_MVP"),
    ("ASIN", "verify_asin.frag", "glsl_vs", "VS_MODE_MVP"),
    ("ALLOC", "verify_alloc.frag", "glsl_vs", "VS_MODE_MVP"),
    ("LOOP", "loop_stair.frag", "glsl_vs", "VS_MODE_MVP"),
    # Vertex frontend proofs.
    ("VSCHAIN", "modulate.frag", "chainmat_vs", "VS_MODE_CHAIN"),
    ("VSARITH", "modulate.frag", "vtx_arith_vs", "VS_MODE_ARITH"),
    # GLES2 completion-batch fragment proofs (literals auto-uploaded).
    ("TERNARY", "fs_ternary.frag", "glsl_vs", "VS_MODE_MVP"),
    ("VREL", "fs_vrelation.frag", "glsl_vs", "VS_MODE_MVP"),
    ("REFRACT", "fs_refract.frag", "glsl_vs", "VS_MODE_MVP"),
    # Frontier proofs: derivatives, GLSL while-break, if/else-in-loop.
    ("DERIV", "fs_deriv.frag", "glsl_vs", "VS_MODE_MVP"),
    ("LOOPBRK", "loop_break.frag", "glsl_vs", "VS_MODE_MVP"),
    ("LOOPCOND", "loop_cond.frag", "glsl_vs", "VS_MODE_MVP"),
]

# Extra non-literal uniform values to upload per pixel shader:
# {fragment stem: [(constant register, (x, y, z, w))]}
EXTRA = {
    "cf_if.frag": [(0, (0.5, 0.5, 0.5, 0.5))],
    "gamma.frag": [(0, (0.45, 0.45, 0.45, 1.0))],
}


def compile_frag(stem):
    src = TD / stem
    spv = src.with_suffix(src.suffix + ".spv")
    subprocess.run([str(GLSLANG), "-G", "--target-env", "opengl",
                    str(src), "-o", str(spv)], check=True,
                   capture_output=True)
    return lower_ex(spirv.Module.parse(spv.read_bytes()))


def compile_vert(stem):
    src = TD / stem
    spv = src.with_suffix(src.suffix + ".spv")
    subprocess.run([str(GLSLANG), "-G", "--target-env", "opengl",
                    str(src), "-o", str(spv)], check=True,
                   capture_output=True)
    return lower_ex(spirv.Module.parse(spv.read_bytes()))[0]


def c_array(name, data):
    lines = [f"static unsigned char {name}[] __attribute__((aligned(4))) = {{"]
    for i in range(0, len(data), 12):
        lines.append("\t" + ", ".join(f"0x{b:02x}" for b in data[i:i + 12])
                     + ",")
    lines.append("};")
    return "\n".join(lines)


def split_manifest(manifest):
    consts = []
    loop_id = -1
    for item in manifest:
        if item[0] == "loop":
            loop_id = item[1]
        else:
            consts.append(item)
    return consts, loop_id


def append_const_array(parts, idx, consts):
    if not consts:
        return
    rows = ",\n\t".join(
        "{ %d, { %ff, %ff, %ff, %ff } }" % (reg, *vals)
        for reg, vals in consts)
    parts.append(f"static const gallery_const gal_c{idx}[] = {{\n"
                 f"\t{rows}\n}};")


def build_loop_proof():
    """Hand-authored PS with a real hardware loop.

    r2 = c1 (0); for (aL = 0; aL < i0.x; aL++) r2 += c0; oC0 = r2.
    The app uploads loop constant i0 each frame, so brightness should step
    with the animated trip count.
    """
    V = xu.AluVectorOpcode
    T = _templates()

    def alu(op, dest, srcs, export=False):
        f = dict(T["mov"], vector_opcode=op, vector_dest=dest, is_export=export,
                 vector_write_mask=0xF, vector_clamp=False,
                 scalar_write_mask=0)
        for i, (reg, is_temp) in enumerate(srcs, start=1):
            f[f"src{i}_reg"] = reg
            f[f"src{i}_is_temp"] = is_temp
            f[f"src{i}_swizzle"] = 0
            f[f"src{i}_negate"] = False
        return f

    p = PixelProgram()
    p.input(0, 5)
    p.input(1, 10)
    p.alu(**alu(V.kMax, 2, [(1, False), (1, False)]))
    p.begin_loop(loop_id=0)
    p.alu(**alu(V.kAdd, 2, [(2, True), (0, False)]))
    p.end_loop()
    p.alloc_colors()
    p.alu(**alu(V.kMax, 0, [(2, True), (2, True)], export=True))
    consts = [ce.Constant("c0", ce.RS_FLOAT4, 0, 1, ce.CLASS_VECTOR,
                          ce.TYPE_FLOAT, rows=1, columns=4),
              ce.Constant("c1", ce.RS_FLOAT4, 1, 1, ce.CLASS_VECTOR,
                          ce.TYPE_FLOAT, rows=1, columns=4)]
    blob = p.finish(constants=consts)
    manifest = [(0, (0.12, 0.12, 0.12, 0.12)),
                (1, (0.0, 0.0, 0.0, 0.0)),
                ("loop", 0, "i0")]
    return blob, manifest


def build_alconst_proof():
    """Hand-authored PS proving aL-relative constant addressing: c[base + aL].

    r2 = 0; for (aL = 0; aL < i0.x; aL++) r2 += c[0 + aL]; oC0 = r2.
    The host uploads i0 = (count=4, start=0, step=1), so aL is 0,1,2,3 and the
    one loop-body instruction reads a *different* constant register each pass.
    With c0=(.5,0,0), c1=(0,.5,0), c2=(0,0,.5), c3=(.1,.1,.1) the working sum
    is r2=(.6,.6,.6) -- light grey. If aL-relative addressing silently fell
    back to a fixed c0, r2=4*c0=(2,0,0)->clamp(1,0,0) -- pure red. So the
    console verdict is unambiguous: grey = working, red = broken.
    """
    V = xu.AluVectorOpcode
    T = _templates()

    def vec(op, dest, srcs, export=False):
        f = dict(T["mov"], vector_opcode=op, vector_dest=dest, is_export=export,
                 vector_write_mask=0xF, vector_clamp=False, scalar_write_mask=0)
        for i, (reg, is_temp, neg, crel) in enumerate(srcs, start=1):
            f[f"src{i}_reg"] = reg
            f[f"src{i}_is_temp"] = is_temp
            f[f"src{i}_swizzle"] = 0
            f[f"src{i}_negate"] = neg
            f[f"src{i}_const_rel"] = crel
        return f

    p = PixelProgram()
    p.input(0, 5)
    p.input(1, 10)
    # r2 = c0 - c0 = 0 (zero the accumulator without spending a literal slot)
    p.alu(**vec(V.kAdd, 2, [(0, False, False, False),
                            (0, False, True, False)]))
    p.begin_loop(loop_id=0)
    # r2 += c[0 + aL]: src2 is the aL-relative constant (base 0).
    p.alu(**vec(V.kAdd, 2, [(2, True, False, False),
                            (0, False, False, True)]))
    p.end_loop()
    p.alloc_colors()
    p.alu(**vec(V.kMax, 0, [(2, True, False, False),
                            (2, True, False, False)], export=True))
    consts = [ce.Constant(f"c{i}", ce.RS_FLOAT4, i, 1, ce.CLASS_VECTOR,
                          ce.TYPE_FLOAT, rows=1, columns=4) for i in range(4)]
    blob = p.finish(constants=consts)
    manifest = [(0, (0.5, 0.0, 0.0, 0.0)),
                (1, (0.0, 0.5, 0.0, 0.0)),
                (2, (0.0, 0.0, 0.5, 0.0)),
                (3, (0.1, 0.1, 0.1, 0.0)),
                ("loop", 0, "i0")]
    return blob, manifest


def build_whilebrk_proof():
    """Hand-authored PS proving the data-dependent `while`-break: a predicated
    LOOP_END exits early when a value computed in the body crosses a threshold.

    r2 = 0;
    for (aL = 0; aL < i0.x (=16); aL++) {
        r2 += c0 (=1/16 grey);
        r3  = r2 - c2 (threshold .25);
        setp_ge p0 <- r3.x;            # p0 = (r2.x >= .25)
    } break-if (p0 == true);
    oC0 = r2.
    The loop constant caps the trip at 16, but the predicate breaks it at pass
    4 (r2.x reaches .25), so the working result is r2=(.25,.25,.25) -- 25% grey.
    If the predicated break were ignored the loop would run all 16 passes to
    r2=(1,1,1) -- pure white. 25% grey = working, white = broken.
    """
    V = xu.AluVectorOpcode
    S = xu.AluScalarOpcode
    T = _templates()

    def vec(op, dest, srcs, export=False):
        f = dict(T["mov"], vector_opcode=op, vector_dest=dest, is_export=export,
                 vector_write_mask=0xF, vector_clamp=False, scalar_write_mask=0)
        for i, (reg, is_temp, neg) in enumerate(srcs, start=1):
            f[f"src{i}_reg"] = reg
            f[f"src{i}_is_temp"] = is_temp
            f[f"src{i}_swizzle"] = 0
            f[f"src{i}_negate"] = neg
        return f

    p = PixelProgram()
    p.input(0, 5)
    p.input(1, 10)
    p.alu(**vec(V.kMax, 2, [(1, False, False), (1, False, False)]))  # r2 = 0
    p.begin_loop(loop_id=0)
    p.alu(**vec(V.kAdd, 2, [(2, True, False), (0, False, False)]))   # r2 += c0
    p.alu(**vec(V.kAdd, 3, [(2, True, False), (2, False, True)]))    # r3 = r2-c2
    # setp_ge p0 <- r3.xxxx (scalar pipe; vector pipe writes nothing). The
    # scalar setp reads src3; broadcast r3.x to every lane (swizzle .xxxx=0x6c)
    # so the uniform comparison agrees across all 64 invocations -> the loop
    # breaks for the whole quad at once.
    p.alu(**dict(T["mov"], vector_opcode=V.kMax, vector_dest=0,
                 vector_write_mask=0, vector_clamp=False,
                 scalar_opcode=S.kSetpGe, scalar_dest=0, scalar_write_mask=0,
                 src3_reg=3, src3_is_temp=True, src3_swizzle=0x6c))
    p.end_loop(predicated_break=True, break_condition=True)
    p.alloc_colors()
    p.alu(**vec(V.kMax, 0, [(2, True, False), (2, True, False)], export=True))
    consts = [ce.Constant("c0", ce.RS_FLOAT4, 0, 1, ce.CLASS_VECTOR,
                          ce.TYPE_FLOAT, rows=1, columns=4),
              ce.Constant("c1", ce.RS_FLOAT4, 1, 1, ce.CLASS_VECTOR,
                          ce.TYPE_FLOAT, rows=1, columns=4),
              ce.Constant("c2", ce.RS_FLOAT4, 2, 1, ce.CLASS_VECTOR,
                          ce.TYPE_FLOAT, rows=1, columns=4)]
    blob = p.finish(constants=consts)
    manifest = [(0, (0.0625, 0.0625, 0.0625, 0.0)),  # 1/16 grey step
                (1, (0.0, 0.0, 0.0, 0.0)),            # zero init
                (2, (0.25, 0.25, 0.25, 0.0)),         # break threshold
                ("loop", 0, "i0")]
    return blob, manifest


def main():
    parts = ["/* generated by gen_header.py - GLSL gallery compiled by xsc; "
             "do not edit. */", ""]

    blobs = {
        "glsl_vs": compile_vert("spincube.vert"),
        "chainmat_vs": compile_vert("chainmat.vert"),
        "vtx_arith_vs": compile_vert("vtx_arith.vert"),
        "cubedir_vs": compile_vert("cubedir.vert"),
    }
    for name in ("xost_vs1", "xost_ps0"):
        blobs[name] = (ROOT / "examples" / "out" / f"{name}.bin").read_bytes()
    for name, data in blobs.items():
        parts.append(c_array(name, data))
        parts.append("")

    entries = []
    for i, (label, stem, vs_name, vs_mode) in enumerate(GALLERY):
        blob, manifest = compile_frag(stem)
        consts, loop_id = split_manifest(manifest)
        consts += EXTRA.get(stem, [])
        parts.append(c_array(f"gal_ps{i}", blob))
        append_const_array(parts, i, consts)
        parts.append("")
        entries.append((label, i, len(consts), vs_name, vs_mode, loop_id))

    li = len(GALLERY)
    lblob, lmanifest = build_loop_proof()
    lconsts, lloop_id = split_manifest(lmanifest)
    parts.append(c_array(f"gal_ps{li}", lblob))
    append_const_array(parts, li, lconsts)
    parts.append("")
    entries.append(("LOOPHW", li, len(lconsts), "glsl_vs", "VS_MODE_MVP",
                    lloop_id))

    di = len(entries)
    dblob, dmanifest = compile_frag("fs_discard.frag")
    dconsts, dloop_id = split_manifest(dmanifest)
    parts.append(c_array(f"gal_ps{di}", dblob))
    append_const_array(parts, di, dconsts)
    parts.append("")
    entries.append(("DISCARD", di, len(dconsts), "glsl_vs", "VS_MODE_MVP",
                    dloop_id))

    ci = len(entries)
    cblob, cmanifest = compile_frag("fs_cubemap.frag")
    cconsts, cloop_id = split_manifest(cmanifest)
    parts.append(c_array(f"gal_ps{ci}", cblob))
    append_const_array(parts, ci, cconsts)
    parts.append("")
    entries.append(("CUBEMAP", ci, len(cconsts), "cubedir_vs",
                    "VS_MODE_MVP", cloop_id))

    gi = len(entries)
    gblob, gmanifest = compile_frag("loop_uniform.frag")
    gconsts, gloop_id = split_manifest(gmanifest)
    parts.append(c_array(f"gal_ps{gi}", gblob))
    append_const_array(parts, gi, gconsts)
    parts.append("")
    entries.append(("GLSLLOOP", gi, len(gconsts), "glsl_vs",
                    "VS_MODE_MVP", gloop_id))

    ai = len(entries)
    ablob, amanifest = build_alconst_proof()
    aconsts, aloop_id = split_manifest(amanifest)
    parts.append(c_array(f"gal_ps{ai}", ablob))
    append_const_array(parts, ai, aconsts)
    parts.append("")
    entries.append(("ALCONST", ai, len(aconsts), "glsl_vs", "VS_MODE_MVP",
                    aloop_id))

    wi = len(entries)
    wblob, wmanifest = build_whilebrk_proof()
    wconsts, wloop_id = split_manifest(wmanifest)
    parts.append(c_array(f"gal_ps{wi}", wblob))
    append_const_array(parts, wi, wconsts)
    parts.append("")
    entries.append(("WHILEBRK", wi, len(wconsts), "glsl_vs", "VS_MODE_MVP",
                    wloop_id))

    # LOOPARR: GLSL `vec4 s=0; for(i<u_count) s += u_w[i]; o=s;` -> a real loop
    # whose body reads the aL-relative array element c[base+aL]. The host fills
    # the array (base..base+3) with four distinct colors; their sum is light
    # grey if the frontend wired the aL-relative read, else wrong/black.
    ri = len(entries)
    rblob, rman = compile_frag("loop_array.frag")
    arr_base = next(it[1] for it in rman if it[0] == "array")
    rloop = next(it[1] for it in rman if it[0] == "loop")
    rconsts = [(arr_base + 0, (0.4, 0.0, 0.0, 0.0)),
               (arr_base + 1, (0.0, 0.4, 0.0, 0.0)),
               (arr_base + 2, (0.0, 0.0, 0.4, 0.0)),
               (arr_base + 3, (0.1, 0.1, 0.1, 0.0))]
    parts.append(c_array(f"gal_ps{ri}", rblob))
    append_const_array(parts, ri, rconsts)
    parts.append("")
    entries.append(("LOOPARR", ri, len(rconsts), "glsl_vs", "VS_MODE_MVP",
                    rloop))

    parts.append("#define GALLERY_N %d" % len(entries))
    parts.append("static const gallery_entry gallery[GALLERY_N] = {")
    for label, i, nman, vs_name, vs_mode, loop_id in entries:
        cptr = f"gal_c{i}" if nman else "0"
        parts.append(f'\t{{ "{label}", gal_ps{i}, {vs_name}, {vs_mode}, '
                     f'{nman}, {cptr}, {loop_id} }},')
    parts.append("};")
    parts.append("")

    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}  ({len(entries) - 3} GLSL slots + "
          "3 hand-authored hardware proofs: LOOPHW, ALCONST, WHILEBRK)")
    for label, i, nman, _vs_name, _vs_mode, loop_id in entries:
        loop = f", loop i{loop_id}" if loop_id >= 0 else ""
        print(f"  gal_ps{i} {label}: {nman} literal const(s){loop}")


if __name__ == "__main__":
    main()
