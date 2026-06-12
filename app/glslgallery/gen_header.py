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
# gallery order. Keep never-hardware-tested vertex paths at the end so a hang
# cannot hide earlier fragment slots.
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
                (1, (0.0, 0.0, 0.0, 0.0))]
    return blob, manifest


def main():
    parts = ["/* generated by gen_header.py - GLSL gallery compiled by xsc; "
             "do not edit. */", ""]

    blobs = {
        "glsl_vs": compile_vert("spincube.vert"),
        "chainmat_vs": compile_vert("chainmat.vert"),
        "vtx_arith_vs": compile_vert("vtx_arith.vert"),
    }
    for name in ("xost_vs1", "xost_ps0"):
        blobs[name] = (ROOT / "examples" / "out" / f"{name}.bin").read_bytes()
    for name, data in blobs.items():
        parts.append(c_array(name, data))
        parts.append("")

    entries = []
    for i, (label, stem, vs_name, vs_mode) in enumerate(GALLERY):
        blob, manifest = compile_frag(stem)
        consts = list(manifest) + EXTRA.get(stem, [])
        parts.append(c_array(f"gal_ps{i}", blob))
        if consts:
            rows = ",\n\t".join(
                "{ %d, { %ff, %ff, %ff, %ff } }" % (reg, *vals)
                for reg, vals in consts)
            parts.append(f"static const gallery_const gal_c{i}[] = {{\n"
                         f"\t{rows}\n}};")
        parts.append("")
        entries.append((label, i, len(consts), vs_name, vs_mode))

    li = len(GALLERY)
    lblob, lmanifest = build_loop_proof()
    parts.append(c_array(f"gal_ps{li}", lblob))
    rows = ",\n\t".join("{ %d, { %ff, %ff, %ff, %ff } }" % (reg, *vals)
                        for reg, vals in lmanifest)
    parts.append(f"static const gallery_const gal_c{li}[] = {{\n\t{rows}\n}};")
    parts.append("")
    entries.append(("LOOPHW", li, len(lmanifest), "glsl_vs", "VS_MODE_MVP"))

    parts.append("#define GALLERY_N %d" % len(entries))
    parts.append("static const gallery_entry gallery[GALLERY_N] = {")
    for label, i, nman, vs_name, vs_mode in entries:
        cptr = f"gal_c{i}" if nman else "0"
        parts.append(f'\t{{ "{label}", gal_ps{i}, {vs_name}, {vs_mode}, '
                     f'{nman}, {cptr} }},')
    parts.append("};")
    parts.append("")

    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}  ({len(GALLERY)} GLSL slots + 1 hardware-loop proof)")
    for label, i, nman, _vs_name, _vs_mode in entries:
        print(f"  gal_ps{i} {label}: {nman} literal const(s) to upload")


if __name__ == "__main__":
    main()
