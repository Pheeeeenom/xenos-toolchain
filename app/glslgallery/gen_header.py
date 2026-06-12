#!/usr/bin/env python3
"""Compiles a gallery of GLSL fragment shaders FROM SOURCE at build time and
embeds them + their literal-constant manifests for the app to upload.

Each entry: the compiled container bytes, and the host-upload manifest
(constant register + 4 floats) that lower_ex() reported for the shader's GLSL
literals. The app uploads the manifest before drawing — this is what proves
the literal-constant contract on real hardware.

Vertex shader (glsl_vs) and HUD shaders are also embedded.
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

# (label, glsl source stem) — fragment shaders, in gallery order
GALLERY = [
    ("MODULATE", "modulate.frag"),   # tex * color
    ("TINT", "tint.frag"),           # vec4(tex.rgb*color.rgb, tex.a)
    ("GRAY", "gray.frag"),           # luminance desaturate (dot/dp3 + literal)
    ("GLOW", "sat.frag"),            # clamp(tex+color,0,1) -> add_sat
    ("BLEND50", "blend50.frag"),     # mix(tex,color,0.5)  (literal pool)
    ("IFELSE", "cf_if.frag"),        # if(tex.r>thr) color else tex  (cnd!)
    ("SWIZ", "sw_swizzle.frag"),     # tex.bgra * color
    ("CLAMP", "clamprange.frag"),    # clamp(tex*color, 0.2, 0.8)
    ("FLRABS", "floorabs.frag"),     # abs(tex-color)+floor(color)
    ("GAMMA", "gamma.frag"),         # pow(tex, gamma) — scalar pipe log2/exp2
    # --- verification slots (G10+): safe ops first, then crash suspects so a
    #     hang doesn't hide the rest ---
    ("SMOOTH", "verify_smooth.frag"),  # G10 smoothstep (division — now fixed)
    ("MOD", "verify_mod.frag"),        # G11 mod (division — now fixed; ==FRACT)
    ("FRACT", "verify_fract.frag"),    # G12 fract -> 4 sawtooth (native)
    ("TRUNC", "verify_trunc.frag"),    # G13 trunc -> 4 flat steps
    ("ROUND", "verify_round.frag"),    # G14 round -> 4 flat steps
    ("CROSS", "verify_cross.frag"),    # G15 cross product -> color field
    ("NEST", "verify_nest.frag"),      # G16 nested if/else -> RGB quadrants
    # --- crash suspects (never hardware-tested ops), isolated ---
    ("SQRT", "verify_sqrt.frag"),      # G17 isolate kSqrt (scalar pipe)
    ("SIGN", "verify_sign.frag"),      # G18 isolate kSgt (sign compare)
    ("ATAN", "verify_atan.frag"),      # G19 atan: only mul/mad (proven ops)
    ("ASIN", "verify_asin.frag"),      # G20 asin: sqrt+sign+poly (the crasher)
    # --- register allocator proof ---
    ("ALLOC", "verify_alloc.frag"),    # G21 25 GPR -> 5 via allocator; R=x,G=y
    # --- loop unrolling proof (xsc.unroll) ---
    ("LOOP", "loop_stair.frag"),       # G22 5x step() staircase: trip count +
    #                                        induction-fold + post-loop RMW
]

# extra (non-literal) uniform values to upload per shader: {stem: [(reg, 4f)]}
EXTRA = {
    "cf_if.frag": [(0, (0.5, 0.5, 0.5, 0.5))],   # u_threshold = 0.5
    "gamma.frag": [(0, (0.45, 0.45, 0.45, 1.0))],  # u_gamma = 0.45 (brighten)
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
    """Hand-authored PS with a REAL hardware loop (not unrolled) to prove
    LOOP_START/LOOP_END + loop-constant trip count on silicon:
        r0 = c1 (0);  for (aL = 0; aL < i0.x; aL++) r0 += c0;  oC0 = r0;
    The app uploads the trip count i0 each frame (animated), so the cube's
    flat brightness = count * c0 steps as the loop count changes."""
    V = xu.AluVectorOpcode
    T = _templates()

    def alu(op, dest, srcs, export=False):
        f = dict(T["mov"], vector_opcode=op, vector_dest=dest, is_export=export,
                 vector_write_mask=0xF, vector_clamp=False, scalar_write_mask=0)
        for i, (reg, is_temp) in enumerate(srcs, start=1):
            f[f"src{i}_reg"] = reg
            f[f"src{i}_is_temp"] = is_temp
            f[f"src{i}_swizzle"] = 0
            f[f"src{i}_negate"] = False
        return f

    p = PixelProgram()
    p.input(0, 5)                                                # r0 = v_texcoord (match the cube VS)
    p.input(1, 10)                                               # r1 = v_color
    # accumulate in r2 — r0/r1 are reserved for the VS-exported interpolators
    p.alu(**alu(V.kMax, 2, [(1, False), (1, False)]))            # r2 = c1 (0)
    p.begin_loop(loop_id=0)
    p.alu(**alu(V.kAdd, 2, [(2, True), (0, False)]))            # r2 += c0
    p.end_loop()
    p.alloc_colors()                                            # ALLOC pixel export
    p.alu(**alu(V.kMax, 0, [(2, True), (2, True)], export=True))  # oC0 = r2
    consts = [ce.Constant("c0", ce.RS_FLOAT4, 0, 1, ce.CLASS_VECTOR,
                          ce.TYPE_FLOAT, rows=1, columns=4),
              ce.Constant("c1", ce.RS_FLOAT4, 1, 1, ce.CLASS_VECTOR,
                          ce.TYPE_FLOAT, rows=1, columns=4)]
    blob = p.finish(constants=consts)
    manifest = [(0, (0.12, 0.12, 0.12, 0.12)), (1, (0.0, 0.0, 0.0, 0.0))]
    return blob, manifest


def main():
    parts = ["/* generated by gen_header.py - GLSL gallery compiled by xsc; "
             "do not edit. */", ""]

    # vertex + HUD shaders
    blobs = {"glsl_vs": compile_vert("spincube.vert")}
    for name in ("xost_vs1", "xost_ps0"):
        blobs[name] = (ROOT / "examples" / "out" / f"{name}.bin").read_bytes()
    for name, data in blobs.items():
        parts.append(c_array(name, data))
        parts.append("")

    # gallery fragment shaders + manifests (literal pool + extra uniforms)
    entries = []
    for i, (label, stem) in enumerate(GALLERY):
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
        entries.append((label, i, len(consts)))

    # final slot: hand-authored real-hardware-loop proof (not from GLSL)
    li = len(GALLERY)
    lblob, lmanifest = build_loop_proof()
    parts.append(c_array(f"gal_ps{li}", lblob))
    rows = ",\n\t".join("{ %d, { %ff, %ff, %ff, %ff } }" % (reg, *vals)
                        for reg, vals in lmanifest)
    parts.append(f"static const gallery_const gal_c{li}[] = {{\n\t{rows}\n}};")
    parts.append("")
    entries.append(("LOOPHW", li, len(lmanifest)))

    parts.append("#define GALLERY_N %d" % len(entries))
    parts.append("static const gallery_entry gallery[GALLERY_N] = {")
    for label, i, nman in entries:
        cptr = f"gal_c{i}" if nman else "0"
        parts.append(f'\t{{ "{label}", gal_ps{i}, {nman}, {cptr} }},')
    parts.append("};")
    parts.append("")

    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}  ({len(GALLERY)} GLSL shaders + 1 hardware-loop proof)")
    for label, i, nman in entries:
        print(f"  gal_ps{i} {label}: {nman} literal const(s) to upload")


if __name__ == "__main__":
    main()
