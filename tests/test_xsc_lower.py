#!/usr/bin/env python3
"""Exit gate for the xsc GLSL->Xenos pipeline (Phase 3a).

Compiles the two canonical test shaders from SPIR-V through xsc.lower and
checks them against the hardware-proven hand-built equivalents:
  - modulate.frag must be BYTE-IDENTICAL to the modulate combiner
    (comb_0_modulate.bin) — already photographed on console.
  - spincube.vert must be oracle-equivalent to xost_vs2 (XenosRecomp accepts
    it and emits the mvp dp4 transform + passthrough interpolators).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import xsc.spirv as spirv  # noqa: E402
from xsc.lower import lower  # noqa: E402

TD = ROOT / "xsc" / "testdata"
RECOMP = (ROOT / "reference" / "XenosRecomp" / "build" / "XenosRecomp" /
          "XenosRecomp.exe")
SHADER_COMMON = (ROOT / "reference" / "XenosRecomp" / "XenosRecomp" /
                 "shader_common.h")


def compile_glsl(name):
    return lower(spirv.Module.parse((TD / f"{name}.spv").read_bytes()))


def oracle_hlsl(blob, tag):
    binp = TD / f"_{tag}.bin"
    hlsl = TD / f"_{tag}.hlsl"
    binp.write_bytes(blob)
    r = subprocess.run([str(RECOMP), str(binp), str(hlsl),
                        str(SHADER_COMMON)], capture_output=True)
    if r.returncode != 0:
        return None
    return hlsl.read_text()


def main():
    ok = True

    # fragment: byte-identical to the photographed combiner
    frag = compile_glsl("modulate.frag")
    gold = (ROOT / "examples" / "out" / "comb_0_modulate.bin").read_bytes()
    if frag == gold:
        print(f"  PASS modulate.frag byte-identical to combiner ({len(frag)} B)")
    else:
        ok = False
        print(f"  FAIL modulate.frag: {len(frag)} B vs combiner {len(gold)} B")

    # novel shader (never hand-built): mix() must decompose to the lerp
    # identity x + (y-x)*a and oracle-validate. The proof xsc *compiles*.
    blend = compile_glsl("blend.frag")
    bhlsl = oracle_hlsl(blend, "blend_frag")
    if bhlsl is None:
        ok = False
        print("  FAIL blend.frag: XenosRecomp rejected it")
    else:
        need = ["r1.xyzw + -r2.xyzw", "* u_blend.xyzw + r2.xyzw"]
        miss = [s for s in need if s not in bhlsl]
        if miss:
            ok = False
            print(f"  FAIL blend.frag mix() decomposition wrong: missing "
                  f"{miss}")
        else:
            print(f"  PASS blend.frag: novel mix() -> correct lerp "
                  f"(sub+mad), oracle-validated ({len(blend)} B)")

    # swizzle / CompositeConstruct: vec4(tex.rgb * color.rgb, tex.a)
    tint = compile_glsl("tint.frag")
    thlsl = oracle_hlsl(tint, "tint_frag")
    if thlsl is None:
        ok = False
        print("  FAIL tint.frag: XenosRecomp rejected it")
    else:
        need = ["oC0.xyz = max(r3.xyz", "r2.xyzw * r1.xyzw",
                "oC0.w = max(r2.w"]
        miss = [s for s in need if s not in thlsl]
        if miss:
            ok = False
            print(f"  FAIL tint.frag swizzle/construct wrong: missing {miss}")
        else:
            print(f"  PASS tint.frag: .rgb swizzle + vec4(rgb,a) construct "
                  f"-> masked exports ({len(tint)} B)")

    # VectorTimesScalar: modulate by a scalar uniform (.xxxx broadcast)
    bright = compile_glsl("bright.frag")
    bhl = oracle_hlsl(bright, "bright_frag")
    if bhl is None or "u_brightness.xxxx" not in bhl:
        ok = False
        print("  FAIL bright.frag: VectorTimesScalar broadcast wrong")
    else:
        print(f"  PASS bright.frag: vec * scalar uniform (.xxxx) "
              f"({len(bright)} B)")

    # vertex: oracle-equivalent to the mvp transform
    vert = compile_glsl("spincube.vert")
    hlsl = oracle_hlsl(vert, "spincube_vert")
    if hlsl is None:
        ok = False
        print("  FAIL spincube.vert: XenosRecomp rejected it")
    else:
        need = ["dot(r0.wzxy, u_mvp(0)", "dot(r0.wzxy, u_mvp(3)",
                "oTexCoord0", "oColor0"]
        miss = [s for s in need if s not in hlsl]
        if miss:
            ok = False
            print(f"  FAIL spincube.vert HLSL missing {miss}")
        else:
            print(f"  PASS spincube.vert oracle = mvp dp4 transform + "
                  f"passthrough ({len(vert)} B)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
