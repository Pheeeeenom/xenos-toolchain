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

    # literal constant: mix(tex, color, 0.5) -> 0.5 in a pooled constant reg
    from xsc.lower import lower_ex
    bl50, manifest = lower_ex(spirv.Module.parse(
        (TD / "blend50.frag.spv").read_bytes()))
    b50hl = oracle_hlsl(bl50, "blend50_frag")
    want_man = [(0, (0.5, 0.5, 0.5, 0.5))]
    if b50hl is None or "_lit0.xyzw + r2.xyzw" not in b50hl or \
            manifest != want_man:
        ok = False
        print(f"  FAIL blend50.frag: literal pool wrong (manifest {manifest})")
    else:
        print(f"  PASS blend50.frag: literal 0.5 -> pooled const c0, "
              f"manifest {manifest} ({len(bl50)} B)")

    # scalar pipe: normalize + length (rsq/sqrt), and the full mathfns
    # stress shader (normalize+length+pow+floor) must compile + oracle-pass
    for name, needle, desc in [
        ("normlen", "rsqrt(r3.w)", "normalize (dp3+rsq) + length (dp3+sqrt)"),
        ("mathfns", "exp2(", "normalize+length+pow(log2/exp2)+floor stress"),
        ("explog", "sin(r1.y)", "exp/log natural + reflect(mad) + sin/cos"),
    ]:
        blob = compile_glsl(f"{name}.frag")
        h = oracle_hlsl(blob, f"{name}_frag")
        if h is None or needle not in h:
            ok = False
            print(f"  FAIL {name}.frag: {desc}")
        else:
            print(f"  PASS {name}.frag: {desc} ({len(blob)} B)")

    # control flow: if/else select -> branch-free cnd op (no jumps)
    cfif = compile_glsl("cf_if.frag")
    chl = oracle_hlsl(cfif, "cf_if_frag")
    if chl is None or "select(r3.xyzw > 0.0, r1.xyzw, r2.xyzw)" not in chl:
        ok = False
        print("  FAIL cf_if.frag: if/else cnd select wrong")
    else:
        print(f"  PASS cf_if.frag: if(a>b) A else B -> cndgt select, "
              f"branch-free ({len(cfif)} B)")

    # nested if/else -> chained cnd selects (inner select feeds the outer)
    nif = compile_glsl("nested_if.frag")
    nhl = oracle_hlsl(nif, "nested_if_frag")
    if nhl is None or "r4.xyzw = select(r5" not in nhl or \
            "oC0.xyzw = select(r3" not in nhl:
        ok = False
        print("  FAIL nested_if.frag: nested if/else chained-select wrong")
    else:
        print(f"  PASS nested_if.frag: nested if/else -> chained cnd selects, "
              f"branch-free ({len(nif)} B)")

    # non-identity swizzle (.bgra) + arbitrary clamp + floor/abs
    for name, needle, desc in [
        ("sw_swizzle", "max(r2.zyxw", "non-identity swizzle .bgra"),
        ("clamprange", "min(r4.xyzw, _lit1.xyzw)",
         "clamp(x,0.2,0.8) -> max/min + pooled literals"),
        ("floorabs", "floor(r1.xyzw)", "floor() + abs() as max(x,-x)"),
    ]:
        blob = compile_glsl(f"{name}.frag")
        h = oracle_hlsl(blob, f"{name}_frag")
        if h is None or needle not in h:
            ok = False
            print(f"  FAIL {name}.frag: {desc}")
        else:
            print(f"  PASS {name}.frag: {desc} ({len(blob)} B)")

    # phase-3b breadth: float division, sign(), tan()
    for name, needles, desc in [
        ("divide", ["rcp(u_denom.x)", "rcp(u_denom.w)"],
         "vec/vec division -> per-component rcp + mul"),
        ("absign", ["max(u_offset.xyzw, -u_offset.xyzw)", "> _lit1.xyzw"],
         "abs()=max(x,-x), sign()=sgt(x,0)-sgt(0,x)"),
        ("tangent", ["sin(u_phase.x)", "cos(u_phase.x)"],
         "tan(x)=sin(x)*rcp(cos(x))"),
    ]:
        blob = compile_glsl(f"{name}.frag")
        h = oracle_hlsl(blob, f"{name}_frag")
        miss = [s for s in needles if not h or s not in h]
        if h is None or miss:
            ok = False
            print(f"  FAIL {name}.frag: {desc} (missing {miss})")
        else:
            print(f"  PASS {name}.frag: {desc} ({len(blob)} B)")

    # phase-3b breadth II: rounding family, angles, step/smoothstep, mod, cross,
    # two-source vector assembly
    for name, needles, desc in [
        ("rounding", ["frac(", "trunc(", "floor(-"],
         "fract/trunc/ceil=-floor(-x)/round=floor(x+.5)"),
        ("angles", ["* _lit0.xyzw", "* _lit1.xyzw"],
         "radians/degrees -> mul by pooled const"),
        ("stepsmooth", [">= _lit", "saturate("],
         "step (kSge) + smoothstep (clamp + hermite t*t*(3-2t))"),
        ("modcross", [".zxyw *", "-u_period.xyzw *"],
         "cross (yzx/zxy mad) + mod (x - y*floor(x/y))"),
        ("shuffle2", ["oC0.xy = max(r2", "oC0.z = max(r3"],
         "two-source vec assembly (t.xy + color.zw)"),
        ("invtrig", ["sqrt(", "> _lit"],
         "asin (A-S sqrt form) / acos=pi/2-asin / atan (odd poly)"),
    ]:
        blob = compile_glsl(f"{name}.frag")
        h = oracle_hlsl(blob, f"{name}_frag")
        miss = [s for s in needles if not h or s not in h]
        if h is None or miss:
            ok = False
            print(f"  FAIL {name}.frag: {desc} (missing {miss})")
        else:
            print(f"  PASS {name}.frag: {desc} ({len(blob)} B)")

    # loops: constant-trip for/while unrolled to branch-free SSA (xsc.unroll),
    # then lowered + oracle-validated. Assert the loop was eliminated and the
    # body replicated the right number of times (the induction var is folded).
    from xsc.unroll import unroll as _unroll
    for name, want_adds, desc in [
        ("loop_accum", 4, "for i<4: s+=uv.x -> 4x unrolled accumulate"),
        ("loop_weight", 4, "for i<4: s+=uv.x*float(i) -> induction folded"),
        ("loop_while", 4, "while i<4 (increment in body) -> 4x unroll"),
        ("loop_nested", 9, "nested for 3x3 -> 9x (innermost-first)"),
        ("loop_stair", 5, "5x step() staircase + post-loop *0.2 (mem2reg RMW)"),
    ]:
        mu = spirv.Module.parse((TD / f"{name}.frag.spv").read_bytes())
        _unroll(mu)
        n_loop = sum(1 for i in mu.instrs if i.opcode == 246)  # OpLoopMerge
        n_add = sum(1 for i in mu.instrs if i.opcode == 129)   # OpFAdd
        blob = compile_glsl(f"{name}.frag")
        h = oracle_hlsl(blob, f"{name}_frag")
        if h is None or n_loop != 0 or n_add != want_adds:
            ok = False
            print(f"  FAIL {name}.frag: {desc} (loops={n_loop} adds={n_add} "
                  f"want {want_adds}, oracle={'ok' if h else 'REJECT'})")
        else:
            print(f"  PASS {name}.frag: {desc} ({len(blob)} B)")

    # dot(): luminance desaturation (vec3 dot -> dp3, broadcast scalar)
    gray = compile_glsl("gray.frag")
    ghl = oracle_hlsl(gray, "gray_frag")
    if ghl is None or "dot(r2.xyz, _lit0.xyz)" not in ghl or \
            "oC0.xyz = max(r3.xyz" not in ghl:
        ok = False
        print("  FAIL gray.frag: dot()/dp3 desaturation wrong")
    else:
        print(f"  PASS gray.frag: dot(rgb, luma) -> dp3 broadcast, "
              f"vec4(lum,lum,lum,a) ({len(gray)} B)")

    # clamp(x,0,1) -> free saturate modifier (no extra instruction)
    sat = compile_glsl("sat.frag")
    shl = oracle_hlsl(sat, "sat_frag")
    if shl is None or "saturate(r2.xyzw + r1.xyzw)" not in shl:
        ok = False
        print("  FAIL sat.frag: clamp->saturate wrong")
    else:
        print(f"  PASS sat.frag: clamp(x,0,1) -> add_sat fused, free "
              f"({len(sat)} B)")

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
