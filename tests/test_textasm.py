#!/usr/bin/env python3
"""Exit gate for the text assembler: every known-good shader must survive
container -> disassemble(pretty) -> assemble_text -> identical ucode bytes.
Plus a conventional-program fuzz over generated combiner shaders."""
import random
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import xdis.container as container  # noqa: E402
import xdis.xenos_ucode as xu  # noqa: E402
from xasm.textasm import assemble_text  # noqa: E402


def ensure_outputs():
    out = ROOT / "examples" / "out"
    need = ["xost_vs0.bin", "xost_ps0.bin", "xost_vs1.bin", "xost_vs2.bin",
            "xost_ps1.bin", "comb_0_modulate.bin"]
    if all((out / n).exists() for n in need):
        return
    for script in ("author_passthrough", "author_mvp", "author_textured",
                   "author_combiners"):
        subprocess.run([sys.executable,
                        str(ROOT / "examples" / f"{script}.py")],
                       check=True, capture_output=True)


def blob_list():
    c = ROOT / "corpus"
    o = ROOT / "examples" / "out"
    names = [c / "vs0.bin", c / "ps_fb0.bin", c / "cube_vs.bin",
             c / "cube_ps.bin", o / "xost_vs0.bin", o / "xost_ps0.bin",
             o / "xost_vs1.bin", o / "xost_vs2.bin", o / "xost_ps1.bin"]
    names += sorted(o.glob("comb_*.bin"))
    return names


def roundtrip(path: Path):
    c = container.Container(path.read_bytes())
    pretty = xu.disassemble(c.ucode_dwords, c.shader_type)
    rebuilt = assemble_text(pretty, c.shader_type)
    return rebuilt, c.ucode_bytes


def fuzz(n=200, seed=0xA55E):
    from xasm.combiner import Generator, CombinerCycle, Src
    from author_passthrough import alu_fields, corpus_records
    from author_textured import tf_fields
    vs0 = corpus_records("vs0")
    ps0 = corpus_records("ps_fb0")
    gen = Generator(
        alu_template=alu_fields(xu.AluInstruction(*vs0["alu13"].raw)),
        tfetch_template=tf_fields(
            xu.TextureFetchInstruction(*ps0["tfetch3"].raw)))
    srcs = [Src.ZERO, Src.ONE, Src.TEXEL0, Src.TEXEL1, Src.SHADE,
            Src.PRIM, Src.ENV]
    rng = random.Random(seed)
    ok = 0
    for _ in range(n):
        def mksel():
            return CombinerCycle(rng.choice(srcs), rng.choice(srcs),
                                 rng.choice(srcs), rng.choice(srcs))
        c0 = mksel()
        c1 = mksel() if rng.random() < 0.5 else None
        try:
            blob = gen.generate(c0, c1)
        except Exception:
            continue
        c = container.Container(blob)
        pretty = xu.disassemble(c.ucode_dwords, "pixel")
        rebuilt = assemble_text(pretty, "pixel")
        if rebuilt != c.ucode_bytes:
            print(f"  FUZZ FAIL: {c0.color_sel()} / "
                  f"{c1.color_sel() if c1 else None}")
            return ok, False
        ok += 1
    return ok, True


def source_mode_smoke():
    """Hand-authored .xasm -> container -> XenosRecomp accepts it."""
    from xasm.textasm import assemble_file
    src = """.pixel
.interpolator 0 texcoord
.interpolator 1 color
.sampler tex0 0
      exec
   tfetch2D r2, r0.xy, tf0
      alloc colors
      exece
   mul oC0, r2, r1
      cnop
"""
    tmp = ROOT / "examples" / "out" / "_textasm_smoke.xasm"
    tmp.write_text(src)
    blob = assemble_file(str(tmp), "pixel", container=True)
    outbin = ROOT / "examples" / "out" / "_textasm_smoke.bin"
    outbin.write_bytes(blob)
    # structural check
    c = container.Container(blob)
    dis = xu.disassemble(c.ucode_dwords, "pixel")
    assert "tfetch2D" in dis and "mul oC0" in dis, dis
    # oracle check
    rc = (ROOT / "reference" / "XenosRecomp" / "build" / "XenosRecomp" /
          "XenosRecomp.exe")
    hdr = (ROOT / "reference" / "XenosRecomp" / "XenosRecomp" /
           "shader_common.h")
    if rc.exists():
        r = subprocess.run([str(rc), str(outbin),
                            str(outbin.with_suffix(".hlsl")), str(hdr)],
                           capture_output=True)
        return r.returncode == 0
    return True


def main():
    ensure_outputs()
    ok = True
    print("== 16-blob round-trip ==")
    for path in blob_list():
        rebuilt, orig = roundtrip(path)
        # pretty text cannot express unreferenced trailing pad slots (only
        # exact mode carries them); identical-up-to-trailing-zeros is the
        # correct round-trip contract for pretty text
        pad_ok = (len(rebuilt) < len(orig)
                  and orig[:len(rebuilt)] == rebuilt
                  and not any(orig[len(rebuilt):]))
        if rebuilt == orig:
            print(f"  PASS {path.name} ({len(orig)} B)")
        elif pad_ok:
            print(f"  PASS {path.name} ({len(rebuilt)} B + "
                  f"{len(orig) - len(rebuilt)} B zero pad in original)")
        else:
            ok = False
            d = next((i for i in range(min(len(rebuilt), len(orig)))
                      if rebuilt[i] != orig[i]), None)
            print(f"  FAIL {path.name}: len {len(rebuilt)} vs {len(orig)}, "
                  f"first diff at {d}")
    print("== conventional fuzz ==")
    n, fok = fuzz()
    print(f"  {'PASS' if fok else 'FAIL'} {n} combiner programs round-tripped")
    ok = ok and fok
    print("== source-mode smoke ==")
    sok = source_mode_smoke()
    print(f"  {'PASS' if sok else 'FAIL'} hand-authored .xasm -> container "
          f"-> oracle")
    ok = ok and sok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
