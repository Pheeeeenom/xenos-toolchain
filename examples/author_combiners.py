#!/usr/bin/env python3
"""Generates the combiner demo bank: six classic N64 combiner modes
compiled to Xenos pixel shaders by xasm.combiner.Generator.

Outputs examples/out/comb_<i>_<name>.bin and prints each disassembly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import xdis.container as container  # noqa: E402
import xdis.xenos_ucode as xu  # noqa: E402
from xasm.combiner import CombinerCycle, Generator, Src  # noqa: E402
from author_passthrough import alu_fields, corpus_records  # noqa: E402
from author_textured import tf_fields  # noqa: E402

OUT = ROOT / "examples" / "out"
Z, O, T, S, P, E = (Src.ZERO, Src.ONE, Src.TEXEL0, Src.SHADE,
                    Src.PRIM, Src.ENV)
T1, C = Src.TEXEL1, Src.COMBINED

# (name, cycle0[, cycle1]): out = (A-B)*C+D per cycle;
# cycle1 sees cycle0's result as COMBINED
BANK = [
    ("modulate",  CombinerCycle(T, Z, S, Z)),                      # tex*shade
    ("tex_prim",  CombinerCycle(T, Z, P, Z, aa=Z, ab=Z, ac=Z, ad=O)),
    ("lerp_pe_t", CombinerCycle(P, E, T, E, aa=Z, ab=Z, ac=Z, ad=O)),
    ("decal",     CombinerCycle(T, Z, O, Z)),                      # tex only
    ("shade",     CombinerCycle(S, Z, O, Z, aa=Z, ab=Z, ac=Z, ad=O)),
    ("inv_tex_p", CombinerCycle(O, T, P, Z, aa=Z, ab=Z, ac=Z, ad=O)),
    # --- 2-cycle modes ---
    ("multitex",  CombinerCycle(T, Z, T1, Z),                  # t0*t1
                  CombinerCycle(C, Z, S, Z)),                  # *shade
    ("lerp2_p",   CombinerCycle(T, T1, S, T1),                 # lerp(t1,t0,sh)
                  CombinerCycle(C, Z, P, Z, aa=Z, ab=Z, ac=Z, ad=O)),
    ("add_glow",  CombinerCycle(T, Z, S, Z),                   # t0*shade
                  CombinerCycle(T1, Z, O, C, aa=Z, ab=Z, ac=Z, ad=O)),
]


def main():
    vs0 = corpus_records("vs0")
    ps0 = corpus_records("ps_fb0")
    gen = Generator(
        alu_template=alu_fields(xu.AluInstruction(*vs0["alu13"].raw)),
        tfetch_template=tf_fields(
            xu.TextureFetchInstruction(*ps0["tfetch3"].raw)))

    OUT.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(BANK):
        name, cycles = entry[0], entry[1:]
        blob = gen.generate(*cycles)
        path = OUT / f"comb_{i}_{name}.bin"
        path.write_bytes(blob)
        c = container.Container(blob)
        kind = "2-cycle" if len(cycles) == 2 else "1-cycle"
        print(f"=== {i}: {name} ({kind}, {len(blob)} B, "
              f"{c.sh_size // 12} slots) ===")
        print(xu.disassemble(c.ucode_dwords, "pixel"))


if __name__ == "__main__":
    main()
