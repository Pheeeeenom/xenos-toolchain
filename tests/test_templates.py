#!/usr/bin/env python3
"""xsc's instruction templates (xsc/templates.py) against the corpus.

The templates are written from Xenia's definitions. This compares them with the
corpus instructions they once came from, read as test vectors, so a field that
drifts from what compiled shaders carry shows up. The known differences are the
fields the lowering always overrides, and the dp4 source swizzles: the corpus
writes .wzxy on both operands, which leaves a 4-component dot unchanged, and the
template writes the identity.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import xdis.xenos_ucode as xu  # noqa: E402
from xsc.templates import templates  # noqa: E402
from author_passthrough import alu_fields, vf_fields, corpus_records  # noqa: E402
from author_textured import tf_fields  # noqa: E402

# Fields the lowering sets on every use of the template (xsc/lower.py).
OVERRIDDEN = {
    "vf": {"dest"},
    "dp4": {"vector_opcode", "vector_dest", "is_export", "vector_write_mask",
            "src1_reg", "src1_is_temp", "src2_reg", "src2_is_temp"},
    "mov": {"vector_opcode", "vector_dest", "src1_reg", "src2_reg"},
    "tf": {"dest", "src", "fetch_constant_index", "dimension"},
}
INTENDED = {"dp4": {"src1_swizzle", "src2_swizzle"}}


def corpus_templates():
    vs0 = corpus_records("vs0")
    ps0 = corpus_records("ps_fb0")
    return {
        "vf": vf_fields(xu.VertexFetchInstruction(*vs0["vfetch3"].raw)),
        "dp4": alu_fields(xu.AluInstruction(*vs0["alu7"].raw)),
        "mov": alu_fields(xu.AluInstruction(*vs0["alu13"].raw)),
        "tf": tf_fields(xu.TextureFetchInstruction(*ps0["tfetch3"].raw)),
    }


def main():
    ok = True
    ours = templates()
    theirs = corpus_templates()
    for name, fields in theirs.items():
        if set(ours[name]) != set(fields):
            ok = False
            print(f"  FAIL {name}: field sets differ: "
                  f"{sorted(set(ours[name]) ^ set(fields))}")
            continue
        allowed = OVERRIDDEN[name] | INTENDED.get(name, set())
        drift = [f for f in fields if ours[name][f] != fields[f]
                 and f not in allowed]
        if drift:
            ok = False
            print(f"  FAIL {name}: differs from the corpus in "
                  + ", ".join(f"{f} ({ours[name][f]} vs {fields[f]})"
                              for f in drift))
        else:
            print(f"  PASS {name}: matches the corpus but for the fields "
                  f"the lowering sets{' and the intended swizzles' if name in INTENDED else ''}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
