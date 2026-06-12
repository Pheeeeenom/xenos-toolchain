#!/usr/bin/env python3
"""Exit gate for real hardware loops (LOOP_START/LOOP_END) emitted by
xasm.program. Builds a pixel shader with a hardware loop and checks:
  - the control-flow program has LOOP_START/LOOP_END with correct jump
    addresses (skip past LOOP_END; body = the exec after LOOP_START), and
  - XenosRecomp decompiles it to `for (aL = 0; aL < i0.x; aL++)` (the trip
    count comes from loop constant i0, uploaded by the host at draw time).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import xdis.container as container          # noqa: E402
import xdis.xenos_ucode as xu               # noqa: E402
import xdis.container_emit as ce            # noqa: E402
from xsc.lower import _templates            # noqa: E402
from xasm.program import PixelProgram       # noqa: E402
from tests.test_xsc_lower import oracle_hlsl  # noqa: E402

_V = xu.AluVectorOpcode


def build_loop_shader():
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
    p.input(0, 5)                                                 # r0 = v_texcoord (match VS)
    p.input(1, 10)                                                # r1 = v_color
    # accumulate in r2 — r0/r1 hold the VS-exported interpolators
    p.alu(**alu(_V.kMax, 2, [(1, False), (1, False)]))            # r2 = c1 (0)
    p.begin_loop(loop_id=0)
    p.alu(**alu(_V.kAdd, 2, [(2, True), (0, False)]))             # r2 += c0
    p.end_loop()
    p.alloc_colors()                                              # ALLOC pixel export
    p.alu(**alu(_V.kMax, 0, [(2, True), (2, True)], export=True))  # oC0 = r2
    consts = [ce.Constant(n, ce.RS_FLOAT4, r, 1, ce.CLASS_VECTOR, ce.TYPE_FLOAT,
                          rows=1, columns=4) for n, r in (("c0", 0), ("c1", 1))]
    return p.finish(constants=consts)


def cf_ops(blob):
    """(opcode, loop_id, address) per control-flow instruction, in order."""
    dwords = container.Container(blob).ucode_dwords
    out = []
    for i in range(len(dwords) // 3):
        for cf in xu.UnpackControlFlowInstructions(dwords[i * 3:i * 3 + 3]):
            op = cf.opcode()
            if op == xu.ControlFlowOpcode.kLoopStart:
                out.append(("loop_start", cf.loop_start.loop_id(),
                            cf.loop_start.address()))
            elif op == xu.ControlFlowOpcode.kLoopEnd:
                out.append(("loop_end", cf.loop_end.loop_id(),
                            cf.loop_end.address()))
            elif op == xu.ControlFlowOpcode.kAlloc:
                out.append(("alloc", None, None))
            elif xu.IsControlFlowOpcodeExec(op):
                out.append(("exec", None, cf.exec.address()))
    return out


def main():
    ok = True
    blob = build_loop_shader()
    ops = cf_ops(blob)

    # CF must be: exec, loop_start(skip=Le+1), exec(body), loop_end(body=Ls+1), exec
    kinds = [o[0] for o in ops]
    try:
        ls = kinds.index("loop_start")
        le = kinds.index("loop_end")
    except ValueError:
        ls = le = -1
    structure_ok = (ls >= 0 and le == ls + 2 and kinds[ls + 1] == "exec"
                    and ops[ls][2] == le + 1        # skip past LOOP_END
                    and ops[le][2] == ls + 1)       # body = exec after LOOP_START
    if structure_ok:
        print(f"  PASS program loop CF: LOOP_START i{ops[ls][1]} skip->L{ops[ls][2]}"
              f", LOOP_END body->L{ops[le][2]} ({len(blob)} B)")
    else:
        ok = False
        print(f"  FAIL program loop CF structure: {ops}")

    hlsl = oracle_hlsl(blob, "program_loop")
    loop_ok = hlsl is not None and "for (aL = 0; aL < i0.x; aL++)" in hlsl
    if loop_ok:
        print("  PASS oracle: decompiles to for(aL=0; aL<i0.x; aL++) r0+=c0")
    else:
        ok = False
        print(f"  FAIL oracle loop: {'rejected' if hlsl is None else 'no for(aL) loop'}")

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
