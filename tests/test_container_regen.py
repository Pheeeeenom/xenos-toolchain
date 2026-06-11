#!/usr/bin/env python3
"""Container-emitter exit test: rebuild both golden containers from semantic
parameters and demand byte-identical output.

Parameters below are the *meaning* of each blob (decoded in docs/container.md
and the Phase 0/1 analysis); if the emitter and these parameters reproduce
the XDK compiler's bytes exactly, the format knowledge is proven complete
enough to author new containers.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "xdis"))

import container_emit as ce  # noqa: E402

CORPUS = ROOT / "corpus"


def first_diff(a: bytes, b: bytes):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def regen_vs0() -> bytes:
    orig = (CORPUS / "vs0.bin").read_bytes()
    ucode = orig[0xD4:0xD4 + 0xB4]
    return ce.build_container(
        "vertex", ucode,
        constants=[ce.Constant("ortho", ce.RS_FLOAT4, 0, 4,
                               ce.CLASS_MATRIX_COLS, ce.TYPE_FLOAT,
                               rows=4, columns=4)],
        program_control=0x00210002,
        context_misc=0x00000000,
        interpolators=[
            ce.Interpolator(0, ce.USAGE_TEXCOORD, 0, flags=0x3),
            ce.Interpolator(1, ce.USAGE_TEXCOORD, 1, flags=0x13),
            ce.Interpolator(2, ce.USAGE_COLOR, 0, flags=0x2F),
        ],
        vertex_elements=[
            ce.VertexElement(3, ce.USAGE_POSITION, 0, flags=0x1),
            ce.VertexElement(4, ce.USAGE_TEXCOORD, 0, flags=0x0),
            ce.VertexElement(5, ce.USAGE_TEXCOORD, 1, flags=0x0),
            ce.VertexElement(6, ce.USAGE_COLOR, 0, flags=0x3),
        ],
        vs_tail_count=3,
        vs_mystery=0x290,
        export_slots=[0xB, 0xC, 0xD],
    )


def regen_ps_fb0() -> bytes:
    orig = (CORPUS / "ps_fb0.bin").read_bytes()
    ucode = orig[0xDC:0xDC + 0x54]
    return ce.build_container(
        "pixel", ucode,
        constants=[
            ce.Constant("tex0", ce.RS_SAMPLER, 0, 1,
                        ce.CLASS_OBJECT, 0xC,  # Sampler2D
                        rows=1, columns=1),
            ce.Constant("use_tex", ce.RS_BOOL, 0, 1,
                        ce.CLASS_SCALAR, ce.TYPE_BOOL,
                        rows=1, columns=1),
        ],
        program_control=0x10000200,
        context_misc=0x00000008,
        field10=0x00020000,
        interpolators=[
            ce.Interpolator(0, ce.USAGE_TEXCOORD, 0, flags=0x3),
            ce.Interpolator(1, ce.USAGE_TEXCOORD, 1, flags=0x3),
            ce.Interpolator(2, ce.USAGE_COLOR, 0, flags=0xF),
        ],
        ps_field18=0x00030007,
        ps_outputs=0x1,
    )


def main():
    ok = True
    for name, regen in (("vs0", regen_vs0), ("ps_fb0", regen_ps_fb0)):
        orig = (CORPUS / f"{name}.bin").read_bytes()
        new = regen()
        d = first_diff(orig, new)
        if d is None:
            print(f"PASS {name}: byte-identical ({len(new)} bytes)")
        else:
            ok = False
            lo = max(0, d - 8)
            print(f"FAIL {name}: first diff at {d:#x} "
                  f"(orig len {len(orig)}, new len {len(new)})")
            print(f"  orig[{lo:#x}:] {orig[lo:d+8].hex(' ')}")
            print(f"  new [{lo:#x}:] {new[lo:d+8].hex(' ')}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
