#!/usr/bin/env python3
"""Exact-mode totality test: for every corpus blob,
disassemble(exact=True) -> reconstruct_from_exact -> bytes identical to the
original ucode region. Also asserts default output is unchanged (golden)."""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "xdis"))

import container  # noqa: E402
import xenos_ucode  # noqa: E402


def main():
    ok = True
    for name in ("vs0", "ps_fb0", "cube_vs", "cube_ps"):
        blob = (ROOT / "corpus" / f"{name}.bin").read_bytes()
        c = container.Container(blob)
        exact_text = xenos_ucode.disassemble(c.ucode_dwords, c.shader_type,
                                             exact=True)
        rebuilt = xenos_ucode.reconstruct_from_exact(exact_text)
        if rebuilt == c.ucode_bytes:
            print(f"PASS {name}: exact text -> {len(rebuilt)} bytes, "
                  f"identical")
        else:
            ok = False
            print(f"FAIL {name}: rebuilt {len(rebuilt)} bytes vs "
                  f"{len(c.ucode_bytes)} original")
        # default output must remain pinned to the golden snapshot
        pretty = xenos_ucode.disassemble(c.ucode_dwords, c.shader_type)
        golden = (ROOT / "tests" / "golden" / f"{name}.pretty.txt"
                  ).read_text()
        # golden files lack the banner; compare disassembly body only
        if pretty.strip() == "\n".join(
                l for l in golden.splitlines() if not l.startswith("===")
                ).strip():
            print(f"PASS {name}: default output unchanged")
        else:
            ok = False
            print(f"FAIL {name}: default output drifted from golden")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
