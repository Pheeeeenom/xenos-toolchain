#!/usr/bin/env python3
"""xsc lowers Vulkan-flavoured SPIR-V (glslang -V), as vulkan4xenon hands it.

vk_triangle.{vert,frag} are compiled both ways in xsc/testdata: .spv with -V and
.gl.spv with -G --target-env opengl. The containers must be identical, and the
oracle must accept them. Descriptor sets and push constants come later.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import xsc.spirv as spirv  # noqa: E402
from xsc.lower import lower  # noqa: E402
from test_xsc_lower import TD, oracle_hlsl  # noqa: E402


def lowered(path):
    return lower(spirv.Module.parse(path.read_bytes()))


def main():
    ok = True
    for stage in ("vert", "frag"):
        vulkan = lowered(TD / f"vk_triangle.{stage}.spv")
        opengl = lowered(TD / f"vk_triangle.{stage}.gl.spv")
        if vulkan != opengl:
            ok = False
            print(f"  FAIL vk_triangle.{stage}: Vulkan and OpenGL SPIR-V "
                  f"lower differently")
        elif oracle_hlsl(vulkan, f"vk_triangle_{stage}") is None:
            ok = False
            print(f"  FAIL vk_triangle.{stage}: XenosRecomp rejected it")
        else:
            print(f"  PASS vk_triangle.{stage}: Vulkan SPIR-V lowers as the "
                  f"OpenGL one ({len(vulkan)} B), oracle OK")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
