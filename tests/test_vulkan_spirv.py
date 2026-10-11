#!/usr/bin/env python3
"""xsc lowers Vulkan-flavoured SPIR-V (glslang -V), as vulkan4xenon hands it.

- vk_triangle.{vert,frag} are compiled both ways in xsc/testdata: .spv with -V
  and .gl.spv with -G --target-env opengl. The containers must be identical, and
  the oracle must accept them.
- vk_blocks.vert and vk_matrices.vert read uniform and push constant blocks.
  Their registers must be where resolve.py's convention puts them, and running
  the program on the blocks' raw std140 bytes must give the right position and
  colour. The oracle can't see values, so a small simulator of the ALU checks
  them, as tests/test_matrix.py does for fragment shaders.
- vk_samplers.frag must take fetch constants in (set, binding) order.
"""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import xsc.spirv as spirv  # noqa: E402
import xdis.container as container  # noqa: E402
import xdis.xenos_ucode as xu  # noqa: E402
from xsc.lower import lower, lower_ex  # noqa: E402
from xsc.resolve import resolve_interface  # noqa: E402
from test_xsc_lower import TD, oracle_hlsl  # noqa: E402

V = xu.AluVectorOpcode
OPOS = int(xu.ExportRegister.kVSPosition)


def module(name):
    return spirv.Module.parse((TD / name).read_bytes())


def lowered(path):
    return lower(spirv.Module.parse(path.read_bytes()))


def _src(op, n, regs, consts):
    reg = op.src_reg(n)
    val = list(regs.get(reg & 0x3F, [0.0] * 4)) if op.src_is_temp(n) else list(consts.get(reg, [0.0] * 4))
    val = [val[xu.AluInstruction.GetSwizzledComponentIndex(op.src_swizzle(n), i)] for i in range(4)]
    return [-x for x in val] if op.src_negate(n) else val


def run_vertex(blob, inputs, consts):
    """Runs a vertex program's vector ALU; returns its exports by register (62 is the position)."""
    regs = {k: list(v) for k, v in inputs.items()}
    exports = {}
    c = container.Container(blob)
    for rec in xu._decode_program(c.ucode_dwords, c.shader_type)[0]:
        if rec.kind != "alu":
            continue
        op = xu.AluInstruction(*rec.raw)
        mask = op.GetVectorOpResultWriteMask()
        if not mask:
            continue
        s1, s2 = _src(op, 1, regs, consts), _src(op, 2, regs, consts)
        vo = op.vector_opcode()
        if vo == V.kMul:
            r = [s1[i] * s2[i] for i in range(4)]
        elif vo == V.kAdd:
            r = [s1[i] + s2[i] for i in range(4)]
        elif vo == V.kMax:
            r = [max(s1[i], s2[i]) for i in range(4)]
        elif vo == V.kMad:
            s3 = _src(op, 3, regs, consts)
            r = [s1[i] * s2[i] + s3[i] for i in range(4)]
        elif vo == V.kDp4:
            r = [sum(s1[i] * s2[i] for i in range(4))] * 4
        else:
            raise SystemExit(f"test_vulkan_spirv: unhandled vector op {vo}")
        dst = exports if op.is_export() else regs
        out = dst.setdefault(op.vector_dest(), [0.0] * 4)
        for i in range(4):
            if mask & (1 << i):
                out[i] = r[i]
    return exports


def registers(raw, first):
    """A block's raw bytes as the float constant registers the layer copies them into."""
    floats = struct.unpack(f"<{len(raw) // 4}f", raw)
    return {first + i: list(floats[4 * i:4 * i + 4]) for i in range(len(floats) // 4)}


def columns(m):
    """A row-indexed 4x4 matrix as std140's column-major bytes."""
    return struct.pack("<16f", *(m[r][c] for c in range(4) for r in range(4)))


def mat_vec(m, v):
    return [sum(m[r][c] * v[c] for c in range(4)) for r in range(4)]


def vec_mat(v, m):
    return [sum(v[r] * m[r][c] for r in range(4)) for c in range(4)]


def mat_mat(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


def close(a, b):
    return all(abs(x - y) <= 1e-4 for x, y in zip(a, b))


MVP = [[1.0, 2.0, 0.0, 3.0], [0.0, 1.0, -1.0, 0.5], [2.0, 0.0, 1.0, -1.0], [0.0, 0.5, 0.0, 1.0]]
VIEW = [[0.0, -1.0, 0.0, 2.0], [1.0, 0.0, 0.0, -3.0], [0.0, 0.0, 1.0, 0.25], [0.0, 0.0, 0.0, 1.0]]
PROJECTION = [[1.5, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.2], [0.0, 0.0, -1.0, 0.0]]
POSITION = [0.5, -2.0, 3.0, 1.0]


def check_triangle():
    ok = True
    for stage in ("vert", "frag"):
        vulkan = lowered(TD / f"vk_triangle.{stage}.spv")
        opengl = lowered(TD / f"vk_triangle.{stage}.gl.spv")
        if vulkan != opengl:
            ok = False
            print(f"  FAIL vk_triangle.{stage}: Vulkan and OpenGL SPIR-V lower differently")
        elif oracle_hlsl(vulkan, f"vk_triangle_{stage}") is None:
            ok = False
            print(f"  FAIL vk_triangle.{stage}: XenosRecomp rejected it")
        else:
            print(f"  PASS vk_triangle.{stage}: Vulkan SPIR-V lowers as the OpenGL one ({len(vulkan)} B), oracle OK")
    return ok


def check_blocks():
    m = module("vk_blocks.vert.spv")
    si = resolve_interface(m)
    layout = [(b.kind, b.descriptor_set, b.binding, b.reg, b.regs) for b in si.blocks]
    want = [("push", None, None, 0, 1), ("uniform", 0, 0, 1, 5), ("uniform", 0, 1, 6, 1)]
    if layout != want:
        print(f"  FAIL vk_blocks.vert: blocks at {layout}, want {want}")
        return False
    blob, manifest = lower_ex(m)
    color, tint, scale, offset = [0.5, 0.25, 1.0, 2.0], [2.0, 4.0, 0.5, 1.0], 3.0, [0.125, -0.5]
    consts = registers(struct.pack("<4f", *color), 0)
    consts.update(registers(columns(MVP) + struct.pack("<4f", *tint), 1))
    consts.update(registers(struct.pack("<f4x2f", scale, *offset), 6))
    for reg, values in manifest:
        consts[reg] = list(values)
    exports = run_vertex(blob, {0: POSITION}, consts)
    want_position = mat_vec(MVP, POSITION)
    want_color = [color[i] * tint[i] * scale + (offset + [0.0, 0.0])[i] for i in range(4)]
    if not close(exports.get(OPOS, []), want_position) or not close(exports.get(0, []), want_color):
        print(f"  FAIL vk_blocks.vert: position {exports.get(OPOS)} (want {want_position}), "
              f"colour {exports.get(0)} (want {want_color})")
        return False
    if oracle_hlsl(blob, "vk_blocks_vert") is None:
        print("  FAIL vk_blocks.vert: XenosRecomp rejected it")
        return False
    print("  PASS vk_blocks.vert: push c0, blocks c1-c5 and c6; mvp * position and the colour from raw std140 bytes")
    return True


def check_matrices():
    m = module("vk_matrices.vert.spv")
    blob, manifest = lower_ex(m)
    consts = registers(columns(VIEW) + columns(PROJECTION), 0)
    for reg, values in manifest:
        consts[reg] = list(values)
    exports = run_vertex(blob, {0: POSITION}, consts)
    want_position = mat_vec(mat_mat(PROJECTION, VIEW), POSITION)
    want_row = vec_mat(POSITION, VIEW)
    if not close(exports.get(OPOS, []), want_position) or not close(exports.get(0, []), want_row):
        print(f"  FAIL vk_matrices.vert: position {exports.get(OPOS)} (want {want_position}), "
              f"row {exports.get(0)} (want {want_row})")
        return False
    if oracle_hlsl(blob, "vk_matrices_vert") is None:
        print("  FAIL vk_matrices.vert: XenosRecomp rejected it")
        return False
    print("  PASS vk_matrices.vert: projection * view * position and position * view over columns")
    return True


def check_samplers():
    m = module("vk_samplers.frag.spv")
    si = resolve_interface(m)
    slots = sorted((v.descriptor_set, v.binding, slot) for v, slot in si.samplers)
    blocks = [(b.kind, b.reg, b.regs) for b in si.blocks]
    if slots != [(0, 2, 0), (1, 0, 1)] or blocks != [("push", 0, 2)]:
        print(f"  FAIL vk_samplers.frag: samplers {slots}, blocks {blocks}")
        return False
    blob = lower(m)
    c = container.Container(blob)
    fetched = sorted(xu.TextureFetchInstruction(*r.raw).fetch_constant_index()
                     for r in xu._decode_program(c.ucode_dwords, c.shader_type)[0] if r.kind == "tfetch")
    if fetched != [0, 1] or oracle_hlsl(blob, "vk_samplers_frag") is None:
        print(f"  FAIL vk_samplers.frag: fetches from {fetched}, or the oracle rejected it")
        return False
    print("  PASS vk_samplers.frag: set 0 binding 2 -> tf0, set 1 binding 0 -> tf1, push offset 16 -> c1")
    return True


def main():
    results = [check_triangle(), check_blocks(), check_matrices(), check_samplers()]
    ok = all(results)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
