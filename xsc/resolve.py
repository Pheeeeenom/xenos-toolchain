#!/usr/bin/env python3
"""SPIR-V type + interface resolution for xsc.

Turns a parsed `spirv.Module` into:
  - a type table (id -> ResolvedType, with vec/mat shapes), and
  - a ShaderInterface: each interface variable classified into the Xenos
    role its loads/stores lower to (vfetch / interpolator / constant /
    sampler / position / color export), plus the metadata the container
    emitter needs (vertex elements, interpolators, constants, samplers).

Conventions for the GLSL<->Xenos mapping (documented; v1):
  - VS inputs: assigned vfetch in Location order; Xenos vertex-element usage
    by location: 0->POSITION, 1->COLOR, 2->TEXCOORD, 3+->TEXCOORD(idx++).
    The host vertex format must match this order (same as our hand shaders).
  - VS outputs: BuiltIn Position -> oPos; Location N -> interpolator oN
    (usage TEXCOORD for vec2, COLOR otherwise — a heuristic; overridable).
  - FS inputs: Location N -> interpolator register rN (must match the VS).
  - Uniform block members: float/vecN -> 1 constant register; matN -> N
    constant registers; register index = Offset/16.
  - Samplers (UniformConstant sampled image): tfetch fetch-constant = Binding.
  - FS output Location 0 -> oC0.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import xsc.spirv as spirv  # noqa: E402

# SPIR-V StorageClass
SC_UNIFORM_CONSTANT = 0
SC_INPUT = 1
SC_UNIFORM = 2
SC_OUTPUT = 3
SC_FUNCTION = 7
SC_PUSH_CONSTANT = 9

# SPIR-V Decoration
DEC_BLOCK = 2
DEC_COL_MAJOR = 5
DEC_MATRIX_STRIDE = 7
DEC_BUILTIN = 11
DEC_LOCATION = 30
DEC_BINDING = 33
DEC_DESCRIPTOR_SET = 34
DEC_OFFSET = 35

# SPIR-V BuiltIn
BUILTIN_POSITION = 0
BUILTIN_POINT_SIZE = 1
# input builtins we ignore as real vertex inputs
BUILTIN_VERTEX_ID = 5
BUILTIN_INSTANCE_ID = 6
_IGNORED_INPUT_BUILTINS = {BUILTIN_VERTEX_ID, BUILTIN_INSTANCE_ID}


@dataclass
class RType:
    kind: str                 # void/bool/int/float/vector/matrix/pointer/
                              # struct/image/sampledimage/array/function/void
    comp: int | None = None   # element type id (vector/matrix/pointer/array)
    count: int | None = None  # vector size / matrix columns / array length
    members: tuple = ()        # struct member type ids
    storage_class: int | None = None  # pointer

    def __repr__(self):
        if self.kind == "vector":
            return f"vec{self.count}"
        if self.kind == "matrix":
            return f"mat{self.count}"
        if self.kind == "float":
            return "float"
        if self.kind == "pointer":
            return f"ptr({self.storage_class})"
        return self.kind


def resolve_types(m: spirv.Module) -> dict:
    t: dict[int, RType] = {}
    for ins in m.instrs:
        op, rid, ops = ins.opcode, ins.result_id, ins.operands
        if op == 19:
            t[rid] = RType("void")
        elif op == 20:
            t[rid] = RType("bool")
        elif op == 21:
            t[rid] = RType("int")
        elif op == 22:
            t[rid] = RType("float")
        elif op == 23:  # TypeVector comp count
            t[rid] = RType("vector", comp=ops[0], count=ops[1])
        elif op == 24:  # TypeMatrix colType colCount
            t[rid] = RType("matrix", comp=ops[0], count=ops[1])
        elif op == 25:
            t[rid] = RType("image")
        elif op == 26:
            t[rid] = RType("sampler")
        elif op == 27:  # TypeSampledImage imageType
            t[rid] = RType("sampledimage", comp=ops[0])
        elif op == 28:  # TypeArray elemType lengthConstId
            t[rid] = RType("array", comp=ops[0], count=ops[1])
        elif op == 30:  # TypeStruct members...
            t[rid] = RType("struct", members=tuple(ops))
        elif op == 32:  # TypePointer storageClass pointee
            t[rid] = RType("pointer", storage_class=ops[0], comp=ops[1])
        elif op == 33:
            t[rid] = RType("function")
    return t


def _decos(m: spirv.Module, vid: int) -> dict:
    """Flatten a variable's decorations into {name: args}."""
    out = {}
    for deco, args in m.decorations.get(vid, []):
        out[deco] = args
    return out


@dataclass
class Var:
    vid: int
    name: str
    storage_class: int
    pointee: int             # resolved type id behind the pointer
    location: int | None = None
    binding: int | None = None
    builtin: int | None = None


@dataclass
class ConstantSlot:
    name: str
    reg: int                 # base constant register
    regs: int                # how many (mat4 = 4)
    member: int


@dataclass
class ShaderInterface:
    stage: str
    vfetch: list = field(default_factory=list)        # (Var, usage, uidx)
    interp_out: list = field(default_factory=list)    # (Var, o_index, usage)
    interp_in: list = field(default_factory=list)     # (Var, r_reg, usage)
    position_out: int | None = None                   # gl_Position var id
    color_out: list = field(default_factory=list)     # (Var, oC index)
    constants: list = field(default_factory=list)     # ConstantSlot
    samplers: list = field(default_factory=list)      # (Var, fetch_slot)
    block_var: int | None = None                      # uniform block var id
    # role lookup the lowering pass uses: var_id -> ("kind", payload)
    roles: dict = field(default_factory=dict)


def _usage_for_location(loc: int):
    import xdis.container_emit as ce
    table = {0: (ce.USAGE_POSITION, 0), 1: (ce.USAGE_COLOR, 0),
             2: (ce.USAGE_TEXCOORD, 0)}
    return table.get(loc, (ce.USAGE_TEXCOORD, max(0, loc - 2)))


def resolve_interface(m: spirv.Module) -> ShaderInterface:
    import xdis.container_emit as ce
    types = resolve_types(m)
    ep = m.entry_points[0]
    stage = "vertex" if ep[0] == 0 else "pixel"
    si = ShaderInterface(stage)
    _next_const = [0]  # mutable cursor for default-uniform constant regs

    # collect all Variables
    variables = []
    for ins in m.instrs:
        if ins.opcode == 59:  # OpVariable result=ptr-type, sc
            ptr = types.get(ins.result_type)
            sc = ins.operands[0]
            d = _decos(m, ins.result_id)
            loc = d.get(DEC_LOCATION, (None,))[0]
            binding = d.get(DEC_BINDING, (None,))[0]
            builtin = d.get(DEC_BUILTIN, (None,))[0]
            variables.append(Var(
                ins.result_id, m.names.get(ins.result_id, ""),
                sc, ptr.comp if ptr else None, loc, binding, builtin))

    for v in variables:
        if v.storage_class == SC_INPUT:
            if v.builtin in _IGNORED_INPUT_BUILTINS:
                continue
            if stage == "vertex":
                usage, uidx = _usage_for_location(v.location or 0)
                si.vfetch.append((v, usage, uidx))
                si.roles[v.vid] = ("vfetch", len(si.vfetch) - 1)
            else:  # pixel interpolator input
                r = v.location or 0
                shape = types.get(v.pointee)
                usage = (ce.USAGE_TEXCOORD
                         if shape and shape.kind == "vector"
                         and shape.count == 2 else ce.USAGE_COLOR)
                si.interp_in.append((v, r, usage))
                si.roles[v.vid] = ("interp_in", r)
        elif v.storage_class == SC_OUTPUT:
            st_out = types.get(v.pointee)
            if st_out and st_out.kind == "struct":
                # gl_PerVertex: the member decorated BuiltIn Position is
                # gl_Position; stores reach it via AccessChain(var, member).
                for mi in range(len(st_out.members)):
                    for deco, args in m.member_decorations.get(
                            (v.pointee, mi), []):
                        if deco == DEC_BUILTIN and args[0] == \
                                BUILTIN_POSITION:
                            si.position_out = v.vid
                            si.roles[v.vid] = ("position_struct", mi)
                continue
            if v.builtin == BUILTIN_POSITION:
                si.position_out = v.vid
                si.roles[v.vid] = ("position_struct", None)
            elif v.builtin is not None:
                continue  # PointSize etc. ignored in v1
            elif stage == "vertex":
                o = v.location or 0
                shape = types.get(v.pointee)
                usage = (ce.USAGE_TEXCOORD
                         if shape and shape.kind == "vector"
                         and shape.count == 2 else ce.USAGE_COLOR)
                si.interp_out.append((v, o, usage))
                si.roles[v.vid] = ("interp_out", o)
            else:  # pixel color output
                si.color_out.append((v, v.location or 0))
                si.roles[v.vid] = ("color", v.location or 0)
        elif v.storage_class == SC_UNIFORM:
            # uniform block: struct (maybe via Block decoration)
            st = types.get(v.pointee)
            if st and st.kind == "struct":
                si.block_var = v.vid
                si.roles[v.vid] = ("block", None)
                for mi, mtid in enumerate(st.members):
                    off = None
                    for deco, args in m.member_decorations.get(
                            (v.pointee, mi), []):
                        if deco == DEC_OFFSET:
                            off = args[0]
                    reg = (off // 16) if off is not None else mi
                    mt = types.get(mtid)
                    regs = mt.count if mt and mt.kind == "matrix" else 1
                    mname = m.member_names.get((v.pointee, mi), f"m{mi}")
                    si.constants.append(ConstantSlot(mname, reg, regs, mi))
        elif v.storage_class == SC_UNIFORM_CONSTANT:
            st = types.get(v.pointee)
            if st and st.kind in ("sampledimage", "image", "sampler"):
                slot = v.binding if v.binding is not None else len(si.samplers)
                si.samplers.append((v, slot))
                si.roles[v.vid] = ("sampler", slot)
            elif st and st.kind in ("matrix", "vector", "float"):
                # default-block uniform (bare `uniform mat4 u_mvp;`):
                # constant register base = Location, else next free.
                regs = st.count if st.kind == "matrix" else 1
                reg = v.location if v.location is not None else _next_const[0]
                _next_const[0] = max(_next_const[0], reg + regs)
                si.constants.append(
                    ConstantSlot(v.name or f"u{reg}", reg, regs, -1))
                si.roles[v.vid] = ("constant", (reg, regs))

    si.vfetch.sort(key=lambda x: x[0].location or 0)
    si.interp_in.sort(key=lambda x: x[1])
    si.interp_out.sort(key=lambda x: x[1])
    return si


def main():
    for path in sys.argv[1:]:
        m = spirv.Module.parse(Path(path).read_bytes())
        si = resolve_interface(m)
        print(f"=== {path}  [{si.stage}]")
        for v, usage, uidx in si.vfetch:
            print(f"  vfetch  {v.name:14} loc{v.location} -> "
                  f"usage{usage}.{uidx}")
        for v, r, usage in si.interp_in:
            print(f"  interp_in  {v.name:11} -> r{r} (usage{usage})")
        for v, o, usage in si.interp_out:
            print(f"  interp_out {v.name:11} -> o{o} (usage{usage})")
        if si.position_out is not None:
            print(f"  position   %{si.position_out} -> oPos")
        for v, idx in si.color_out:
            print(f"  color_out  {v.name:11} -> oC{idx}")
        for cs in si.constants:
            print(f"  constant   {cs.name:11} -> c{cs.reg}"
                  f"{'' if cs.regs == 1 else f'..c{cs.reg+cs.regs-1}'}")
        for v, slot in si.samplers:
            print(f"  sampler    {v.name:11} -> tf{slot}")


if __name__ == "__main__":
    main()
