#!/usr/bin/env python3
"""Declarative Xenos shader Program builder on top of xasm.encode.

The caller declares fetches, ALU instructions, and stage boundaries
(alloc_* markers); finish() performs:
  - instruction slot layout and control-flow program construction
    (exec/exec_end entries with addresses, counts, sequence bits),
  - register-usage scanning -> SQ_PROGRAM_CNTL,
  - VertexElement / Interpolator / export-slot metadata,
  - container emission via xdis.container_emit.

Instruction *encodings* are supplied as field dicts (matching the
xasm.encode builder kwargs); proven field templates are typically extracted
from the golden corpus via the decoders and adapted (see
examples/author_textured.py).

Conventions baked in (validated on hardware by app/triangle + app/spincube):
exec entries use is_predicate_clean=True; fetch-bearing execs carry vc_hi=15;
sequence bit0 marks fetch slots; VS tail flags: POSITION=0x1, COLOR=0x3,
TEXCOORD=0x0; interpolator flags: VS (index<<4)|nibble, PS nibble
(nibble: TEXCOORD=0x3, COLOR=0xF).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import xdis.container_emit as ce  # noqa: E402
import xdis.xenos_ucode as xu  # noqa: E402
from xasm import encode as enc  # noqa: E402

K = xu.ControlFlowOpcode

MAX_EXEC_COUNT = 6  # exec count field is 3 bits; stay below the 7 limit

_USAGE_NIBBLE = {ce.USAGE_TEXCOORD: 0x3, ce.USAGE_COLOR: 0xF}
_VE_FLAGS = {ce.USAGE_POSITION: 0x1, ce.USAGE_COLOR: 0x3,
             ce.USAGE_TEXCOORD: 0x0}


def swizzle4(s: str) -> int:
    """ALU source swizzle text ('wzxy') -> 8-bit delta encoding."""
    assert len(s) == 4
    out = 0
    for i, ch in enumerate(s):
        out |= ((("xyzw".index(ch)) - i) & 3) << (2 * i)
    return out


def mask4(s: str) -> int:
    """write mask text ('xy', 'xyzw') -> 4-bit mask."""
    out = 0
    for ch in s:
        out |= 1 << "xyzw".index(ch)
    return out


class _Instr:
    def __init__(self, kind: str, triple, is_fetch: bool):
        self.kind = kind
        self.triple = tuple(triple)
        self.is_fetch = is_fetch
        self.slot = -1


class _Program:
    shader_type = None

    def __init__(self):
        self._stages: list[list[_Instr]] = [[]]  # instr blocks
        self._allocs: list[int] = []             # alloc_type between stages
        self._max_reg = 0
        self.interpolators: list[ce.Interpolator] = []
        self._export_stage_idx = None

    # ---- declarations ----

    def _track_regs(self, fields: dict):
        for key in ("dest", "src"):
            if key in fields:
                self._max_reg = max(self._max_reg, fields[key])
        if "vector_dest" in fields and not fields.get("is_export"):
            self._max_reg = max(self._max_reg, fields["vector_dest"])
        for i in (1, 2, 3):
            if fields.get(f"src{i}_is_temp"):
                self._max_reg = max(self._max_reg, fields[f"src{i}_reg"])

    def alu(self, **fields) -> _Instr:
        self._track_regs(fields)
        ins = _Instr("alu", enc.build_alu(**fields), is_fetch=False)
        self._stages[-1].append(ins)
        return ins

    def _alloc(self, alloc_type: int):
        self._allocs.append(alloc_type)
        self._stages.append([])

    # ---- layout ----

    def _cf_entry_count(self) -> int:
        n = 0
        for i, stage in enumerate(self._stages):
            chunks = max(1, (len(stage) + MAX_EXEC_COUNT - 1)
                         // MAX_EXEC_COUNT) if stage else 0
            n += chunks
            if i < len(self._allocs):
                n += 1
        return n

    def _build_ucode(self) -> bytes:
        n_cf = self._cf_entry_count()
        first_slot = (n_cf + 1) // 2  # cf pairs occupy one slot each
        # assign slots
        slot = first_slot
        for stage in self._stages:
            for ins in stage:
                ins.slot = slot
                slot += 1
        total_slots = slot

        cf = []
        n_stages = len(self._stages)
        for si, stage in enumerate(self._stages):
            chunks = [stage[i:i + MAX_EXEC_COUNT]
                      for i in range(0, len(stage), MAX_EXEC_COUNT)] or []
            for ci, chunk in enumerate(chunks):
                is_last = (si == n_stages - 1 and ci == len(chunks) - 1)
                seq = 0
                has_fetch = False
                for k, ins in enumerate(chunk):
                    if ins.is_fetch:
                        seq |= 0b01 << (2 * k)
                        has_fetch = True
                vc = 15 if (has_fetch and self.shader_type == "vertex") else 0
                cf.append(enc.build_cf_exec(
                    opcode=K.kExecEnd if is_last else K.kExec,
                    address=chunk[0].slot, count=len(chunk),
                    is_yield=False, sequence=seq,
                    vc_hi=vc, vc_lo=0,
                    is_predicate_clean=True, addressing_mode=0))
            if si < len(self._allocs):
                cf.append(enc.build_cf_alloc(
                    opcode=K.kAlloc, size=0, is_unserialized=False,
                    alloc_type=self._allocs[si]))
        while len(cf) % 2:
            cf.append((0, 0))  # cnop
        assert len(cf) // 2 == first_slot, "CF layout self-check failed"

        dwords = []
        for i in range(0, len(cf), 2):
            dwords.extend(enc.pack_cf_pair(cf[i], cf[i + 1]))
        for stage in self._stages:
            for ins in stage:
                dwords.extend(ins.triple)
        assert len(dwords) == total_slots * 3
        return struct.pack(f">{len(dwords)}I", *dwords)


class VertexProgram(_Program):
    shader_type = "vertex"

    def __init__(self):
        super().__init__()
        self.vertex_elements: list[ce.VertexElement] = []

    def fetch(self, usage: int, template: dict, dest: int,
              usage_index: int = 0) -> _Instr:
        """Template vfetch (patched at load by Xe_ShaderApplyVFetchPatches)."""
        fields = dict(template, dest=dest)
        self._track_regs(fields)
        ins = _Instr("vfetch", enc.build_vfetch_full(**fields), is_fetch=True)
        self._stages[-1].append(ins)
        self.vertex_elements.append(
            ce.VertexElement(0, usage, usage_index,
                             flags=_VE_FLAGS.get(usage, 0)))
        ins._ve = self.vertex_elements[-1]
        return ins

    def alloc_position(self):
        self._alloc(1)

    def alloc_interpolators(self):
        self._alloc(2)
        self._export_stage_idx = len(self._stages) - 1

    def export(self, o: int, usage: int, template: dict,
               usage_index: int = 0, **overrides) -> _Instr:
        """Interpolator export ALU (template + vector_dest=o)."""
        fields = dict(template, vector_dest=o, **overrides)
        ins = self.alu(**fields)
        self.interpolators.append(ce.Interpolator(
            o, usage, usage_index,
            flags=(len(self.interpolators) << 4)
                  | _USAGE_NIBBLE.get(usage, 0)))
        return ins

    def finish(self, constants: list) -> bytes:
        ucode = self._build_ucode()
        for ins_list in self._stages:
            for ins in ins_list:
                if hasattr(ins, "_ve"):
                    ins._ve.slot = ins.slot
        export_slots = [ins.slot
                        for ins in self._stages[self._export_stage_idx]]
        return ce.build_container(
            "vertex", ucode, constants=constants,
            program_control=(
                self._max_reg |               # vs_num_reg (value-1 wants
                (1 << 16) |                   # max reg index; vs_resource
                ((len(self.interpolators) - 1) << 20)),  # vs_export_count
            context_misc=0,
            interpolators=self.interpolators,
            vertex_elements=self.vertex_elements,
            vs_tail_count=len(self.interpolators),
            export_slots=export_slots)


class PixelProgram(_Program):
    shader_type = "pixel"

    def input(self, r: int, usage: int, usage_index: int = 0):
        """Declare interpolator arriving in rN (must match VS export order)."""
        assert r == len(self.interpolators), "declare inputs in r0,r1,... order"
        self.interpolators.append(ce.Interpolator(
            r, usage, usage_index, flags=_USAGE_NIBBLE.get(usage, 0)))
        self._max_reg = max(self._max_reg, r)

    def tfetch(self, template: dict, dest: int, src: int) -> _Instr:
        fields = dict(template, dest=dest, src=src)
        self._track_regs(fields)
        ins = _Instr("tfetch", enc.build_tfetch(**fields), is_fetch=True)
        self._stages[-1].append(ins)
        return ins

    def alloc_colors(self):
        self._alloc(2)

    def finish(self, constants: list) -> bytes:
        ucode = self._build_ucode()
        return ce.build_container(
            "pixel", ucode, constants=constants,
            program_control=(self._max_reg << 8) | (2 << 27),
            context_misc=0x8,
            field10=0x20000,
            interpolators=self.interpolators,
            ps_field18=(len(self.interpolators) << 16) | 7,
            ps_outputs=0x1)
