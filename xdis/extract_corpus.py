#!/usr/bin/env python3
"""Extract the golden shader-blob corpus from open-source 360 homebrew.

Currently: mupen64-360's source/shaders.S — GNU as syntax, one labeled
.byte-array per shader, with a size table. Output: corpus/<name>.bin
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHADERS_S = ROOT / "reference" / "mupen64-360" / "source" / "shaders.S"
CORPUS = ROOT / "corpus"

LABEL_RE = re.compile(r"^(data_[A-Za-z0-9_]+):")
BYTE_RE = re.compile(r"^\s*\.byte\s+(.*)$")
LONG_RE = re.compile(r"^\s*\.long\s+(\d+)\s*$")
SIZE_LABEL_RE = re.compile(r"^([A-Za-z0-9_]+_size_table):")


def parse(path: Path):
    blobs = {}        # label -> bytearray
    size_tables = {}  # table label -> list of sizes
    cur_blob = None
    cur_sizes = None
    for line in path.read_text().splitlines():
        m = LABEL_RE.match(line)
        if m:
            cur_blob, cur_sizes = m.group(1), None
            blobs[cur_blob] = bytearray()
            continue
        m = SIZE_LABEL_RE.match(line)
        if m:
            cur_sizes, cur_blob = m.group(1), None
            size_tables[cur_sizes] = []
            continue
        m = BYTE_RE.match(line)
        if m and cur_blob is not None:
            for tok in m.group(1).split(","):
                blobs[cur_blob].append(int(tok.strip(), 0))
            continue
        m = LONG_RE.match(line)
        if m and cur_sizes is not None:
            size_tables[cur_sizes].append(int(m.group(1)))
            continue
        if line.startswith((".globl", ".align")) or not line.strip():
            continue
        # any other directive ends the current blob/table
        cur_blob = cur_sizes = None
    return blobs, size_tables


def main():
    blobs, size_tables = parse(SHADERS_S)
    CORPUS.mkdir(exist_ok=True)
    # declared sizes: vs_size_table -> data_vs<N>, ps_fb_size_table -> data_ps_fb<N>
    declared = {}
    for table, sizes in size_tables.items():
        prefix = "data_" + table.replace("_size_table", "")
        for i, sz in enumerate(sizes):
            declared[f"{prefix}{i}"] = sz
    ok = True
    for label, data in blobs.items():
        want = declared.get(label)
        if want is not None and len(data) < want:
            print(f"ERROR {label}: extracted {len(data)} bytes < declared {want}")
            ok = False
            continue
        if want is not None and len(data) > want:
            # .align padding tail; trim to declared size
            data = data[:want]
        out = CORPUS / (label.removeprefix("data_") + ".bin")
        out.write_bytes(bytes(data))
        note = f"(declared {want})" if want is not None else "(no size entry)"
        print(f"{out.name}: {len(data)} bytes {note}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
