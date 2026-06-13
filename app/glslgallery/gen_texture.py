#!/usr/bin/env python3
"""Embed a PNG as the gallery's 2D texture (build-time; no runtime decoder).

Reads the user's texture.png from the fable root, center-crops to square,
downscales to 256x256, and emits source/texture_data.h as an ARGB8888 u32 array
(0xAARRGGBB, opaque) matching the Xe_CreateTexture(XE_FMT_8888|XE_FMT_ARGB)
pixel convention the gallery already uses. Regenerate with: python gen_texture.py
"""
from pathlib import Path

from PIL import Image

SRC = Path("D:/claude/fable/texture.png")
SIZE = 256
OUT = Path(__file__).resolve().parent / "source" / "texture_data.h"


def main():
    img = Image.open(SRC).convert("RGBA")
    w, h = img.size
    s = min(w, h)                                  # center-crop to square
    img = img.crop(((w - s) // 2, (h - s) // 2,
                    (w + s) // 2, (h + s) // 2)).resize(
        (SIZE, SIZE), Image.LANCZOS)
    px = img.load()
    vals = []
    for y in range(SIZE):
        for x in range(SIZE):
            r, g, b, _a = px[x, y]
            vals.append(0xFF000000 | (r << 16) | (g << 8) | b)
    out = [f"/* generated from {SRC.name} ({w}x{h} -> {SIZE}x{SIZE}) by "
           f"gen_texture.py; do not edit. */",
           f"#define GALLERY_TEX_SIZE {SIZE}",
           f"static const unsigned int gallery_tex[{SIZE}*{SIZE}] = {{"]
    for i in range(0, len(vals), 12):
        out.append("\t" + ", ".join(f"0x{v:08x}" for v in vals[i:i + 12]) + ",")
    out.append("};")
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT}  ({SIZE}x{SIZE}, {len(vals)} px)")


if __name__ == "__main__":
    main()
