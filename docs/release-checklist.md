# Release checklist — going public

Ordered pre-flight. Items marked **[USER]** need a human decision; the rest
are mechanical. Companion document: [`license-audit.md`](license-audit.md)
(the audit's action list is step 1 here — do it first, everything else is
cosmetic by comparison).

## 1. License actions (from the audit — blocking)

- [ ] **[USER]** Choose the license: MIT or BSD-3-Clause (tradeoffs in
      audit §2). Add `LICENSE` at repo root, including the
      "toolchain output carries no obligations" sentence.
- [ ] Add `THIRD_PARTY_NOTICES.md`: Xenia BSD-3 full text + freedreno MIT
      (for `xdis/xenos_ucode.py`), libxenon BSD-3 (demo ELFs + copied cube
      HLSL), XenosRecomp courtesy mention, devkitPRO-template note for the
      app Makefiles.
- [ ] Repoint `xdis/xenos_ucode.py`'s header from `reference/xenia/LICENSE`
      to the in-repo notices file.
- [ ] Add SPDX headers to our original files.
- [ ] **[USER]** Corpus decision: keep `corpus/*.bin` (recommended) or drop.
      Either way, add `corpus/README.md` with per-file provenance,
      regeneration instructions, and removal-on-request note (audit §3).
- [ ] Add attribution lines to `corpus/cube_vs.hlsl` / `cube_ps.hlsl`.

## 2. Repo hygiene

Current state (from `git ls-files`): `.gitignore` covers `reference/`,
`__pycache__/`, `*.pyc` — and nothing else. **Build artifacts ARE
committed:**

- `app/{triangle,spincube,texcube,combiner}/build/` — `main.o`, `main.d`,
  `*.elf.map`
- `app/*/{*.elf,*.elf32}` — linked binaries (~MB-scale, will bloat history
  on every rebuild)
- `tests/xcomb_parity/build/` — `harness.exe`, `harness.obj`, `xcomb.obj`
  (a committed Windows .exe is also a malware-scanner magnet on a fresh
  repo)

Recommended:

- [ ] Extend `.gitignore`:
      ```
      app/*/build/
      app/*.elf
      app/*/*.elf
      app/*/*.elf32
      tests/xcomb_parity/build/
      examples/__pycache__/
      ```
- [ ] `git rm -r --cached` the paths above and commit ("remove build
      artifacts from tracking"). History will still contain them — that's
      fine; the repo is one day old and the alternative (history rewrite)
      isn't worth it. **[USER]** confirm no history rewrite wanted.
- [ ] **[USER]** Where should ready-to-boot demo binaries live instead?
      Recommendation: attach `*.elf32` to a GitHub Release ("v0.1 —
      hardware-verified demos") rather than tracking them.
- [ ] Keep `examples/out/*.bin` + `*.hlsl` tracked — they are our own
      toolchain's output and the oracle's reading of it; the demo apps'
      `gen_header.py` consume them, and they double as instant proof of
      what the tools emit.
- [ ] **[USER]** Untracked files decision (audit §7): `xasm/textasm.py` +
      `tests/test_textasm.py` — commit as clearly-marked WIP, or hold for a
      follow-up release. If committed, write the missing `docs/textasm.md`
      it references (or strip the reference). Same deliberate decision for
      any other in-flight untracked work present at push time (at audit
      time: `app/rtcomb/`, `examples/author_features.py`,
      `examples/out/feat_*`) — run `git status` last thing before push and
      account for every `??` line.
- [ ] Fresh-clone smoke test in a clean directory **without `reference/`**:
      `python tests/test_exact_total.py`, `test_encode_fuzz.py`,
      `test_container_regen.py`, the three `examples/author_*.py` scripts,
      and `python xdis/xdis.py corpus/vs0.bin`. (Verified by inspection:
      none of the tracked tests import from `reference/`; `test_xcomb_parity`
      additionally needs MSVC — document that in its docstring or a
      CONTRIBUTING note.)
- [ ] Scrub for local-machine paths in tracked files (WSL paths in
      `wslbuild.sh` are intentional/documented; check `*.elf.map` files are
      gone with the build dirs — they embed full local paths today).

## 3. README finalization

- [ ] Insert the four photos as `docs/photos/triangle.jpg`,
      `docs/photos/texcube.jpg`, `docs/photos/combiner_modes.jpg`,
      `docs/photos/hud_cpu0_gpu3.jpg` (filenames already referenced by
      README placeholders) and convert the placeholder table to real
      `![...]()` embeds. The user holds the photos; pick the clearest shot
      per milestone — the HUD shot should legibly show `CPU 0 / GPU 3`.
- [ ] Re-read every README claim against the final tree (e.g. if the text
      assembler gets committed in step 2, update the two "in progress, not
      yet committed" mentions).
- [ ] Add the chosen license name to the README License section.

## 4. Naming **[USER]**

- [ ] Decide the real name ("xost" is a working title; README/PLAN say so).
      Check collisions: GitHub search, PyPI (if a package is ever
      intended), and the 360-scene's existing tool names.
- [ ] If the name changes: update README, PLAN.md, file docstrings that say
      "xost", and decide whether identifier-level strings
      (`xost_vs0`/`xost_ps1` shader names, `app/*/source/xost_shaders.h`)
      are worth renaming. Recommendation: prose yes, identifiers optional —
      they're hardware-verified artifacts and renaming invites
      re-verification.

## 5. GitHub repo setup

- [ ] Repo description (one line): *"Open-source Xbox 360 (Xenos) GPU
      shader toolchain — disassembler, assembler, and N64-combiner
      compiler; no Microsoft tooling; hardware-verified via libxenon."*
- [ ] Topics: `xbox-360`, `xenos`, `gpu`, `shader`, `assembler`,
      `disassembler`, `homebrew`, `libxenon`, `free60`, `reverse-engineering`,
      `n64`, `emulation`.
- [ ] Default branch + protections per taste; enable Issues, disable Wiki
      (docs live in-repo).
- [ ] License detection: confirm GitHub badges the chosen LICENSE correctly.
- [ ] Optional, cheap: a GitHub Actions workflow running the three
      stdlib-only Python test files on push (no deps to install — the CI
      file is ~15 lines). MSVC parity test can stay local-only for now.
- [ ] Create the v0.1 Release with the `.elf32` demos attached (step 2) and
      the photos in the release notes.

## 6. Announcement draft (scene forums — GBAtemp / XboxHacker style)

Factual bullets, no hype; the photos do the talking:

- First way to author Xbox 360 GPU shaders with zero Microsoft tooling —
  no leaked XDK, no XNA shader compiler.
- What it is today: a Xenos microcode disassembler (byte-exact round-trip),
  an assembler library (Python), and an N64-style color-combiner →
  pixel-shader compiler in both Python and dependency-free C99 (the C
  version generates shaders at runtime on the console, byte-identical to
  the Python one).
- Hardware-verified on an RGH console via stock libxenon: triangles,
  spinning cube with live CPU/GPU HUD, textured cube, and 9 combiner modes
  including 2-cycle / dual-texture — photos in the README.
- Validated against XenosRecomp as an independent oracle plus fuzzing and
  byte-exact regeneration of real XDK-compiled blobs.
- What it is NOT yet: no HLSL/GLSL frontend (planned: Mesa/NIR via the
  freedreno a2xx backend — Adreno 200 is Xenos-derived); text assembler in
  progress; control flow / loops not yet hardware-proven.
- Why it matters: every open 360 project that does 3D — including
  mupen64-360 — has shipped leaked-XDK compiler output since the beginning.
  This is the path off that.
- Ask: testers with RGH/JTAG units, and emulator devs who want a legal
  shader backend (the combiner generator was built for exactly that).
- Link the license-audit doc up front; invite provenance scrutiny rather
  than waiting for it.

## 7. Post-release: questions to expect (prep answers now)

- **"Isn't this just Xenia's code?"** — The ISA *tables* in one file are
  transcribed from Xenia under BSD-3 with full attribution (see
  THIRD_PARTY_NOTICES). The encoders, builder, combiner compilers, container
  emitter, and apps are original. Link audit §1.
- **"Those corpus .bin files are Microsoft compiler output — can you ship
  them?"** — Honest answer prepared in audit §3: 4 small interoperability
  test vectors, public in scene repos since ~2009, provenance documented,
  regeneration instructions included, removal-on-request stated. Link the
  audit; don't improvise in-thread.
- **"Does this use the leaked XDK?"** — No. XDK *output* (already public in
  GPL scene repos) is used read-only as test vectors; no Microsoft tool, SDK,
  header, or library is used or required anywhere in the pipeline.
- **"GPL concerns from mupen64-360/libxenon?"** — libxenon is BSD-3 (a
  common misconception that it's GPL — audit §4 quotes its LICENSE);
  mupen64-360 was only ever a *source of test blobs*, no code taken.
- **"When GLSL/HLSL?"** — Point at PLAN.md Phase 3 (Mesa/NIR, freedreno
  a2xx). No date promises.
- **"Will this work with my RGH3/JTAG/XDK-console?"** — Verified on one
  zero-fuse RGH unit; libxenon-bootable consoles should behave identically,
  testers welcome (that's the announcement's ask).
- Watch for: someone asking about the quarantine note in commit `9048202` —
  answer is in audit §6 (unknown-origin file kept out of the repo,
  clean-room replacement written; the disclosure is deliberate).

## 8. After the dust settles

- [ ] Update project memory: libxenon license correction (BSD-3, not GPL),
      final name, chosen license.
- [ ] PLAN.md risk #4 references "GPL (libxenon…)" — correct it or annotate.
- [ ] Open tracking issues for: text assembler, HLSL/GLSL frontend (Phase
      3), unproven hardware features (bool constants/cjmp, loops,
      predication, ubyte4, MRT/depth export).
