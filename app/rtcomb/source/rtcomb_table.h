/* rtcomb_table.h — the deterministic combiner-config table.
 *
 * Shared by the console app (source/main.c) and the host-side sanity test
 * (hosttest/hosttest.c), so the exact configs the console feeds to
 * xcomb_generate() are proven valid (generate != 0 and xdis-parseable)
 * on the host before ever running on hardware.
 *
 * Entries mirror well-known modes from examples/author_combiners.py BANK
 * so the visuals are immediately recognizable next to app/combiner:
 * a mix of 1-cycle and 2-cycle, covering TEXEL0/TEXEL1/SHADE/PRIM/ENV/ONE
 * and COMBINED (cycle 1 only).
 */
#ifndef RTCOMB_TABLE_H
#define RTCOMB_TABLE_H

#include "../../../xcomb/xcomb.h"

typedef struct {
	xcomb_cycle c0;
	xcomb_cycle c1;     /* only meaningful when two_cycle != 0 */
	int two_cycle;
	const char *name;
} rtcomb_config;

#define RT_Z  XCOMB_ZERO
#define RT_O  XCOMB_ONE
#define RT_T  XCOMB_TEXEL0
#define RT_T1 XCOMB_TEXEL1
#define RT_S  XCOMB_SHADE
#define RT_P  XCOMB_PRIM
#define RT_E  XCOMB_ENV
#define RT_C  XCOMB_COMBINED

/* alpha pipeline forced opaque: (0-0)*0+1 */
#define RT_A_OPAQUE RT_Z, RT_Z, RT_Z, RT_O

static const rtcomb_config rtcomb_table[] = {
	/* 0: tex * shade (alpha same as color) */
	{ {RT_T, RT_Z, RT_S, RT_Z,  RT_T, RT_Z, RT_S, RT_Z},
	  {0, 0, 0, 0, 0, 0, 0, 0}, 0, "modulate" },
	/* 1: tex * PRIM, opaque */
	{ {RT_T, RT_Z, RT_P, RT_Z,  RT_A_OPAQUE},
	  {0, 0, 0, 0, 0, 0, 0, 0}, 0, "tex_prim" },
	/* 2: lerp(ENV, PRIM, tex), opaque */
	{ {RT_P, RT_E, RT_T, RT_E,  RT_A_OPAQUE},
	  {0, 0, 0, 0, 0, 0, 0, 0}, 0, "lerp_pe_t" },
	/* 3: (1 - tex) * PRIM, opaque */
	{ {RT_O, RT_T, RT_P, RT_Z,  RT_A_OPAQUE},
	  {0, 0, 0, 0, 0, 0, 0, 0}, 0, "inv_tex_p" },
	/* 4: 2-cycle: (t0 * t1) * shade */
	{ {RT_T, RT_Z, RT_T1, RT_Z,  RT_T, RT_Z, RT_T1, RT_Z},
	  {RT_C, RT_Z, RT_S,  RT_Z,  RT_C, RT_Z, RT_S,  RT_Z}, 1, "multitex" },
	/* 5: 2-cycle: t0 * shade + t1 (glow add), opaque */
	{ {RT_T,  RT_Z, RT_S, RT_Z,  RT_T, RT_Z, RT_S, RT_Z},
	  {RT_T1, RT_Z, RT_O, RT_C,  RT_A_OPAQUE}, 1, "add_glow" },
};

#define RTCOMB_TABLE_LEN \
	((int)(sizeof(rtcomb_table) / sizeof(rtcomb_table[0])))

#endif /* RTCOMB_TABLE_H */
