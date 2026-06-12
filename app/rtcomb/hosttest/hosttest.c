/* hosttest.c — host-side sanity check for the rtcomb console demo.
 *
 * Proves on the PC, before ever running on hardware, that:
 *  1. every entry of the EXACT deterministic config table the console
 *     uses (source/rtcomb_table.h, shared header) generates a container
 *     (xcomb_generate != 0) that python xdis/xdis.py parses cleanly
 *     (subprocess exit code 0);
 *  2. the console's random-config value ranges can never get stuck:
 *     every 1-cycle RGB combo over the random source pool (ZERO..ENV,
 *     7^4 = 2401) plus a seeded sample of 2-cycle combos (cycle 1 drawn
 *     from ZERO..COMBINED, mirroring rnd_cycle in source/main.c)
 *     generates successfully and fits XCOMB_MAX_CONTAINER.
 *
 * Build & run (from this directory, MSVC x64 shell):
 *   cl /nologo /W4 /WX hosttest.c ..\..\..\xcomb\xcomb.c /Fe:hosttest.exe
 *   hosttest.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../source/rtcomb_table.h"

/* repo root: argv[1] if given, else assumes CWD = this directory */
static const char *repo_root = "..\\..\\..";

static unsigned char buf[XCOMB_MAX_CONTAINER];

static size_t gen(const rtcomb_config *cfg)
{
	return xcomb_generate(&cfg->c0, cfg->two_cycle ? &cfg->c1 : NULL,
			      buf, sizeof(buf));
}

/* same xorshift32 as the console app */
static unsigned int rng_state = 0x9E3779B9u;

static unsigned int xorshift32(void)
{
	rng_state ^= rng_state << 13;
	rng_state ^= rng_state >> 17;
	rng_state ^= rng_state << 5;
	return rng_state;
}

static xcomb_src rnd_src(int allow_combined)
{
	return (xcomb_src)(xorshift32() % (allow_combined ? 8u : 7u));
}

static void rnd_cycle(xcomb_cycle *c, int allow_combined)
{
	c->a = rnd_src(allow_combined);
	c->b = rnd_src(allow_combined);
	c->c = rnd_src(allow_combined);
	c->d = rnd_src(allow_combined);
	c->aa = XCOMB_ZERO; c->ab = XCOMB_ZERO;
	c->ac = XCOMB_ZERO; c->ad = XCOMB_ONE;
}

int main(int argc, char **argv)
{
	int i, fails = 0;

	if (argc > 1)
		repo_root = argv[1];

	/* 1. deterministic table: generate + xdis parse */
	for (i = 0; i < RTCOMB_TABLE_LEN; i++) {
		const rtcomb_config *cfg = &rtcomb_table[i];
		size_t len = gen(cfg);
		if (!len) {
			printf("FAIL det[%d] %s: xcomb_generate == 0\n",
			       i, cfg->name);
			fails++;
			continue;
		}
		char fname[64], cmd[256];
		sprintf(fname, "rt_det_%d.bin", i);
		FILE *f = fopen(fname, "wb");
		if (!f || fwrite(buf, 1, len, f) != len) {
			printf("FAIL det[%d] %s: cannot write %s\n",
			       i, cfg->name, fname);
			fails++;
			if (f)
				fclose(f);
			continue;
		}
		fclose(f);
		sprintf(cmd, "python %s\\xdis\\xdis.py %s > nul 2>&1",
			repo_root, fname);
		int rc = system(cmd);
		printf("det[%d] %-10s %d-cycle: %3u bytes, xdis rc=%d%s\n",
		       i, cfg->name, cfg->two_cycle ? 2 : 1,
		       (unsigned)len, rc, rc ? "  <-- FAIL" : "");
		if (rc != 0)
			fails++;
	}

	/* 2a. exhaustive 1-cycle RGB sweep over the random source pool */
	{
		rtcomb_config cfg;
		int a, b, c, d, swept = 0, bad = 0;
		memset(&cfg, 0, sizeof(cfg));
		cfg.two_cycle = 0;
		cfg.c0.aa = XCOMB_ZERO; cfg.c0.ab = XCOMB_ZERO;
		cfg.c0.ac = XCOMB_ZERO; cfg.c0.ad = XCOMB_ONE;
		for (a = 0; a <= XCOMB_ENV; a++)
		for (b = 0; b <= XCOMB_ENV; b++)
		for (c = 0; c <= XCOMB_ENV; c++)
		for (d = 0; d <= XCOMB_ENV; d++) {
			cfg.c0.a = (xcomb_src)a; cfg.c0.b = (xcomb_src)b;
			cfg.c0.c = (xcomb_src)c; cfg.c0.d = (xcomb_src)d;
			swept++;
			if (!gen(&cfg))
				bad++;
		}
		printf("1-cycle sweep: %d combos, %d failed generation\n",
		       swept, bad);
		fails += bad;
	}

	/* 2b. seeded 2-cycle random sample, mirroring the console path */
	{
		rtcomb_config cfg;
		int n, bad = 0;
		memset(&cfg, 0, sizeof(cfg));
		cfg.two_cycle = 1;
		for (n = 0; n < 4096; n++) {
			rnd_cycle(&cfg.c0, 0);
			rnd_cycle(&cfg.c1, 1);
			if (!gen(&cfg))
				bad++;
		}
		printf("2-cycle random sample: 4096 combos, %d failed\n",
		       bad);
		fails += bad;
	}

	printf(fails ? "HOSTTEST FAIL (%d)\n" : "HOSTTEST PASS\n", fails);
	return fails ? 1 : 0;
}
