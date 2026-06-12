/* xost texcube — textured spinning cube + CPU/GPU HUD.
 *
 * Two fully open-toolchain shader pairs in one frame:
 *   cube: glsl_vs (mvp + texcoord/color interpolators) + glsl_ps
 *         (tfetch2D * vertex color)
 *   HUD:  xost_vs1 (identity mvp) + xost_ps0 (color passthrough)
 * Texture: 64x64 procedural checkerboard, linear ARGB8888.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <xenos/xe.h>
#include <xenos/xenos.h>
#include <xenos/edram.h>
#include <console/console.h>
#include <ppc/timebase.h>
#include "xost_shaders.h"

/* ---------------- matrices (same as spincube) ---------------- */

typedef float mat4[4][4];

static void m_identity(mat4 m)
{
	memset(m, 0, sizeof(mat4));
	m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0f;
}

static void m_mul(mat4 out, const mat4 a, const mat4 b)
{
	int r, c, k;
	mat4 t;
	for (r = 0; r < 4; r++)
		for (c = 0; c < 4; c++) {
			float s = 0.0f;
			for (k = 0; k < 4; k++)
				s += a[r][k] * b[k][c];
			t[r][c] = s;
		}
	memcpy(out, t, sizeof(mat4));
}

static void m_persp(mat4 m, float fovy, float aspect, float n, float f)
{
	float ct = 1.0f / tanf(fovy * 0.5f);
	memset(m, 0, sizeof(mat4));
	m[0][0] = ct / aspect;
	m[1][1] = ct;
	m[2][2] = f / (n - f);
	m[2][3] = n * f / (n - f);
	m[3][2] = -1.0f;
}

static void m_rot_y(mat4 m, float a)
{
	m_identity(m);
	m[0][0] = cosf(a);  m[0][2] = sinf(a);
	m[2][0] = -sinf(a); m[2][2] = cosf(a);
}

static void m_rot_x(mat4 m, float a)
{
	m_identity(m);
	m[1][1] = cosf(a); m[1][2] = -sinf(a);
	m[2][1] = sinf(a); m[2][2] = cosf(a);
}

static void m_translate_z(mat4 m, float tz)
{
	m_identity(m);
	m[2][3] = tz;
}

/* ---------------- cube: 24 verts (per-face UVs), built at runtime ------- */

static const float corners[8][3] = {
	{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
	{-1, -1,  1}, {1, -1,  1}, {1, 1,  1}, {-1, 1,  1},
};
static const int faces[6][4] = {
	{0, 1, 2, 3}, {5, 4, 7, 6}, {4, 0, 3, 7},
	{1, 5, 6, 2}, {3, 2, 6, 7}, {4, 5, 1, 0},
};
static const float face_col[6][3] = {
	{1.0f, 0.6f, 0.6f}, {0.6f, 1.0f, 0.6f}, {0.6f, 0.6f, 1.0f},
	{1.0f, 1.0f, 0.6f}, {0.6f, 1.0f, 1.0f}, {1.0f, 0.6f, 1.0f},
};
static const float face_uv[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

#define CUBE_FLOATS (24 * 10)
static float cube_verts[CUBE_FLOATS];
static unsigned short cube_idx[36];

static void build_cube(void)
{
	int f, v;
	for (f = 0; f < 6; f++) {
		for (v = 0; v < 4; v++) {
			float *o = &cube_verts[(f * 4 + v) * 10];
			const float *p = corners[faces[f][v]];
			o[0] = p[0]; o[1] = p[1]; o[2] = p[2]; o[3] = 1.0f;
			o[4] = face_col[f][0]; o[5] = face_col[f][1];
			o[6] = face_col[f][2]; o[7] = 1.0f;
			o[8] = face_uv[v][0];  o[9] = face_uv[v][1];
		}
		unsigned short b = f * 4;
		unsigned short *ix = &cube_idx[f * 6];
		ix[0] = b; ix[1] = b + 1; ix[2] = b + 2;
		ix[3] = b; ix[4] = b + 2; ix[5] = b + 3;
	}
}

/* ---------------- 7-segment HUD (same as spincube) ---------------- */

#define HUD_MAX_VERTS 4096
static float hud[HUD_MAX_VERTS * 8];
static int hud_verts;

static void hud_rect(float x, float y, float w, float h, const float *col)
{
	if (hud_verts + 6 > HUD_MAX_VERTS)
		return;
	float x1 = x + w, y1 = y - h;
	float quad[6][2] = {
		{x, y}, {x1, y}, {x1, y1},
		{x, y}, {x1, y1}, {x, y1},
	};
	int i;
	for (i = 0; i < 6; i++) {
		float *v = &hud[hud_verts++ * 8];
		v[0] = quad[i][0]; v[1] = quad[i][1];
		v[2] = 0.0f;       v[3] = 1.0f;
		v[4] = col[0]; v[5] = col[1]; v[6] = col[2]; v[7] = 1.0f;
	}
}

static const unsigned char seg_digit[10] = {
	0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F
};
#define SEG_C 0x39
#define SEG_P 0x73
#define SEG_U 0x3E
#define SEG_G 0x3D
#define SEG_PCT 0x49

static void hud_char(float cx, float cy, float cw, float ch,
		     unsigned char mask, const float *col)
{
	float th = cw * 0.22f, half = ch * 0.5f;
	if (mask & 0x01) hud_rect(cx + th, cy, cw - 2 * th, th, col);
	if (mask & 0x02) hud_rect(cx + cw - th, cy - th, th, half - th, col);
	if (mask & 0x04) hud_rect(cx + cw - th, cy - half - th * 0.5f, th,
				  half - th * 0.5f, col);
	if (mask & 0x08) hud_rect(cx + th, cy - ch + th, cw - 2 * th, th, col);
	if (mask & 0x10) hud_rect(cx, cy - half - th * 0.5f, th,
				  half - th * 0.5f, col);
	if (mask & 0x20) hud_rect(cx, cy - th, th, half - th, col);
	if (mask & 0x40) hud_rect(cx + th, cy - half + th * 0.5f,
				  cw - 2 * th, th, col);
}

static void hud_row(float x, float y, const unsigned char *label, int nlabel,
		    int value, const float *col)
{
	float cw = 0.030f, ch = 0.075f, adv = cw * 1.45f;
	int i;
	for (i = 0; i < nlabel; i++) {
		hud_char(x, y, cw, ch, label[i], col);
		x += adv;
	}
	x += adv * 0.5f;
	if (value > 999) value = 999;
	if (value < 0) value = 0;
	if (value >= 100) hud_char(x, y, cw, ch,
				   seg_digit[(value / 100) % 10], col);
	x += adv;
	if (value >= 10) hud_char(x, y, cw, ch,
				  seg_digit[(value / 10) % 10], col);
	x += adv;
	hud_char(x, y, cw, ch, seg_digit[value % 10], col);
	x += adv;
	hud_char(x, y, cw, ch, SEG_PCT, col);
}

/* ---------------- main ---------------- */

int main(void)
{
	xenos_init(VIDEO_MODE_AUTO);
	console_init();
	printf("xost texcube: textured + HUD, two open shader pairs\n");

	struct XenosDevice _xe, *xe = &_xe;
	Xe_Init(xe);

	struct XenosSurface *fb = Xe_GetFramebufferSurface(xe);
	Xe_SetRenderTarget(xe, fb);

	static const struct XenosVBFFormat vbf_tex =
	{
		3, {
			{XE_USAGE_POSITION, 0, XE_TYPE_FLOAT4},
			{XE_USAGE_COLOR,    0, XE_TYPE_FLOAT4},
			{XE_USAGE_TEXCOORD, 0, XE_TYPE_FLOAT2},
		}
	};
	static const struct XenosVBFFormat vbf_plain =
	{
		2, {
			{XE_USAGE_POSITION, 0, XE_TYPE_FLOAT4},
			{XE_USAGE_COLOR,    0, XE_TYPE_FLOAT4},
		}
	};

	/* textured pair */
	struct XenosShader *ps_tex = Xe_LoadShaderFromMemory(xe, glsl_ps);
	Xe_InstantiateShader(xe, ps_tex, 0);
	struct XenosShader *vs_tex = Xe_LoadShaderFromMemory(xe, glsl_vs);
	Xe_InstantiateShader(xe, vs_tex, 0);
	Xe_ShaderApplyVFetchPatches(xe, vs_tex, 0, &vbf_tex);

	/* HUD pair */
	struct XenosShader *ps_hud = Xe_LoadShaderFromMemory(xe, xost_ps0);
	Xe_InstantiateShader(xe, ps_hud, 0);
	struct XenosShader *vs_hud = Xe_LoadShaderFromMemory(xe, xost_vs1);
	Xe_InstantiateShader(xe, vs_hud, 0);
	Xe_ShaderApplyVFetchPatches(xe, vs_hud, 0, &vbf_plain);

	/* geometry */
	build_cube();
	struct XenosVertexBuffer *vb =
		Xe_CreateVertexBuffer(xe, sizeof(cube_verts));
	void *v = Xe_VB_Lock(xe, vb, 0, sizeof(cube_verts), XE_LOCK_WRITE);
	memcpy(v, cube_verts, sizeof(cube_verts));
	Xe_VB_Unlock(xe, vb);

	struct XenosIndexBuffer *ib =
		Xe_CreateIndexBuffer(xe, sizeof(cube_idx), XE_FMT_INDEX16);
	void *ip = Xe_IB_Lock(xe, ib, 0, sizeof(cube_idx), XE_LOCK_WRITE);
	memcpy(ip, cube_idx, sizeof(cube_idx));
	Xe_IB_Unlock(xe, ib);

	struct XenosVertexBuffer *hvb = Xe_CreateVertexBuffer(xe, sizeof(hud));

	/* 64x64 checkerboard texture, linear ARGB8888 */
	struct XenosSurface *tex =
		Xe_CreateTexture(xe, 64, 64, 1, XE_FMT_8888 | XE_FMT_ARGB, 0);
	unsigned char *tp = Xe_Surface_LockRect(xe, tex, 0, 0, 0, 0,
						XE_LOCK_WRITE);
	int yy, xx;
	for (yy = 0; yy < 64; yy++) {
		unsigned int *row = (unsigned int *)(tp + yy * tex->wpitch);
		for (xx = 0; xx < 64; xx++)
			row[xx] = (((xx >> 3) ^ (yy >> 3)) & 1)
				? 0xFFFFFFFF : 0xFF404040;
	}
	Xe_Surface_Unlock(xe, tex);

	edram_init(xe);
	printf("rendering...\n");
	console_close();

	const uint64_t frame_ticks = PPC_TIMEBASE_FREQ / 60;
	const float cpu_col[3] = {0.2f, 1.0f, 0.2f};
	const float gpu_col[3] = {0.2f, 0.9f, 1.0f};
	const unsigned char lbl_cpu[3] = {SEG_C, SEG_P, SEG_U};
	const unsigned char lbl_gpu[3] = {SEG_G, SEG_P, SEG_U};

	float cpu_avg = 0.0f, gpu_avg = 0.0f;
	int show_cpu = 0, show_gpu = 0, refresh = 0;
	uint64_t t0 = mftb();

	while (1)
	{
		uint64_t fstart = mftb();
		float t = (float)((double)(fstart - t0) /
				  (double)PPC_TIMEBASE_FREQ);

		Xe_InvalidateState(xe);
		Xe_SetCullMode(xe, 0);
		Xe_SetZEnable(xe, 1);
		Xe_SetZWrite(xe, 1);
		Xe_SetClearColor(xe, 0xFF202020);

		mat4 mvp, p, tr, ry, rx, ident;
		m_persp(p, 50.0f * (float)M_PI / 180.0f, 1280.0f / 720.0f,
			1.0f, 20.0f);
		m_translate_z(tr, -5.0f);
		m_rot_y(ry, t * 0.9f);
		m_rot_x(rx, t * 0.55f);
		m_mul(mvp, ry, rx);
		m_mul(mvp, tr, mvp);
		m_mul(mvp, p, mvp);
		m_identity(ident);

		hud_verts = 0;
		hud_row(-0.97f, 0.95f, lbl_cpu, 3, show_cpu, cpu_col);
		hud_row(-0.97f, 0.84f, lbl_gpu, 3, show_gpu, gpu_col);
		void *hv = Xe_VB_Lock(xe, hvb, 0, hud_verts * 32,
				      XE_LOCK_WRITE);
		memcpy(hv, hud, hud_verts * 32);
		Xe_VB_Unlock(xe, hvb);

		/* textured cube pass */
		Xe_SetVertexShaderConstantF(xe, 0, (float *)mvp, 4);
		Xe_SetShader(xe, SHADER_TYPE_PIXEL, ps_tex, 0);
		Xe_SetShader(xe, SHADER_TYPE_VERTEX, vs_tex, 0);
		Xe_SetTexture(xe, 0, tex);
		Xe_SetStreamSource(xe, 0, vb, 0, 10);
		Xe_SetIndices(xe, ib);
		Xe_DrawIndexedPrimitive(xe, XE_PRIMTYPE_TRIANGLELIST,
					0, 0, 24, 0, 12);

		/* HUD pass (untextured pair, identity transform) */
		Xe_SetVertexShaderConstantF(xe, 0, (float *)ident, 4);
		Xe_SetShader(xe, SHADER_TYPE_PIXEL, ps_hud, 0);
		Xe_SetShader(xe, SHADER_TYPE_VERTEX, vs_hud, 0);
		Xe_SetStreamSource(xe, 0, hvb, 0, 8);
		Xe_DrawPrimitive(xe, XE_PRIMTYPE_TRIANGLELIST,
				 0, hud_verts / 3);

		uint64_t t_kick = mftb();
		Xe_Resolve(xe);
		Xe_Sync(xe);
		uint64_t t_done = mftb();

		float cpu_pct = 100.0f * (float)(t_kick - fstart)
				/ (float)frame_ticks;
		float gpu_pct = 100.0f * (float)(t_done - t_kick)
				/ (float)frame_ticks;
		cpu_avg += 0.05f * (cpu_pct - cpu_avg);
		gpu_avg += 0.05f * (gpu_pct - gpu_avg);
		if (++refresh >= 20) {
			refresh = 0;
			show_cpu = (int)(cpu_avg + 0.5f);
			show_gpu = (int)(gpu_avg + 0.5f);
		}

		while (mftb() - fstart < frame_ticks)
			;
	}
	return 0;
}
