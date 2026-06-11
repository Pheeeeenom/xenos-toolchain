/* xost spincube — rotating 3D cube + live CPU/GPU usage HUD, all drawn
 * with fully open-toolchain shaders (xost_vs1 mvp transform + xost_ps0).
 *
 * The HUD is a 7-segment vector font built from colored quads — no
 * textures, no console: the same two shaders draw everything. For the HUD
 * pass we simply upload an identity matrix to c0-c3.
 *
 * Metrics (bare metal, measured via the PPC timebase):
 *   CPU% = time spent preparing the frame / 60fps frame budget
 *   GPU% = time from Resolve kick to GPU idle / 60fps frame budget
 * Both are approximations in the honest direction: there is no OS here,
 * so "usage" means "fraction of the frame budget actually busy".
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

/* ---------------- matrices ---------------- */

typedef float mat4[4][4]; /* [row][col], clip = M * column-vector */

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

/* ---------------- cube ---------------- */

static const float verts[] = {
	/*   X      Y      Z     W      R     G     B     A  */
	-1.0f, -1.0f, -1.0f, 1.0f,  0.0f, 0.0f, 0.0f, 1.0f,
	 1.0f, -1.0f, -1.0f, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
	 1.0f,  1.0f, -1.0f, 1.0f,  1.0f, 1.0f, 0.0f, 1.0f,
	-1.0f,  1.0f, -1.0f, 1.0f,  0.0f, 1.0f, 0.0f, 1.0f,
	-1.0f, -1.0f,  1.0f, 1.0f,  0.0f, 0.0f, 1.0f, 1.0f,
	 1.0f, -1.0f,  1.0f, 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
	 1.0f,  1.0f,  1.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f,
	-1.0f,  1.0f,  1.0f, 1.0f,  0.0f, 1.0f, 1.0f, 1.0f,
};

static const unsigned short indices[] = {
	0, 1, 2,  0, 2, 3,
	4, 6, 5,  4, 7, 6,
	0, 4, 5,  0, 5, 1,
	3, 2, 6,  3, 6, 7,
	0, 3, 7,  0, 7, 4,
	1, 5, 6,  1, 6, 2,
};

/* ---------------- 7-segment HUD ---------------- */

#define HUD_MAX_VERTS 4096
static float hud[HUD_MAX_VERTS * 8];
static int hud_verts;

static void hud_rect(float x, float y, float w, float h, const float *col)
{
	/* (x,y) = top-left in clip space (y up); emits 2 triangles */
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

/* segment bits: a=1 b=2 c=4 d=8 e=16 f=32 g=64 */
static const unsigned char seg_digit[10] = {
	0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F
};
#define SEG_C 0x39
#define SEG_P 0x73
#define SEG_U 0x3E
#define SEG_G 0x3D
#define SEG_PCT 0x49  /* a+g+d: a fake but readable '%' stand-in (≡) */

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
	if (value >= 100) { hud_char(x, y, cw, ch,
				     seg_digit[(value / 100) % 10], col); }
	x += adv;
	if (value >= 10) { hud_char(x, y, cw, ch,
				    seg_digit[(value / 10) % 10], col); }
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
	printf("xost spincube + perf HUD\n");

	struct XenosDevice _xe, *xe = &_xe;
	Xe_Init(xe);

	struct XenosSurface *fb = Xe_GetFramebufferSurface(xe);
	Xe_SetRenderTarget(xe, fb);

	static const struct XenosVBFFormat vbf =
	{
		2, {
			{XE_USAGE_POSITION, 0, XE_TYPE_FLOAT4},
			{XE_USAGE_COLOR,    0, XE_TYPE_FLOAT4},
		}
	};

	struct XenosShader *ps = Xe_LoadShaderFromMemory(xe, xost_ps0);
	Xe_InstantiateShader(xe, ps, 0);
	struct XenosShader *vs = Xe_LoadShaderFromMemory(xe, xost_vs1);
	Xe_InstantiateShader(xe, vs, 0);
	Xe_ShaderApplyVFetchPatches(xe, vs, 0, &vbf);

	struct XenosVertexBuffer *vb = Xe_CreateVertexBuffer(xe, sizeof(verts));
	void *v = Xe_VB_Lock(xe, vb, 0, sizeof(verts), XE_LOCK_WRITE);
	memcpy(v, verts, sizeof(verts));
	Xe_VB_Unlock(xe, vb);

	struct XenosIndexBuffer *ib =
		Xe_CreateIndexBuffer(xe, sizeof(indices), XE_FMT_INDEX16);
	void *ip = Xe_IB_Lock(xe, ib, 0, sizeof(indices), XE_LOCK_WRITE);
	memcpy(ip, indices, sizeof(indices));
	Xe_IB_Unlock(xe, ib);

	struct XenosVertexBuffer *hvb =
		Xe_CreateVertexBuffer(xe, sizeof(hud));

	edram_init(xe);
	printf("rendering...\n");
	console_close();

	const uint64_t frame_ticks = PPC_TIMEBASE_FREQ / 60; /* 60 fps cap */
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

		/* ---- CPU phase: matrices + HUD geometry + uploads ---- */
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

		/* cube pass */
		Xe_SetVertexShaderConstantF(xe, 0, (float *)mvp, 4);
		Xe_SetShader(xe, SHADER_TYPE_PIXEL, ps, 0);
		Xe_SetShader(xe, SHADER_TYPE_VERTEX, vs, 0);
		Xe_SetStreamSource(xe, 0, vb, 0, 8);
		Xe_SetIndices(xe, ib);
		Xe_DrawIndexedPrimitive(xe, XE_PRIMTYPE_TRIANGLELIST,
					0, 0, 8, 0, 12);

		/* HUD pass: identity transform, same shaders */
		Xe_SetVertexShaderConstantF(xe, 0, (float *)ident, 4);
		Xe_SetStreamSource(xe, 0, hvb, 0, 8);
		Xe_DrawPrimitive(xe, XE_PRIMTYPE_TRIANGLELIST,
				 0, hud_verts / 3);

		/* ---- GPU phase: kick + drain ---- */
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
		if (++refresh >= 20) { /* update display ~3x/s */
			refresh = 0;
			show_cpu = (int)(cpu_avg + 0.5f);
			show_gpu = (int)(gpu_avg + 0.5f);
		}

		/* 60 fps cap (busy-wait; not counted as usage) */
		while (mftb() - fstart < frame_ticks)
			;
	}
	return 0;
}
