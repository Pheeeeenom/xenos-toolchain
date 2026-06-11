/* xost spincube — a rotating 3D cube drawn with fully open-toolchain
 * shaders (xost_vs1: pos * mvp from c0-c3; xost_ps0: color passthrough).
 *
 * VS contract: oPos[i] = dot4(position, c[i]) — we upload the rows of the
 * column-vector MVP matrix as constants c0..c3.
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

/* right-handed, camera looks down -Z, clip z in [0,1] */
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

/* 8 corners, rainbow colors; 12 triangles. depth handles visibility. */
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
	0, 1, 2,  0, 2, 3,   /* -z */
	4, 6, 5,  4, 7, 6,   /* +z */
	0, 4, 5,  0, 5, 1,   /* -y */
	3, 2, 6,  3, 6, 7,   /* +y */
	0, 3, 7,  0, 7, 4,   /* -x */
	1, 5, 6,  1, 6, 2,   /* +x */
};

int main(void)
{
	xenos_init(VIDEO_MODE_AUTO);
	console_init();
	printf("xost spincube: open-toolchain 3D, now with the third D\n");

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
	void *i = Xe_IB_Lock(xe, ib, 0, sizeof(indices), XE_LOCK_WRITE);
	memcpy(i, indices, sizeof(indices));
	Xe_IB_Unlock(xe, ib);

	edram_init(xe);
	printf("rendering...\n");
	console_close();

	uint64_t t0 = mftb();
	while (1)
	{
		/* wall-clock seconds since start — frame-rate independent */
		float t = (float)((double)(mftb() - t0) /
				  (double)PPC_TIMEBASE_FREQ);
		Xe_InvalidateState(xe);
		Xe_SetCullMode(xe, 0);   /* no culling; depth sorts it out */
		Xe_SetZEnable(xe, 1);
		Xe_SetZWrite(xe, 1);
		Xe_SetClearColor(xe, 0xFF202020);

		mat4 mvp, p, tr, ry, rx;
		m_persp(p, 50.0f * (float)M_PI / 180.0f, 1280.0f / 720.0f,
			1.0f, 20.0f);
		m_translate_z(tr, -5.0f);
		m_rot_y(ry, t * 0.9f);    /* ~7 s per revolution */
		m_rot_x(rx, t * 0.55f);
		m_mul(mvp, ry, rx);   /* model  = Ry * Rx   */
		m_mul(mvp, tr, mvp);  /* view   = T * model */
		m_mul(mvp, p, mvp);   /* clip   = P * view  */
		Xe_SetVertexShaderConstantF(xe, 0, (float *)mvp, 4);

		Xe_SetShader(xe, SHADER_TYPE_PIXEL, ps, 0);
		Xe_SetShader(xe, SHADER_TYPE_VERTEX, vs, 0);
		Xe_SetStreamSource(xe, 0, vb, 0, 8);
		Xe_SetIndices(xe, ib);
		Xe_DrawIndexedPrimitive(xe, XE_PRIMTYPE_TRIANGLELIST,
					0, 0, 8, 0, 12);

		Xe_Resolve(xe);
		Xe_Sync(xe);
	}
	return 0;
}
