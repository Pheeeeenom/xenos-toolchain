/* xost triangle — diagnostic build.
 *
 * Renders two RGB triangles (opposite windings, culling disabled) using the
 * xost-authored shaders, and dumps the post-patch vfetch instruction words
 * so we can verify Xe_ShaderApplyVFetchPatches consumed our container's
 * patch table correctly.
 */
#include <stdio.h>
#include <string.h>
#include <xenos/xe.h>
#include <xenos/xenos.h>
#include <xenos/edram.h>
#include <console/console.h>
#include "xost_shaders.h"

extern void mdelay(int);

static void dump_slot(struct XenosShader *sh, int slot, const char *what)
{
	unsigned int *code = (unsigned int *)sh->shader_instance[0];
	printf("slot %d (%s): %08x %08x %08x\n", slot, what,
	       code[slot * 3], code[slot * 3 + 1], code[slot * 3 + 2]);
}

int main(void)
{
	xenos_init(VIDEO_MODE_AUTO);
	console_init();
	printf("xost triangle diag build 2\n");

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

	/* left triangle: CCW in clip space; right triangle: CW.
	 * culling is disabled, so BOTH should appear; if only one does,
	 * culling was the problem. */
	float tri[] = {
		/*    X       Y      Z     W       R     G     B     A  */
		-0.9f, -0.8f,  0.0f,  1.0f,    1.0f, 0.0f, 0.0f, 1.0f,
		-0.1f, -0.8f,  0.0f,  1.0f,    0.0f, 1.0f, 0.0f, 1.0f,
		-0.5f,  0.8f,  0.0f,  1.0f,    0.0f, 0.0f, 1.0f, 1.0f,

		 0.1f, -0.8f,  0.0f,  1.0f,    1.0f, 1.0f, 0.0f, 1.0f,
		 0.5f,  0.8f,  0.0f,  1.0f,    0.0f, 1.0f, 1.0f, 1.0f,
		 0.9f, -0.8f,  0.0f,  1.0f,    1.0f, 0.0f, 1.0f, 1.0f,
	};

	struct XenosShader *ps = Xe_LoadShaderFromMemory(xe, xost_ps0);
	Xe_InstantiateShader(xe, ps, 0);

	struct XenosShader *vs = Xe_LoadShaderFromMemory(xe, xost_vs0);
	Xe_InstantiateShader(xe, vs, 0);

	printf("-- vs slots BEFORE patch:\n");
	dump_slot(vs, 3, "pos vfetch");
	dump_slot(vs, 4, "col vfetch");

	Xe_ShaderApplyVFetchPatches(xe, vs, 0, &vbf);

	printf("-- vs slots AFTER patch (want fmt/stride/offset filled):\n");
	dump_slot(vs, 3, "pos vfetch");
	dump_slot(vs, 4, "col vfetch");
	dump_slot(vs, 5, "oPos export");
	dump_slot(vs, 6, "o0 export");

	struct XenosVertexBuffer *vb = Xe_CreateVertexBuffer(xe, sizeof(tri));
	void *v = Xe_VB_Lock(xe, vb, 0, sizeof(tri), XE_LOCK_WRITE);
	memcpy(v, tri, sizeof(tri));
	Xe_VB_Unlock(xe, vb);

	edram_init(xe);

	printf("photograph this screen; rendering starts in 15s\n");
	int i;
	for (i = 15; i > 0; i--) {
		printf("  %d...\n", i);
		mdelay(1000);
	}

	while (1)
	{
		Xe_InvalidateState(xe);
		Xe_SetCullMode(xe, 0); /* PA_SU_SC_MODE_CNTL cull bits = none */
		Xe_SetClearColor(xe, 0xFF202020);

		Xe_SetShader(xe, SHADER_TYPE_PIXEL, ps, 0);
		Xe_SetShader(xe, SHADER_TYPE_VERTEX, vs, 0);
		Xe_SetStreamSource(xe, 0, vb, 0, 8);
		Xe_DrawPrimitive(xe, XE_PRIMTYPE_TRIANGLELIST, 0, 2);

		Xe_Resolve(xe);
		Xe_Sync(xe);
	}
	return 0;
}
