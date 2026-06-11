/* xost triangle — the first Xbox 360 homebrew whose GPU shaders were
 * authored entirely with open tools (github: xost project, Phase 1 exit).
 *
 * Renders one RGB triangle using xost_vs0/xost_ps0 (see xost_shaders.h,
 * generated from the authored containers by gen_header.py). Structure
 * follows the libxenon cube example.
 */
#include <stdio.h>
#include <string.h>
#include <xenos/xe.h>
#include <xenos/xenos.h>
#include <xenos/edram.h>
#include <console/console.h>
#include "xost_shaders.h"

int main(void)
{
	xenos_init(VIDEO_MODE_AUTO);
	console_init();
	printf("xost triangle: fully open-toolchain shaders\n");

	struct XenosDevice _xe, *xe = &_xe;
	Xe_Init(xe);

	struct XenosSurface *fb = Xe_GetFramebufferSurface(xe);
	Xe_SetRenderTarget(xe, fb);

	/* must match the VS container's VertexElements:
	 * POSITION0 (template vfetch slot 3), COLOR0 (slot 4) */
	static const struct XenosVBFFormat vbf =
	{
		2, {
			{XE_USAGE_POSITION, 0, XE_TYPE_FLOAT4},
			{XE_USAGE_COLOR,    0, XE_TYPE_FLOAT4},
		}
	};

	/* clip-space positions (the VS is a passthrough) + RGB corners */
	float tri[] = {
		/*    X       Y      Z     W       R     G     B     A  */
		-0.8f, -0.8f,  0.0f,  1.0f,    1.0f, 0.0f, 0.0f, 1.0f,
		 0.8f, -0.8f,  0.0f,  1.0f,    0.0f, 1.0f, 0.0f, 1.0f,
		 0.0f,  0.8f,  0.0f,  1.0f,    0.0f, 0.0f, 1.0f, 1.0f,
	};

	printf("loading xost shaders (vs %u B, ps %u B)...\n",
	       (unsigned)sizeof(xost_vs0), (unsigned)sizeof(xost_ps0));
	struct XenosShader *ps = Xe_LoadShaderFromMemory(xe, xost_ps0);
	Xe_InstantiateShader(xe, ps, 0);

	struct XenosShader *vs = Xe_LoadShaderFromMemory(xe, xost_vs0);
	Xe_InstantiateShader(xe, vs, 0);
	Xe_ShaderApplyVFetchPatches(xe, vs, 0, &vbf);

	struct XenosVertexBuffer *vb = Xe_CreateVertexBuffer(xe, sizeof(tri));
	void *v = Xe_VB_Lock(xe, vb, 0, sizeof(tri), XE_LOCK_WRITE);
	memcpy(v, tri, sizeof(tri));
	Xe_VB_Unlock(xe, vb);

	edram_init(xe);
	printf("rendering. if you can read this, it worked on console.\n");
	console_close();

	while (1)
	{
		Xe_InvalidateState(xe);
		Xe_SetClearColor(xe, 0xFF202020);

		Xe_SetShader(xe, SHADER_TYPE_PIXEL, ps, 0);
		Xe_SetShader(xe, SHADER_TYPE_VERTEX, vs, 0);
		Xe_SetStreamSource(xe, 0, vb, 0, 8); /* 8 floats per vertex */
		Xe_DrawPrimitive(xe, XE_PRIMTYPE_TRIANGLELIST, 0, 1);

		Xe_Resolve(xe);
		Xe_Sync(xe);
	}
	return 0;
}
