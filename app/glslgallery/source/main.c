/* xost glslgallery — a spinning textured cube cycling through a gallery of
 * fragment/vertex shaders WRITTEN IN GLSL and compiled to Xenos microcode by xsc.
 * The default vertex shader is GLSL too; late gallery slots swap in vertex
 * frontend proof shaders. Each shader's literal-constant manifest is
 * uploaded before drawing — this is the on-hardware proof of the whole
 * GLSL frontend (dot/dp3, saturate, literal pool, swizzles).
 *
 *   G0 MODULATE  tex*color     G3 GLOW   clamp(tex+color,0,1)
 *   G1 TINT      rgb tint      G4 BLEND  mix(tex,color,0.5)
 *   G2 GRAY      desaturate    G23 VSCHAIN  chained mat4*vec4
 *                              G24 VSARITH  VS add/mul values
 *                              G25 LOOPHW   real HW loop
 *                              G26 DISCARD  kill half the pixels
 *                              G27 CUBEMAP  cube fetch proof
 *                              G28 GLSLLOOP uniform-bound GLSL loop
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <xenos/xe.h>
#include <xenos/xenos.h>
#include <xenos/edram.h>
#include <console/console.h>
#include <ppc/timebase.h>
#include <usb/usbmain.h>
#include <input/input.h>

typedef struct { int reg; float v[4]; } gallery_const;
typedef struct {
	const char *name;
	unsigned char *code;
	unsigned char *vscode;
	int vs_mode;
	int n_const;
	const gallery_const *consts;
	int loop_id;
} gallery_entry;

enum {
	VS_MODE_MVP = 0,
	VS_MODE_CHAIN = 1,
	VS_MODE_ARITH = 2
};

#include "xost_shaders.h"
#include "texture_data.h"

extern void mdelay(int);

/* ---- matrices ---- */
typedef float mat4[4][4];
static void m_identity(mat4 m) { memset(m, 0, sizeof(mat4));
	m[0][0]=m[1][1]=m[2][2]=m[3][3]=1.0f; }
static void m_mul(mat4 o, const mat4 a, const mat4 b) { int r,c,k; mat4 t;
	for(r=0;r<4;r++)for(c=0;c<4;c++){float s=0;for(k=0;k<4;k++)s+=a[r][k]*b[k][c];t[r][c]=s;}
	memcpy(o,t,sizeof(mat4)); }
static void m_persp(mat4 m,float fy,float as,float n,float f){float ct=1.0f/tanf(fy*0.5f);
	memset(m,0,sizeof(mat4));m[0][0]=ct/as;m[1][1]=ct;m[2][2]=f/(n-f);m[2][3]=n*f/(n-f);m[3][2]=-1.0f;}
static void m_roty(mat4 m,float a){m_identity(m);m[0][0]=cosf(a);m[0][2]=sinf(a);m[2][0]=-sinf(a);m[2][2]=cosf(a);}
static void m_rotx(mat4 m,float a){m_identity(m);m[1][1]=cosf(a);m[1][2]=-sinf(a);m[2][1]=sinf(a);m[2][2]=cosf(a);}
static void m_tz(mat4 m,float z){m_identity(m);m[2][3]=z;}

/* ---- cube: 24 verts (per-face uv), pos/color/uv ---- */
static const float corners[8][3]={{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
	{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};
static const int faces[6][4]={{0,1,2,3},{5,4,7,6},{4,0,3,7},{1,5,6,2},{3,2,6,7},{4,5,1,0}};
static const float fcol[6][3]={{1,.6f,.6f},{.6f,1,.6f},{.6f,.6f,1},{1,1,.6f},{.6f,1,1},{1,.6f,1}};
static const float fuv[4][2]={{0,0},{1,0},{1,1},{0,1}};
static float cube[24*10];
static unsigned short cidx[36];
static void build_cube(void){int f,v;for(f=0;f<6;f++){for(v=0;v<4;v++){
	float*o=&cube[(f*4+v)*10];const float*p=corners[faces[f][v]];
	o[0]=p[0];o[1]=p[1];o[2]=p[2];o[3]=1;o[4]=fcol[f][0];o[5]=fcol[f][1];o[6]=fcol[f][2];o[7]=1;
	o[8]=fuv[v][0];o[9]=fuv[v][1];}
	unsigned short b=f*4;unsigned short*x=&cidx[f*6];x[0]=b;x[1]=b+1;x[2]=b+2;x[3]=b;x[4]=b+2;x[5]=b+3;}}

/* ---- HUD 7-seg ---- */
#define HUDV 4096
static float hud[HUDV*8]; static int hudn;
static void hrect(float x,float y,float w,float h,const float*c){if(hudn+6>HUDV)return;
	float x1=x+w,y1=y-h;float q[6][2]={{x,y},{x1,y},{x1,y1},{x,y},{x1,y1},{x,y1}};int i;
	for(i=0;i<6;i++){float*v=&hud[hudn++*8];v[0]=q[i][0];v[1]=q[i][1];v[2]=0;v[3]=1;
	v[4]=c[0];v[5]=c[1];v[6]=c[2];v[7]=1;}}
static const unsigned char seg[10]={0x3F,0x06,0x5B,0x4F,0x66,0x6D,0x7D,0x07,0x7F,0x6F};
#define S_C 0x39
#define S_P 0x73
#define S_U 0x3E
#define S_G 0x3D
#define S_PCT 0x49
static void hchar(float cx,float cy,float cw,float ch,unsigned char m,const float*c){
	float th=cw*0.22f,hf=ch*0.5f;
	if(m&0x01)hrect(cx+th,cy,cw-2*th,th,c);
	if(m&0x02)hrect(cx+cw-th,cy-th,th,hf-th,c);
	if(m&0x04)hrect(cx+cw-th,cy-hf-th*0.5f,th,hf-th*0.5f,c);
	if(m&0x08)hrect(cx+th,cy-ch+th,cw-2*th,th,c);
	if(m&0x10)hrect(cx,cy-hf-th*0.5f,th,hf-th*0.5f,c);
	if(m&0x20)hrect(cx,cy-th,th,hf-th,c);
	if(m&0x40)hrect(cx+th,cy-hf+th*0.5f,cw-2*th,th,c);}
static void hrow(float x,float y,const unsigned char*lbl,int nl,int val,int pct,const float*c){
	float cw=0.030f,ch=0.075f,adv=cw*1.45f;int i;
	for(i=0;i<nl;i++){hchar(x,y,cw,ch,lbl[i],c);x+=adv;}
	x+=adv*0.5f;if(val<0)val=0;if(val>999)val=999;
	if(val>=100)hchar(x,y,cw,ch,seg[(val/100)%10],c);x+=adv;
	if(val>=10)hchar(x,y,cw,ch,seg[(val/10)%10],c);x+=adv;
	hchar(x,y,cw,ch,seg[val%10],c);if(pct){x+=adv;hchar(x,y,cw,ch,S_PCT,c);}}

int main(void)
{
	xenos_init(VIDEO_MODE_AUTO);
	console_init();
	printf("xost glslgallery: %d GLSL shaders compiled by xsc\n", GALLERY_N);

	struct XenosDevice _xe,*xe=&_xe; Xe_Init(xe);
	usb_init(); usb_do_poll();   /* controller: manual slot select (see loop) */
	struct XenosSurface *fb=Xe_GetFramebufferSurface(xe); Xe_SetRenderTarget(xe,fb);

	static const struct XenosVBFFormat vbf_tex={3,{
		{XE_USAGE_POSITION,0,XE_TYPE_FLOAT4},
		{XE_USAGE_COLOR,0,XE_TYPE_FLOAT4},
		{XE_USAGE_TEXCOORD,0,XE_TYPE_FLOAT2}}};
	static const struct XenosVBFFormat vbf_plain={2,{
		{XE_USAGE_POSITION,0,XE_TYPE_FLOAT4},{XE_USAGE_COLOR,0,XE_TYPE_FLOAT4}}};

	struct XenosShader *gps[GALLERY_N],*gvs[GALLERY_N]; int i;
	for(i=0;i<GALLERY_N;i++){gps[i]=Xe_LoadShaderFromMemory(xe,gallery[i].code);
		Xe_InstantiateShader(xe,gps[i],0);
		gvs[i]=Xe_LoadShaderFromMemory(xe,gallery[i].vscode);
		Xe_InstantiateShader(xe,gvs[i],0);
		Xe_ShaderApplyVFetchPatches(xe,gvs[i],0,&vbf_tex);}

	struct XenosShader *ps_hud=Xe_LoadShaderFromMemory(xe,xost_ps0);Xe_InstantiateShader(xe,ps_hud,0);
	struct XenosShader *vs_hud=Xe_LoadShaderFromMemory(xe,xost_vs1);Xe_InstantiateShader(xe,vs_hud,0);
	Xe_ShaderApplyVFetchPatches(xe,vs_hud,0,&vbf_plain);

	build_cube();
	struct XenosVertexBuffer *vb=Xe_CreateVertexBuffer(xe,sizeof(cube));
	void*v=Xe_VB_Lock(xe,vb,0,sizeof(cube),XE_LOCK_WRITE);memcpy(v,cube,sizeof(cube));Xe_VB_Unlock(xe,vb);
	struct XenosIndexBuffer *ib=Xe_CreateIndexBuffer(xe,sizeof(cidx),XE_FMT_INDEX16);
	void*ip=Xe_IB_Lock(xe,ib,0,sizeof(cidx),XE_LOCK_WRITE);memcpy(ip,cidx,sizeof(cidx));Xe_IB_Unlock(xe,ib);
	struct XenosVertexBuffer *hvb=Xe_CreateVertexBuffer(xe,sizeof(hud));

	struct XenosSurface *tex=Xe_CreateTexture(xe,GALLERY_TEX_SIZE,
		GALLERY_TEX_SIZE,1,XE_FMT_8888|XE_FMT_ARGB,0);
	unsigned char *tp=Xe_Surface_LockRect(xe,tex,0,0,0,0,XE_LOCK_WRITE);int yy,xx;
	for(yy=0;yy<GALLERY_TEX_SIZE;yy++){unsigned int*row=(unsigned int*)(tp+yy*tex->wpitch);
		for(xx=0;xx<GALLERY_TEX_SIZE;xx++)row[xx]=gallery_tex[yy*GALLERY_TEX_SIZE+xx];}
	Xe_Surface_Unlock(xe,tex);
	struct XenosSurface *cubetex=Xe_CreateTexture(xe,64,64*6,1,
		XE_FMT_8888|XE_FMT_ARGB,0);
	unsigned char *cpix=Xe_Surface_LockRect(xe,cubetex,0,0,0,0,XE_LOCK_WRITE);
	unsigned int face_col[6]={0xFFFF4040,0xFF40FF40,0xFF4040FF,
		0xFFFFFF40,0xFF40FFFF,0xFFFF40FF};
	int face;
	for(face=0;face<6;face++)for(yy=0;yy<64;yy++){
		unsigned int*row=(unsigned int*)(cpix+(face*64+yy)*cubetex->wpitch);
		for(xx=0;xx<64;xx++)row[xx]=(((xx>>3)^(yy>>3))&1)?face_col[face]:0xFF202020;}
	Xe_Surface_Unlock(xe,cubetex);
	cubetex->height=64;cubetex->hpitch=64;

	edram_init(xe);
	printf("rendering. cube cycles GLSL shaders every 4s.\n");
	console_close();

	const uint64_t ft=PPC_TIMEBASE_FREQ/60;
	const float cc[3]={0.2f,1,0.2f},gc[3]={0.2f,0.9f,1},nc[3]={1,0.8f,0.2f};
	const unsigned char lc[3]={S_C,S_P,S_U},lg[3]={S_G,S_P,S_U},ln[1]={S_G};
	float ca=0,ga=0;int sc=0,sg=0,rf=0;uint64_t t0=mftb();
	/* controller: D-pad/bumpers step slots manually; START resumes auto-cycle */
	int sel=0,manual=0,pl=0,pr=0,plb=0,prb=0,pst=0;
	struct controller_data_s pad;

	while(1){
		uint64_t fs=mftb();float t=(float)((double)(fs-t0)/(double)PPC_TIMEBASE_FREQ);
		usb_do_poll();
		if(get_controller_data(&pad,0)){
			int nx=(pad.right||pad.rb)&&!(pr||prb);
			int pv=(pad.left||pad.lb)&&!(pl||plb);
			if(nx){manual=1;sel=(sel+1)%GALLERY_N;}
			if(pv){manual=1;sel=(sel+GALLERY_N-1)%GALLERY_N;}
			if(pad.start&&!pst)manual=0;          /* resume auto-cycle */
			pl=pad.left;pr=pad.right;plb=pad.lb;prb=pad.rb;pst=pad.start;
		}
		int mode=manual?sel:((int)(t/4.0f))%GALLERY_N;

		Xe_InvalidateState(xe);Xe_SetCullMode(xe,0);Xe_SetZEnable(xe,1);Xe_SetZWrite(xe,1);
		Xe_SetClearColor(xe,0xFF202020);

		mat4 mvp,model,viewproj,pr,tr,ry,rx,id;
		m_persp(pr,50.0f*(float)M_PI/180.0f,1280.0f/720.0f,1,20);
		m_tz(tr,-5);m_roty(ry,t*0.9f);m_rotx(rx,t*0.55f);
		m_mul(model,ry,rx);m_mul(viewproj,pr,tr);
		m_mul(mvp,tr,model);m_mul(mvp,pr,mvp);m_identity(id);

		hudn=0;hrow(-0.97f,0.95f,lc,3,sc,1,cc);hrow(-0.97f,0.84f,lg,3,sg,1,gc);
		hrow(-0.97f,0.73f,ln,1,mode,0,nc);
		void*hv=Xe_VB_Lock(xe,hvb,0,hudn*32,XE_LOCK_WRITE);memcpy(hv,hud,hudn*32);Xe_VB_Unlock(xe,hvb);

		/* cube with the active GLSL shader; upload its literal manifest */
		if(gallery[mode].vs_mode==VS_MODE_CHAIN){
			Xe_SetVertexShaderConstantF(xe,0,(float*)model,4);
			Xe_SetVertexShaderConstantF(xe,4,(float*)viewproj,4);
		}else{
			Xe_SetVertexShaderConstantF(xe,0,(float*)mvp,4);
			if(gallery[mode].vs_mode==VS_MODE_ARITH){
				float off[4]={0.35f*sinf(t*1.7f),0.25f*cosf(t*1.3f),0,0};
				float tint[4]={1.0f,0.35f+0.65f*(0.5f+0.5f*sinf(t*2.1f)),0.55f,1.0f};
				Xe_SetVertexShaderConstantF(xe,4,off,1);
				Xe_SetVertexShaderConstantF(xe,5,tint,1);
			}
		}
		for(i=0;i<gallery[mode].n_const;i++)
			Xe_SetPixelShaderConstantF(xe,gallery[mode].consts[i].reg,
				(float*)gallery[mode].consts[i].v,1);
		/* Real-loop proof slots upload loop constant i#:
		   count|start<<8|step<<16. Brightness steps with count. */
		if(gallery[mode].loop_id>=0){
			/* ALCONST reads c[0+aL] for aL=0..3, so it needs exactly 4 passes.
			   WHILEBRK relies on a predicated break, so its count is only the
			   upper bound (16) -- the predicate should stop it at pass 4. The
			   other loop slots animate brightness with the trip count. */
			int lcount;
			if(!strcmp(gallery[mode].name,"ALCONST")) lcount=4;
			else if(!strcmp(gallery[mode].name,"LOOPARR")) lcount=4;
			else if(!strcmp(gallery[mode].name,"WHILEBRK")) lcount=16;
			else lcount=1+((int)t)%6;
			int lid=gallery[mode].loop_id;
			xe->integer_constants[8+lid]=(lcount&0xff)|(0<<8)|(1<<16);
			xe->dirty|=DIRTY_INTEGER;
		}
		Xe_SetShader(xe,SHADER_TYPE_PIXEL,gps[mode],0);
		Xe_SetShader(xe,SHADER_TYPE_VERTEX,gvs[mode],0);
		Xe_SetTexture(xe,0,!strcmp(gallery[mode].name,"CUBEMAP")?cubetex:tex);
		if(!strcmp(gallery[mode].name,"CUBEMAP")){
			/* libxenon only writes a 2D fetch constant; promote tex unit 0 to a
			   cubemap in place: fetch dword5 dimension 2D(1)->Cube(3) (set bit10),
			   dword2 stack_depth=5 (6 faces). The 64x384 alloc already lays the 6
			   faces out at the 16384-byte cube slice stride. DIRTY_FETCH is set by
			   Xe_SetTexture, so the draw uploads these. */
			xe->fetch_constants[5]|=(1u<<10);
			xe->fetch_constants[2]|=(5u<<26);
		}
		Xe_SetStreamSource(xe,0,vb,0,10);Xe_SetIndices(xe,ib);
		Xe_DrawIndexedPrimitive(xe,XE_PRIMTYPE_TRIANGLELIST,0,0,24,0,12);

		/* HUD */
		Xe_SetVertexShaderConstantF(xe,0,(float*)id,4);
		Xe_SetShader(xe,SHADER_TYPE_PIXEL,ps_hud,0);Xe_SetShader(xe,SHADER_TYPE_VERTEX,vs_hud,0);
		Xe_SetStreamSource(xe,0,hvb,0,8);Xe_DrawPrimitive(xe,XE_PRIMTYPE_TRIANGLELIST,0,hudn/3);

		uint64_t tk=mftb();Xe_Resolve(xe);Xe_Sync(xe);uint64_t td=mftb();
		float cp=100.0f*(float)(tk-fs)/(float)ft,gp=100.0f*(float)(td-tk)/(float)ft;
		ca+=0.05f*(cp-ca);ga+=0.05f*(gp-ga);
		if(++rf>=20){rf=0;sc=(int)(ca+0.5f);sg=(int)(ga+0.5f);}
		while(mftb()-fs<ft);
	}
	return 0;
}
