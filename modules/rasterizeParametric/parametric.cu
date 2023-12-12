#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

#include "math.cuh"

constexpr int MAX_PATCHES = 1'000'000;

namespace cg = cooperative_groups;

// constexpr float uniformTime = 0.0;
Uniforms uniforms;
Allocator* allocator;

constexpr float PI = 3.1415;
constexpr uint32_t BACKGROUND_COLOR = 0x00332211ull;

constexpr float sh_coefficients[91] = {
	-0.2739740312099,  0.2526670396328,  1.8922271728516,  0.2878578901291, -0.5339795947075, -0.2620058953762,
	 0.1580424904823,  0.0329004973173, -0.1322413831949, -0.1332057565451,  1.0894461870193, -0.6319401264191, -0.0416776277125, -1.0772529840469,  0.1423762738705,
	 0.7941166162491,  0.7490307092667, -0.3428381681442,  0.1024847552180, -0.0219132602215,  0.0499043911695,  0.2162453681231,  0.0921059995890, -0.2611238956451,  0.2549301385880, -0.4534865319729,  0.1922748684883, -0.6200597286224,
	-0.0532187558711, -0.3569841980934,  0.0293972902000, -0.1977960765362, -0.1058669015765,  0.2372217923403, -0.1856198310852, -0.3373193442822, -0.0750469490886,  0.2146576642990, -0.0490148440003,  0.1288588196039,  0.3173974752426,  0.1990085393190, -0.1736343950033, -0.0482443645597, 0.1749017387629,
	-0.0151847425660,  0.0418366046081,  0.0863263587216, -0.0649211244490,  0.0126096132283,  0.0545089217982, -0.0275142164626,  0.0399986574832, -0.0468244261610, -0.1292105653111, -0.0786858322658, -0.0663828464882,  0.0382439706831, -0.0041550330365, -0.0502800566338, -0.0732471630735, 0.0181751900972, -0.0090119333757, -0.0604443282359, -0.1469985252752, -0.0534046899715,
	-0.0896672753415, -0.0130841364808, -0.0112942893801,  0.0272257498541,  0.0626717616331, -0.0222197983050, -0.0018541504308, -0.1653251944056,  0.0409697402846,  0.0749921454327, -0.0282830872616,  0.0006909458525,  0.0625599842287,  0.0812529816082,  0.0914693020772, -0.1197222726745, 0.0376277453183, -0.0832617004142, -0.0482175038043, -0.0839003635737, -0.0349423908400, 0.1204519568256, 0.0783745984003, 0.0297401205976, -0.0505947662525
};

struct Model{
	int functionID;
	float3 position;
};

struct Patch{
	float s_min;
	float s_max;
	float t_min;
	float t_max;
	int modelID;
};

float4 operator*(const mat4& a, const float4& b){
	return make_float4(
		dot(a.rows[0], b),
		dot(a.rows[1], b),
		dot(a.rows[2], b),
		dot(a.rows[3], b)
	);
}

mat4 operator*(const mat4& a, const mat4& b){
	
	mat4 result;

	result.rows[0].x = dot(a.rows[0], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
	result.rows[0].y = dot(a.rows[0], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
	result.rows[0].z = dot(a.rows[0], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
	result.rows[0].w = dot(a.rows[0], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

	result.rows[1].x = dot(a.rows[1], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
	result.rows[1].y = dot(a.rows[1], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
	result.rows[1].z = dot(a.rows[1], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
	result.rows[1].w = dot(a.rows[1], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

	result.rows[2].x = dot(a.rows[2], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
	result.rows[2].y = dot(a.rows[2], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
	result.rows[2].z = dot(a.rows[2], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
	result.rows[2].w = dot(a.rows[2], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

	result.rows[3].x = dot(a.rows[3], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
	result.rows[3].y = dot(a.rows[3], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
	result.rows[3].z = dot(a.rows[3], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
	result.rows[3].w = dot(a.rows[3], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

	return result;
}

void drawPoint(float4 coord, uint64_t* framebuffer, uint32_t color, Uniforms& uniforms){

	int x = coord.x;
	int y = coord.y;

	if(x > 1 && x < uniforms.width  - 2.0f)
	if(y > 1 && y < uniforms.height - 2.0f){

		// SINGLE PIXEL
		uint32_t pixelID = x + int(uniforms.width) * y;
		uint64_t udepth = *((uint32_t*)&coord.w);
		uint64_t encoded = (udepth << 32) | color;

		atomicMin(&framebuffer[pixelID], encoded);
	}
}

void drawSprite(float4 coord, uint64_t* framebuffer, uint32_t color, Uniforms& uniforms){

	int x = coord.x;
	int y = coord.y;

	if(x > 1 && x < uniforms.width  - 2.0f)
	if(y > 1 && y < uniforms.height - 2.0f){

		// POINT SPRITE
		for(int ox : {-2, -1, 0, 1, 2})
		for(int oy : {-2, -1, 0, 1, 2}){
			uint32_t pixelID = (x + ox) + int(uniforms.width) * (y + oy);
			uint64_t udepth = *((uint32_t*)&coord.w);
			uint64_t encoded = (udepth << 32) | color;

			atomicMin(&framebuffer[pixelID], encoded);
		}
	}
}

auto toScreen = [&](float3 p, Uniforms& uniforms){
	float4 ndc = uniforms.transform * float4{p.x, p.y, p.z, 1.0f};

	ndc.x = ndc.x / ndc.w;
	ndc.y = ndc.y / ndc.w;
	ndc.z = ndc.z / ndc.w;

	ndc.x = (ndc.x * 0.5f + 0.5f) * uniforms.width;
	ndc.y = (ndc.y * 0.5f + 0.5f) * uniforms.height;

	return ndc;
};

auto toScreen_locked = [&](float3 p, Uniforms& uniforms){
	float4 ndc = uniforms.locked_transform * float4{p.x, p.y, p.z, 1.0f};

	ndc.x = ndc.x / ndc.w;
	ndc.y = ndc.y / ndc.w;
	ndc.z = ndc.z / ndc.w;

	ndc.x = (ndc.x * 0.5f + 0.5f) * uniforms.width;
	ndc.y = (ndc.y * 0.5f + 0.5f) * uniforms.height;

	return ndc;
};

// s, t in range 0 to 1!
float3 sampleSphere(float s, float t){

	float u = 2.0f * 3.14f * s;
	float v = 3.14f * t;
	
	float3 xyz = {
		cos(u) * sin(v),
		sin(u) * sin(v),
		cos(v)
	};

	return xyz;
};

// s, t in range 0 to 1!
float3 samplePlane(float s, float t){
	return float3{2.0f * s - 1.0f, 0.0f, 2.0f * t - 1.0f};
};

float3 sampleFunkyPlane(float s, float t){

	float scale = 10.0f;
	float height = 0.105f;

	float time = uniforms.time;
	// float time = 123.0;
	float su = s - 0.5f;
	float tu = t - 0.5f;
	// float su = 1.0;
	// float tu = 1.0;
	float d = (su * su + tu * tu);

	// NOTE: It's very important for perf to explicitly specify float literals (e.g. 2.0f)
	float z = height * sin(scale * s + time) * cos(scale * t + time) 
	          + cos(2.0f * time) * 10.0f * height * exp(-1000.0f * d);

	return float3{
		2.0f * (-s + 0.5f), 
		z, 
		2.0f * (-t + 0.5f)
	};
};

float3 sampleExtraFunkyPlane(float s, float t){

	float scale = 10.0f;
	float height = 0.105f;

	float time = uniforms.time;
	// float time = 123.0;
	float su = s - 0.5f;
	float tu = t - 0.5f;
	// float su = 1.0;
	// float tu = 1.0;
	float d = (su * su + tu * tu);

	// NOTE: It's very important for perf to explicitly specify float literals (e.g. 2.0f)
	float z = height * sin(scale * s + time) * cos(scale * t + time) 
	    + cos(2.0f * time) * 10.0f * height * exp(-1000.0f * d)
	    + 0.002f * sin(2.0f * PI * 300.0f * s) * cos(2.0f * PI * 300.0f * t);

	return float3{
		2.0f * (-s + 0.5f), 
		z, 
		2.0f * (-t + 0.5f)
	};
};


// +------------------------------------------------------------------------------+
// |   Helper functions for Spherical Harmonics evaluation BEGIN                  |
// +------------------------------------------------------------------------------+

float sinc(float x) /* Supporting sinc function */
{
	if (abs(x) < 0.0001f)
		return 1.0f;
	else
		return (sin(x) / x);
}

float factorial(int a_number)
{
	switch(a_number) {
		case  0: return 1.0f;
		case  1: return 1.0f;
		case  2: return 2.0f;
		case  3: return 6.0f;
		case  4: return 24.0f;
		case  5: return 120.0f;
		case  6: return 720.0f;
		case  7: return 5040.0f;
		case  8: return 40320.0f;
		case  9: return 362880.0f;
		case 10: return 3628800.0f;
		case 11: return 39916800.0f;
		case 12: return 479001600.0f;
		case 13: return 6227020800.0f;
		case 14: return 87178291200.0f;
		case 15: return 1307674368000.0f;
		case 16: return 20922789888000.0f;
		case 17: return 355687428096000.0f;
		case 18: return 6402373705728000.0f;
		case 19: return 121645100408832000.0f;
		case 20: return 2432902008176640000.0f;
		case 21: return 51090942171709440000.0f;
		case 22: return 1124000727777607680000.0f;
		case 23: return 25852016738884976640000.0f;
		case 24: return 620448401733239439360000.0f;
		case 25: return 15511210043330985984000000.0f;
		case 26: return 403291461126605635584000000.0f;
		case 27: return 10888869450418352160768000000.0f;
		case 28: return 304888344611713860501504000000.0f;
		case 29: return 8841761993739701954543616000000.0f;
		case 30: return 265252859812191058636308480000000.0f;
		case 31: return 8222838654177922817725562880000000.0f;
		case 32: return 263130836933693530167218012160000000.0f;
		case 33: return 8683317618811886495518194401280000000.0f;
	}
	return 0.0f;
}

float P(float l, float m, float x)
{
	// evaluate an Associated Legendre Polynomial P(l,m,x) at x
	float pmm = 1.0f;

	if (m > 0.0f) {
		float somx2 = sqrt((1.0f - x) * (1.0f + x));
		float fact = 1.0f;

		for (float i = 1.0f; i <= m; i = i + 1.0f) {
			pmm *= (-fact) * somx2;
			fact += 2.0f;
		}
	}

	if (l == m) {
		return pmm;
	}

	float pmmp1 = x * (2.0f * m + 1.0f) * pmm;

	if (l == m + 1.0f) {
		return pmmp1;
	}

	float pll = 0.0f;
	for (float ll = m + 2.0f; ll <= l; ll = ll + 1.0f) {
		pll = ((2.0f * ll - 1.0f) * x * pmmp1 - (ll + m - 1.0f) * pmm) / (ll - m);
		pmm = pmmp1;
		pmmp1 = pll;
	}

	return pll;
}

float K(float l, float m)
{
	// renormalisation constant for SH function
	float temp = ((2.0f * l + 1.0f) * factorial(l - m)) / (4.0f * PI * factorial(l + m));

	return sqrt(temp);
}

float SH(float l, float m, float theta, float phi)
{
	// return a point sample of a Spherical Harmonic basis function
	// l is the band, range [0..N]
	// m in the range [-l..l]
	// theta in the range [0..Pi]
	// phi in the range [0..2*Pi]
	const float sqrt2 = 1.41421356237f;

	if (m == 0.0f) {
		return K(l, 0.0f) * P(l, m, cos(theta));
		// return 1.0f;
	}
	else {
		if (m > 0.0f) {
			return sqrt2 * K(l, m) * cos(m * phi) * P(l, m, cos(theta));
			// return 1.0f;
		}
		else {
			// return 1.0f;
			return sqrt2 * K(l, -m) * sin(-m * phi) * P(l, -m, cos(theta));
		}
	}
}

// +------------------------------------------------------------------------------+
// |   Helper functions for Spherical Harmonics evaluation END                    |
// +------------------------------------------------------------------------------+

float3 sampleGlyph(float s, float t){

	auto block = cg::this_thread_block();

	float scale = 10.0f;
	float height = 0.105f;

	float time = uniforms.time;
	// float time = 123.0;
	float su = s - 0.5f;
	float tu = t - 0.5f;
	float u = 2.0f * 3.14f * s;
	float v = 3.14f * t;
	
	// +-----------------------------------------------------------------------------------------------------------------+
	// | Option A BEGIN: Render _one_ SH basis function:                                                                 |
	// |                                                                             Activate by uncommenting until END: |
	// |                                                                             (and disable other options)         |
	// int shBand = 4;
	// int shFunctionIndex = 2;
	// float f = SH(shBand, shFunctionIndex, v, u);
	// f = abs(f);
	// f = lerp(f, 0.5f, max(0.0f, sinf(time)));
	// |                                                                                                    Option A END |
	// +-----------------------------------------------------------------------------------------------------------------+

	// +-----------------------------------------------------------------------------------------------------------------+
	// | Option B BEGIN: Try to render something that is composed of multiple SH basis functions (a "real" glyph):       |
	// |                                                                             Activate by uncommenting until END: |
	// |                                                                             (and disable other options)         |
	// // Fake evaluation
	// float f = 0.f; // Initialize to 0 and then add all the coefficients from the different bands
	// const int l_max = 0; // TODO: Herr Doktor, sobald ich l_max auf irgendwas > 0 setze, geht nix mehr :so_sad:
	// const int totalCoeffs = (l_max+1) * (l_max+1);
	// for (int l = 0; l <= l_max; ++l) {
	// 	for (int m = -l; m <= l; ++m) {
	// 		int runningCoeffId = l*l + l/2 + m;
	// 		float iHaveNoIdeaWhatImScaling = sinf(float(runningCoeffId) / float(totalCoeffs) * 3.14 - time);

	// 		float coeffScale = SH(l, m, v, u);
	// 		coeffScale = abs(coeffScale);
	// 		f += iHaveNoIdeaWhatImScaling * coeffScale;
	// 	}
	// }
	// |                                                                                                    Option B END |
	// +-----------------------------------------------------------------------------------------------------------------+

	// +-----------------------------------------------------------------------------------------------------------------+
	// | Option C BEGIN: SH coefficients from here: https://www.shadertoy.com/view/dlGSDV                                |
	// |                                                                             Activate by uncommenting until END: |
	// |                                                                             (and disable other options)         |
	// The index of the highest used band of the spherical harmonics basis. Must be
	// even, at least 2 and at most 12.
	#define SH_DEGREE 12
	#define SH_COUNT (((SH_DEGREE + 1) * (SH_DEGREE + 2)) / 2)

	int weightIndex = 0;
	float f = 0.f; // Initialize to 0 and then add all the coefficients from the different bands
	for (int l = 0; l <= SH_DEGREE; l += 2) {
		for (int m = -l; m <= l; ++m) {
			float coeffScale = SH(l, m, v, u);
			// float coeffScale = 1.0f;
			// coeffScale = abs(coeffScale); // TODO: not sure if needed

			// static
			// f += sh_coefficients[weightIndex++] * coeffScale;

			// animated
			f += sin(float(0.02f * uniforms.time * weightIndex)) * sh_coefficients[weightIndex++] * coeffScale;
		}
	}
	// |                                                                                                    Option C END |
	// +-----------------------------------------------------------------------------------------------------------------+

	float3 xyz = {
		f * cos(u) * sin(v),
		f * sin(u) * sin(v),
		f * cos(v)
	};
	return xyz;
};


// sampleSinCos, samplePlane, sampleSphere;
// auto sample = sampleSinCos;

auto getSampler(int model){
	switch (model) {
		case MODEL_FUNKY_PLANE:
			return sampleFunkyPlane;
		case MODEL_EXTRA_FUNKY_PLANE:
			return sampleExtraFunkyPlane;
		case MODEL_SPHERE:
			return sampleSphere;
		case MODEL_GLYPH:
			return sampleGlyph;
		default:
			return samplePlane;
	}
};

void generatePatches2(
	Model* models, uint32_t* numModels, 
	Patch* patches, uint32_t* numPatches, 
	int threshold, 
	Uniforms& uniforms, 
	Stats* stats
){

	auto grid = cg::this_grid();

	if(grid.thread_rank() < 30){
		stats->numPatches[grid.thread_rank()] = 0;
	}
	
	Patch* patches_tmp_0 = allocator->alloc<Patch*>(MAX_PATCHES * sizeof(Patch));
	Patch* patches_tmp_1 = allocator->alloc<Patch*>(MAX_PATCHES * sizeof(Patch));
	uint32_t* numPatches_tmp_0 = allocator->alloc<uint32_t*>(4);
	uint32_t* numPatches_tmp_1 = allocator->alloc<uint32_t*>(4);

	struct PatchData{
		Patch* patch;
		uint32_t* counter;
	};

	PatchData* pingpong = allocator->alloc<PatchData*>(2 * sizeof(PatchData));
	pingpong[0] = {patches_tmp_0, numPatches_tmp_0};
	pingpong[1] = {patches_tmp_1, numPatches_tmp_1};

	if(grid.thread_rank() == 0){
		*numPatches_tmp_0 = 0;
		*numPatches_tmp_1 = 0;
	}

	grid.sync();

	// Create initial set of patches
	constexpr int initialPatchGridSize = 8;

	for(int modelID = 0; modelID < *numModels; modelID++){
		if(grid.thread_rank() < initialPatchGridSize * initialPatchGridSize){

			int index = grid.thread_rank();
			int ix = index % initialPatchGridSize;
			int iy = index / initialPatchGridSize;

			float s_min = float(ix + 0) / float(initialPatchGridSize);
			float s_max = float(ix + 1) / float(initialPatchGridSize);
			float t_min = float(iy + 0) / float(initialPatchGridSize);
			float t_max = float(iy + 1) / float(initialPatchGridSize);

			Patch patch = {s_min, s_max, t_min, t_max, modelID};

			patches_tmp_0[modelID * initialPatchGridSize * initialPatchGridSize + index] = patch;
		}
	}

	*numPatches_tmp_0 = *numModels * initialPatchGridSize * initialPatchGridSize;

	grid.sync();

	int level = 0;

	// SUBDIVIDE LARGE PATCHES
	// - if too large, divide and store in target
	// - if not too large, store in <patches>
	// - too large as in pixel size
	auto subdivide = [&](Patch* source, uint32_t* sourceCounter, Patch* target, uint32_t* targetCounter){

		processRange(*sourceCounter, [&](int index){
			Patch patch = source[index];

			float s_c = (patch.s_min + patch.s_max) * 0.5f;
			float t_c = (patch.t_min + patch.t_max) * 0.5f;

			Model model = models[patch.modelID];
			auto sample = getSampler(model.functionID);

			float3 p_00 = sample(patch.s_min, patch.t_min) + model.position;
			float3 p_01 = sample(patch.s_min, patch.t_max) + model.position;
			float3 p_10 = sample(patch.s_max, patch.t_min) + model.position;
			float3 p_11 = sample(patch.s_max, patch.t_max) + model.position;
			float3 p_c = sample(s_c, t_c) + model.position;


			float3 nodeMin = {
				min(min(min(p_00.x, p_01.x), min(p_10.x, p_11.x)), p_c.x),
				min(min(min(p_00.y, p_01.y), min(p_10.y, p_11.y)), p_c.y),
				min(min(min(p_00.z, p_01.z), min(p_10.z, p_11.z)), p_c.z),
			};
			float3 nodeMax = {
				max(max(max(p_00.x, p_01.x), max(p_10.x, p_11.x)), p_c.x),
				max(max(max(p_00.y, p_01.y), max(p_10.y, p_11.y)), p_c.y),
				max(max(max(p_00.z, p_01.z), max(p_10.z, p_11.z)), p_c.z),
			};
			bool isIntersectingFrustum = intersectsFrustum(uniforms.locked_transform, nodeMin, nodeMax);

			if(!isIntersectingFrustum){
				return;
			}

			float4 ps_00 = toScreen_locked(p_00, uniforms);
			float4 ps_01 = toScreen_locked(p_01, uniforms);
			float4 ps_10 = toScreen_locked(p_10, uniforms);
			float4 ps_11 = toScreen_locked(p_11, uniforms);
			float4 ps_c = toScreen_locked(p_c, uniforms);

			float min_x = min(min(min(ps_00.x, ps_01.x), min(ps_10.x, ps_11.x)), ps_c.x);
			float min_y = min(min(min(ps_00.y, ps_01.y), min(ps_10.y, ps_11.y)), ps_c.y);
			float max_x = max(max(max(ps_00.x, ps_01.x), max(ps_10.x, ps_11.x)), ps_c.x);
			float max_y = max(max(max(ps_00.y, ps_01.y), max(ps_10.y, ps_11.y)), ps_c.y);

			float s_x = max_x - min_x;
			float s_y = max_y - min_y;
			float area = s_x * s_y;

			if(area > threshold * threshold){
				// too large, subdivide into 4 smaller patches

				uint32_t targetIndex = atomicAdd(targetCounter, 4);

				if(targetIndex >= MAX_PATCHES) return;

				float s_center = (patch.s_min + patch.s_max) / 2.0f;
				float t_center = (patch.t_min + patch.t_max) / 2.0f;

				Patch patch_00 = {patch.s_min, s_center, patch.t_min, t_center, patch.modelID};
				Patch patch_01 = {patch.s_min, s_center, t_center, patch.t_max, patch.modelID};
				Patch patch_10 = {s_center, patch.s_max, patch.t_min, t_center, patch.modelID};
				Patch patch_11 = {s_center, patch.s_max, t_center, patch.t_max, patch.modelID};

				target[targetIndex + 0] = patch_00;
				target[targetIndex + 1] = patch_01;
				target[targetIndex + 2] = patch_10;
				target[targetIndex + 3] = patch_11;

			}else{
				// small enough, add to list of patches

				// TODO: do backface culling here? 
				// If the patch faces away from the camera, ignore it. 

				// float3 t_01 = p_01 - p_00;
				// float3 t_10 = p_10 - p_00;
				// float3 N = normalize(cross(t_01, t_10));
				// float3 N_v = make_float3(uniforms.view * float4{N.x, N.y, N.z, 0.0});
				
				// float a = dot(N_v, float3{0.0, 0.0, 1.0});
				// if(a < 0.0) return;

				uint32_t targetIndex = atomicAdd(numPatches, 1);

				if(targetIndex >= MAX_PATCHES) return;

				patches[targetIndex] = patch;

				atomicAdd(&stats->numPatches[level], 1);
			}

		});
	};

	grid.sync();

	// DIVIDE IN PING-PONG FASHION
	for(int i = 0; i < 14; i++){

		grid.sync();

		int sourceIndex = (i + 0) % 2;
		int targetIndex = (i + 1) % 2;

		PatchData source = pingpong[sourceIndex];
		PatchData target = pingpong[targetIndex];

		*target.counter = 0;

		grid.sync();

		subdivide(source.patch, source.counter, target.patch, target.counter);

		grid.sync();

		*target.counter = min(*target.counter, MAX_PATCHES);
		*numPatches = min(*numPatches, MAX_PATCHES);

		level++;
	}

}

// Rasterize a patch by sampling a 32x32 grid.
// - We launch with workgroup-size 1024, i.e., 32x32 threads
// - Therefore we ca let each thread process one sample of the patch concurrently
// - However, workgroup threads (unlike warp threads) don't operate simultaneously.
//      - So in order to compute the normal, we compute samples and store the results in shared memory
//      - Then we sync the group, and then each thread loads adjacent samples to compute the normal
void rasterizePatches_32x32(
	Model* models, uint32_t* numModels,
	Patch* patches, uint32_t* numPatches, 
	uint64_t* framebuffer, 
	Uniforms& uniforms,
	Patch* newPatches, uint32_t* numNewPatches, 
	bool createNewPatches
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint32_t& processedPatches = *allocator->alloc<uint32_t*>(4);
	if(grid.thread_rank() == 0){
		processedPatches = 0;
	}

	__shared__ int sh_patchIndex;
	__shared__ float sh_samples[1024 * 3];
	__shared__ int sh_pixelPositions[1024];
	__shared__ bool sh_NeedsRefinement;

	block.sync();

	int loop_max = 10'000;
	for(int loop_i = 0; loop_i < loop_max; loop_i++){

		// grab the index of the next unprocessed patch
		block.sync();
		if(block.thread_rank() == 0){
			sh_patchIndex = atomicAdd(&processedPatches, 1);
			sh_NeedsRefinement = false;
		}
		block.sync();

		if(sh_patchIndex >= *numPatches) break;

		Patch patch = patches[sh_patchIndex];
		Model model = models[patch.modelID];

		auto sample = getSampler(model.functionID);

		float s_min = patch.s_min;
		float s_max = patch.s_max;
		float t_min = patch.t_min;
		float t_max = patch.t_max;

		int index_t = block.thread_rank();
		int index_tx = index_t % 32;
		int index_ty = index_t / 32;

		float uts = float(index_tx) / 32.0f;
		float vts = float(index_ty) / 32.0f;

		float s = (1.0f - uts) * s_min + uts * s_max;
		float t = (1.0f - vts) * t_min + vts * t_max;

		float3 p = sample(s, t) + model.position;

		block.sync();

		// Store samples in shared memory, so that other threads can access them
		sh_samples[3 * index_t + 0] = p.x;
		sh_samples[3 * index_t + 1] = p.y;
		sh_samples[3 * index_t + 2] = p.z;

		block.sync();

		int inx = index_t + (index_tx < 31 ?  1 :  -1);
		int iny = index_t + (index_ty < 31 ? 32 : -32);
		int inxy = index_t;
		if(index_tx < 31) inxy += 1;
		if(index_ty < 31) inxy += 32;

		// Lead adjacent samples (next-x and next-y) to compute normal
		float3 pnx = {sh_samples[3 * inx + 0], sh_samples[3 * inx + 1], sh_samples[3 * inx + 2]};
		float3 pny = {sh_samples[3 * iny + 0], sh_samples[3 * iny + 1], sh_samples[3 * iny + 2]};

		float3 tx = normalize(pnx - p);
		float3 ty = normalize(pny - p);
		float3 N = normalize(cross(ty, tx));

		float4 ps = toScreen(p, uniforms);

		// Compute pixel positions and store them in shared memory so that ajdacent threads can access them
		uint32_t pixelPos; 
		int16_t* pixelPos_u16 = (int16_t*)&pixelPos;
		pixelPos_u16[0] = int(ps.x);
		pixelPos_u16[1] = int(ps.y);
		sh_pixelPositions[index_t] = pixelPos;

		block.sync();

		// compute pixel distances to next samples in x, y, or both directions
		uint32_t pp_00 = sh_pixelPositions[index_t];
		uint32_t pp_10 = sh_pixelPositions[inx];
		uint32_t pp_01 = sh_pixelPositions[iny];
		int16_t* pp_00_u16 = (int16_t*)&pp_00;
		int16_t* pp_10_u16 = (int16_t*)&pp_10;
		int16_t* pp_01_u16 = (int16_t*)&pp_01;

		// the max distance
		int d_max_10 = max(abs(pp_10_u16[0] - pp_00_u16[0]), abs(pp_10_u16[1] - pp_00_u16[1]));
		int d_max_01 = max(abs(pp_01_u16[0] - pp_00_u16[0]), abs(pp_01_u16[1] - pp_00_u16[1]));
		int d_max = max(d_max_10, d_max_01);

		uint32_t color = 0;
		// uint32_t color = patch.dbg * 12345678;
		uint8_t* rgba = (uint8_t*)&color;
		rgba[0] = 200.0f * N.x;
		rgba[1] = 200.0f * N.y;
		rgba[2] = 200.0f * N.z;
		rgba[3] = 255;

		// mark samples where distances to next samples are >1px
		if(index_tx < 31 && index_ty < 31)
		if(d_max > 1){
			// color = 0x00ff00ff;
			sh_NeedsRefinement = true;
		}

		block.sync();

		// if(sh_NeedsRefinement){
		// 	color = 0x0000ffff;
		// }

		if(!createNewPatches){
			color = 0x000000ff;
		}

		// color = (patch.x + 1) * (patch.y + 13) * 1234567;

		// drawSprite(ps, framebuffer, color, uniforms);

		// if(N.x > 10.0)
		drawPoint(ps, framebuffer, color, uniforms);

		block.sync();

		// If pixel distances are too large, create new patches to draw
		if(createNewPatches)
		if(sh_NeedsRefinement && block.thread_rank() == 0){

			uint32_t newPatchIndex = atomicAdd(numNewPatches, 4);

			if(newPatchIndex >= MAX_PATCHES){
				atomicSub(numNewPatches, 4);
				continue;
			}

			// marked as volatile to reduce register pressure and allow larger workgroup size
			volatile float s_center = (patch.s_min + patch.s_max) * 0.5f;
			volatile float t_center = (patch.t_min + patch.t_max) * 0.5f;

			newPatches[newPatchIndex + 0] = {
				patch.s_min, s_center,
				patch.t_min, t_center, 
				patch.modelID
			};

			newPatches[newPatchIndex + 1] = {
				s_center, patch.s_max,
				patch.t_min, t_center,
				patch.modelID
			};

			newPatches[newPatchIndex + 2] = {
				patch.s_min, s_center,
				t_center, patch.t_max,
				patch.modelID
			};

			newPatches[newPatchIndex + 3] = {
				s_center, patch.s_max,
				t_center, patch.t_max,
				patch.modelID
			};
		}
	}
}

// Unlike the 32x32 method, this method draws the patch "line by line".
// - We launch with 128 threads, which sample atx =threadID / 128.0 and y using the loop counter i. 
// - No normals yet, but we could probably sample the first 2x128 samples and compute normals, 
//   and then with each iteration next row of 128 samples and compute normals using previous 128 samples. 
void rasterizePatches_runnin_thru(
	Model* models, uint32_t* numModels,
	Patch* patches, uint32_t* numPatches,
	uint64_t* framebuffer, 
	Uniforms& uniforms
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint32_t& processedPatches = *allocator->alloc<uint32_t*>(4);
	if(grid.thread_rank() == 0){
		processedPatches = 0;
	}

	__shared__ int sh_patchIndex;
	__shared__ float sh_samples[1024 * 3];

	block.sync();

	int loop_max = 10'000;
	for(int loop_i = 0; loop_i < loop_max; loop_i++){

		// grab the index of the next unprocessed triangle
		block.sync();
		if(block.thread_rank() == 0){
			sh_patchIndex = atomicAdd(&processedPatches, 1);
		}
		block.sync();

		if(sh_patchIndex >= *numPatches) break;

		Patch patch = patches[sh_patchIndex];
		Model model = models[patch.modelID];
		auto sample = getSampler(uniforms.model);

		float s_min = patch.s_min;
		float s_max = patch.s_max;
		float t_min = patch.t_min;
		float t_max = patch.t_max;

		int index_t = block.thread_rank();
		float ut = float(index_t) / float(block.num_threads());

		float s = (1.0f - ut) * s_min + ut * s_max;
		float t = t_min;

		
		float steps = 64.0f;
		for(float i = 0.0f; i < steps; i = i + 1.0f){
			float vt = i / steps;
			float t = (1.0f - vt) * t_min + vt * t_max;

			float3 p = sample(s, t) + model.position;
			uint32_t color = 0x000000ff;
			float4 ps = toScreen(p, uniforms);


			color = 1234567.0f * (123.0f + patch.s_min * patch.t_min);

			// if(p.x * p.y == 123.0f)
			drawPoint(ps, framebuffer, color, uniforms);

		}


		

		block.sync();
	}
}

extern "C" __global__
void kernel_generate_scene(
	const Uniforms _uniforms,
	unsigned int* buffer,
	Model* models, uint32_t* numModels,
	Patch* patches, uint32_t* numPatches,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uniforms = _uniforms;

	if(grid.thread_rank() == 0){

		// models[0] = {uniforms.model, float3{ 2.1, 0.0, -2.1}};
		// models[1] = {uniforms.model, float3{ 0.0, 0.0, -2.1}};
		// models[2] = {uniforms.model, float3{-2.1, 0.0, -2.1}};

		// models[3] = {uniforms.model, float3{ 2.1, 0.0, 0.0}};
		// models[4] = {uniforms.model, float3{ 0.0, 0.0, 0.0}};
		// models[5] = {uniforms.model, float3{-2.1, 0.0, 0.0}};

		// models[6] = {uniforms.model, float3{ 2.1, 0.0, 2.1}};
		// models[7] = {uniforms.model, float3{ 0.0, 0.0, 2.1}};
		// models[8] = {uniforms.model, float3{-2.1, 0.0, 2.1}};

		// *numModels = 9;

		models[0] = {uniforms.model, float3{ 0.0, 0.0, 0.0}};

		*numModels = 1;
	}

}

extern "C" __global__
void kernel_generate_patches(
	const Uniforms _uniforms,
	unsigned int* buffer,
	Model* models, uint32_t* numModels,
	Patch* patches, uint32_t* numPatches,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uniforms = _uniforms;

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	grid.sync();
	if(grid.thread_rank() == 0){
		*numPatches = 0;
	}
	grid.sync();

	int threshold = 32;
	if(uniforms.method == METHOD_32X32){
		threshold = 32;
	}else if(uniforms.method == METHOD_RUNNIN_THRU){
		threshold = 64;
	}

	generatePatches2(models, numModels, patches, numPatches, threshold, uniforms, stats);


}

// Compute a whole lot of samples to check how many we can compute in a given time
// - We don't write the results to screen because that takes time, but we need to 
//   act as if we do so that the compiler doesn't optimize sample generation away.
// - Simply do some if that virtually never evaluates to true, and draw only if it's true. 
extern "C" __global__
void kernel_sampleperf_test(
	const Uniforms _uniforms,
	unsigned int* buffer,
	uint64_t* framebuffer,
	Model* models, uint32_t* numModels,
	Patch* patches, uint32_t* numPatches,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uniforms = _uniforms;

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	uint64_t t_00 = nanotime();

	grid.sync();


	auto sampler = getSampler(uniforms.model);
	int gridSize = 10'000; // 100M
	int numPixels = int(uniforms.width * uniforms.height);

	processRange(gridSize * gridSize, [&](int index){

		uint32_t ix = index % gridSize;
		uint32_t iy = index / gridSize;

		float s = float(ix) / float(gridSize);
		float t = float(iy) / float(gridSize);

		float3 sample = sampler(s, t);

		// Some bogus <if> that pretents to do something but virtually never evaluates to true,
		// so that sampler(...) isn't optimized away.
		if(sample.x * sample.y == 123.0f){
			int pixelID = int(sample.x * sample.y * 1234.0f) % numPixels;
			framebuffer[10'000] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
		}

	});

	grid.sync();

	uint64_t t_20 = nanotime();

	// TODO: should do timings in host code with events.
	if(grid.thread_rank() == 0 && (stats->frameID % 100) == 0){
		stats->time_0 = float((t_20 - t_00) / 1000llu) / 1000.0f;
		stats->time_1 = 0.0;
	}

}

extern "C" __global__
void kernel_clear_framebuffer(
	const Uniforms _uniforms,
	unsigned int* buffer,
	uint64_t* framebuffer,
	Model* models, uint32_t* numModels,
	Patch* patches, uint32_t* numPatches,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){

	auto grid = cg::this_grid();

	processRange(0, _uniforms.width * _uniforms.height, [&](int pixelIndex){
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
	});

}

extern "C" __global__
void kernel_framebuffer_to_OpenGL(
	const Uniforms _uniforms,
	unsigned int* buffer,
	uint64_t* framebuffer,
	Model* models, uint32_t* numModels,
	Patch* patches, uint32_t* numPatches,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){

	auto grid = cg::this_grid();

	// transfer framebuffer to opengl texture
	processRange(0, _uniforms.width * _uniforms.height, [&](int pixelIndex){

		int x = pixelIndex % int(_uniforms.width);
		int y = pixelIndex / int(_uniforms.width);

		uint64_t encoded = framebuffer[pixelIndex];
		uint32_t color = encoded & 0xffffffffull;

		surf2Dwrite(color, gl_colorbuffer, x * 4, y);
	});

	if(grid.thread_rank() == 0){
		stats->frameID++;
	}

}

extern "C" __global__
void kernel_rasterize_patches_32x32(
	const Uniforms _uniforms,
	unsigned int* buffer,
	uint64_t* framebuffer,
	Model* models, uint32_t* numModels,
	Patch* patches, uint32_t* numPatches,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	Allocator _allocator(buffer, 0);

	uniforms = _uniforms;
	allocator = &_allocator;

	Patch* newPatches = allocator->alloc<Patch*>(MAX_PATCHES * sizeof(Patch));
	uint32_t* numNewPatches = allocator->alloc<uint32_t*>(4);

	grid.sync();

	if(grid.thread_rank() == 0){
		*numNewPatches = 0;
	}

	grid.sync();

	rasterizePatches_32x32(
		models, numModels,
		patches, numPatches, 
		framebuffer, 
		uniforms,
		newPatches, numNewPatches, true
	);
	grid.sync();

	if(uniforms.enableRefinement){
		// the earlier call to rasterizePatches checked for holes and created
		// a refined list of patches. render them now. 
		rasterizePatches_32x32(
			models, numModels,
			newPatches, numNewPatches, 
			framebuffer, 
			uniforms,
			patches, numPatches, false
		);
		grid.sync();
	}

	grid.sync();

	

}

extern "C" __global__
void kernel_rasterize_patches_runnin_thru(
	const Uniforms _uniforms,
	unsigned int* buffer,
	uint64_t* framebuffer,
	Model* models, uint32_t* numModels,
	Patch* patches, uint32_t* numPatches,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	Allocator _allocator(buffer, 0);

	uniforms = _uniforms;
	allocator = &_allocator;

	grid.sync();
	rasterizePatches_runnin_thru(models, numModels, patches, numPatches, framebuffer, uniforms);

}

// just some debugging. and checking how many registers a simple kernel utilizes.
extern "C" __global__
void kernel_test(
	const Uniforms _uniforms,
	unsigned int* buffer,
	Model* models, uint32_t* numModels,
	Patch* patches, uint32_t* numPatches,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();


	uint32_t size = uniforms.width * uniforms.height;
	uint32_t totalThreadCount = blockDim.x * gridDim.x;
	int itemsPerThread = size / totalThreadCount + 1;

	for(int i = 0; i < itemsPerThread; i++){
		int block_offset  = itemsPerThread * blockIdx.x * blockDim.x;
		int thread_offset = itemsPerThread * threadIdx.x;
		int index = block_offset + thread_offset + i;

		if(index >= size){
			break;
		}

		int x = index % int(uniforms.width);
		int y = index / int(uniforms.width);

		uint32_t color = 0x00112233;

		surf2Dwrite(color, gl_colorbuffer, x * 4, y);
	}

}
