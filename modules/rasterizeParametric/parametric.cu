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
constexpr float HALF_PI = PI * 0.5f;
constexpr float QUARTER_PI = PI * 0.25f;
constexpr float TWO_PI = 2.f * PI;
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

void drawPoint(float4 coord, uint64_t* framebuffer, uint64_t* heatmap,  uint32_t color, Uniforms& uniforms){

	int x = coord.x;
	int y = coord.y;

	if(x > 1 && x < uniforms.width  - 2.0f)
	if(y > 1 && y < uniforms.height - 2.0f){

		// SINGLE PIXEL
		uint32_t pixelID = x + int(uniforms.width) * y;
		uint64_t udepth = *((uint32_t*)&coord.w);
		uint64_t encoded = (udepth << 32) | color;

		atomicMin(&framebuffer[pixelID], encoded);

		atomicAdd(&heatmap[pixelID], 1);
	}
}

void drawSprite(float4 coord, uint64_t* framebuffer, uint64_t* heatmap,  uint32_t color, Uniforms& uniforms){

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

void drawSprite_w(float3 pos, uint64_t* framebuffer, uint64_t* heatmap,  uint32_t color, Uniforms& uniforms){

	float4 ndc = uniforms.transform * float4{pos.x, pos.y, pos.z, 1.0f};
	ndc.x = ndc.x / ndc.w;
	ndc.y = ndc.y / ndc.w;
	ndc.z = ndc.z / ndc.w;

	float4 coord = float4{
		(ndc.x * 0.5f + 0.5f) * uniforms.width,
		(ndc.y * 0.5f + 0.5f) * uniforms.height,
		ndc.z, ndc.w
	};

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

auto toScreen_locked = [&](float3 p, const Uniforms& uniforms){
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
			f += sh_coefficients[weightIndex++] * coeffScale;

			// animated
			// f += sin(float(0.02f * uniforms.time * weightIndex)) * sh_coefficients[weightIndex++] * coeffScale;
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

/** Moves the points onto the sphere surface
 * \brief R2 -> R3 sphere
 * \param u Expected input range: [0, PI]
 * \param v Expected input range: [0, 2*PI]
 * \return Position according to the parametric sphere equation
 */
float3 to_sphere(float u, float v) {
	return float3{
	    /* x: */ sinf(u) * cosf(v),
	    /* y: */ sinf(u) * sinf(v),
	    /* z: */ cosf(u)};
}

template<typename T> 
void swap(T& t1, T& t2) {
    T temp = std::move(t1);
    t1 = std::move(t2);
    t2 = std::move(temp);
}

// s, t in range 0 to 1!
float3 sampleJohisHeart(float s, float t){
	float u = PI * s;
	float v = TWO_PI * t;

	auto pos = to_sphere(u, v);

	if (u < HALF_PI) {
		pos.z *= 1.f - (cosf(sqrtf(sqrtf(abs(pos.x*PI*0.7f))))*0.8f);
	}
	else {
		pos.x *= sinf(u) * sinf(u);
	}
	pos.x *= 0.9f;
	pos.y *= 0.4f;

	swap(pos.y, pos.z);

	return pos;
};

// s, t in range 0 to 1!
float3 sampleSpherehog(float s, float t){
	float u = PI * s;
	float v = TWO_PI * t;
	
	// Position:
	auto pos = to_sphere(u, v);

	constexpr float NUMSPIKES = 10.f;
	constexpr float SPIKENARROWNESS = 100.f;
	float spikeheight = 0.5f;

	auto repeatU =  u / PI * NUMSPIKES - roundf(u / PI * NUMSPIKES );
	auto repeatV = v / PI * NUMSPIKES - roundf(v / PI * NUMSPIKES );
	auto d = repeatU*repeatU + repeatV*repeatV;
	float r = 1.f + exp(-d * SPIKENARROWNESS) * spikeheight;

	pos *= r;
	swap(pos.y, pos.z);

	return pos;
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
		case JOHIS_HEART:
			return sampleJohisHeart;
		case SPHEREHOG:
			return sampleSpherehog;
		default:
			return samplePlane;
	}
};

void generatePatchSamples(Patch* patch, Model* models){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	
	// if(grid.thread_rank() == 0){
	// 	printf("%i \n", models[patch->modelID].functionID);
	// }
	auto sampler = getSampler(MODEL_GLYPH);
	// auto sampler = getSampler(models[patch->modelID].functionID);

	patch->min = {Infinity, Infinity, Infinity};
	patch->max = {-Infinity, -Infinity, -Infinity};

	// cg::this_grid().sync();
	block.sync();

	// if(block.thread_rank() == 0 && uniforms.frameCount % 500 == 0){
	// 	printf("generate block: %i, %llu \n", grid.block_rank(), patch);
	// }

	for(
		int index = block.thread_rank();
		index < POINTS_PER_PATCH;
		index += block.num_threads()
	){
		int x = index % PATCH_RESOLUTION;
		int y = index / PATCH_RESOLUTION;

		float u = float(x) / float(PATCH_RESOLUTION);
		float v = float(y) / float(PATCH_RESOLUTION);

		float s = u * (patch->st_max.x - patch->st_min.x) + patch->st_min.x;
		float t = v * (patch->st_max.y - patch->st_min.y) + patch->st_min.y;

		Point point;
		point.pos = sampler(s, t);
		point.rgba[0] = 255.0 * u;
		point.rgba[1] = 255.0 * v;
		point.rgba[2] = 0;

		// if(index > 100){
		// 	point.color = 0x00ff00ff;
		// }

		// if(index < 32)
		// if(uniforms.frameCount % 500 == 0)
		// {
		// 	printf("[%4i] (%.1f, %.1f, %.1f) - %i \n", index, point.pos.x, point.pos.y, point.pos.z, point.rgba[0]);
		// 	// printf("%.1f, %.1f, %.1f - %i \n", point.pos.x, point.pos.y, point.pos.z, point.rgba[0]);
		// }

		atomicMinFloat(&patch->min.x, point.pos.x);
		atomicMinFloat(&patch->min.y, point.pos.y);
		atomicMinFloat(&patch->min.z, point.pos.z);
		atomicMaxFloat(&patch->max.x, point.pos.x);
		atomicMaxFloat(&patch->max.y, point.pos.y);
		atomicMaxFloat(&patch->max.z, point.pos.z);
		
		patch->points[index] = point;
	}

	block.sync();

	// if(block.thread_rank() < 32 && uniforms.frameCount % 500 == 0){
	// 	Point point = patch->points[block.thread_rank()];
	// 	printf("[%4i] (%.1f, %.1f, %.1f) - %i \n", block.thread_rank(), point.pos.x, point.pos.y, point.pos.z, point.rgba[0]);
	// }






	// processRange(POINTS_PER_PATCH, [&](int index){
	// 	int x = index % PATCH_RESOLUTION;
	// 	int y = index / PATCH_RESOLUTION;

	// 	float u = float(x) / float(PATCH_RESOLUTION);
	// 	float v = float(y) / float(PATCH_RESOLUTION);

	// 	float s = u * (patch->st_max.x - patch->st_min.x) + patch->st_min.x;
	// 	float t = v * (patch->st_max.y - patch->st_min.y) + patch->st_min.y;

	// 	Point point;
	// 	point.pos = sampler(s, t);

	// 	atomicMinFloat(&patch->min.x, point.pos.x);
	// 	atomicMinFloat(&patch->min.y, point.pos.y);
	// 	atomicMinFloat(&patch->min.z, point.pos.z);
	// 	atomicMaxFloat(&patch->max.x, point.pos.x);
	// 	atomicMaxFloat(&patch->max.y, point.pos.y);
	// 	atomicMaxFloat(&patch->max.z, point.pos.z);
		
	// 	patch->points[index] = point;
	// });
}

extern "C" __global__
void kernel_create_scene(
	const Uniforms _uniforms,
	unsigned int* buffer,
	uint64_t* framebuffer, uint64_t* heatmap,
	Model* models, uint32_t* numModels,
	Patch** patches, uint32_t* numPatches,
	PatchPool* patchPool, Patch* patchStorage,
	Stats* stats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// if(grid.thread_rank() == 0 && uniforms.frameCount % 500 == 0) 
	// printf("================ create scene ================ \n");

	uniforms = _uniforms;

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;
	
	patchPool->capacity = PATCHES_CAPACITY;
	patchPool->offset = 0;

	processRange(PATCHES_CAPACITY, [&](int index){
		patchPool->pointers[index] = &patchStorage[index];
	});

	grid.sync();

	if(grid.block_rank() == 0)
	{ // Root
		patchPool->offset = 1;
		uint32_t patchIndex = 0;
		Patch* root = &patchStorage[patchIndex];
		root->st_min = {0.0f, 0.0f};
		root->st_max = {1.0f, 1.0f};
		root->parent = nullptr;
		root->children[0] = nullptr;
		root->children[1] = nullptr;
		root->children[2] = nullptr;
		root->children[3] = nullptr;
		root->modelID = uniforms.model;
		root->numPoints = POINTS_PER_PATCH;
		patches[0] = root;

		generatePatchSamples(root, models);
	}

	// {
	// 	uint32_t parentIndex = 0;
	// 	Patch* parent = &patchStorage[parentIndex];

	// 	uint32_t patchIndex = 1;
	// 	Patch* patch = &patchStorage[patchIndex];
	// 	patch->st_min = {0.0f, 0.0f};
	// 	patch->st_max = {1.0f, 0.5f};
	// 	patch->parent = parent;
	// 	patch->children[0] = nullptr;
	// 	patch->children[1] = nullptr;
	// 	patch->children[2] = nullptr;
	// 	patch->children[3] = nullptr;
	// 	patch->modelID = uniforms.model;
	// 	patch->numPoints = POINTS_PER_PATCH;

	// 	parent->children[0] = patch;

	// 	patches[1] = patch;

	// 	generatePatchSamples(patch, models);
	// }

	if(grid.thread_rank() == 0){
		// patches[0] = patchIndex;
		*numPatches = 1;
	}
}



extern "C" __global__
void kernel_update_patches(
	const Uniforms _uniforms,
	unsigned int* buffer,
	uint64_t* framebuffer, uint64_t* heatmap,
	Model* models, uint32_t* numModels,
	Patch** patches, uint32_t* numPatches,
	PatchPool* patchPool, Patch* patchStorage,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uniforms = _uniforms;

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	// return;

	auto min8 = [](float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7){

		float m0 = min(f0, f1);
		float m1 = min(f2, f3);
		float m2 = min(f4, f5);
		float m3 = min(f6, f7);

		float n0 = min(m0, m1);
		float n1 = min(m2, m3);

		return min(n0, n1);
	};

	auto max8 = [](float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7){

		float m0 = max(f0, f1);
		float m1 = max(f2, f3);
		float m2 = max(f4, f5);
		float m3 = max(f6, f7);

		float n0 = max(m0, m1);
		float n1 = max(m2, m3);

		return max(n0, n1);
	};

	int lNumPatches = *numPatches;

	uint32_t* next_numNodes = allocator->alloc<uint32_t*>(4);
	Patch** next_patches = allocator->alloc<Patch**>(PATCHES_CAPACITY * sizeof(Patch**));

	uint32_t* numNeedDeletion = allocator->alloc<uint32_t*>(4);
	Patch** needDeletion = allocator->alloc<Patch**>(PATCHES_CAPACITY * sizeof(Patch**));

	uint32_t* numNeedExpansion = allocator->alloc<uint32_t*>(4);
	Patch** needExpansion = allocator->alloc<Patch**>(PATCHES_CAPACITY * sizeof(Patch**));

	*next_numNodes = 0;
	*numNeedDeletion = 0;
	*numNeedExpansion = 0;

	grid.sync();

	processRange(lNumPatches, [&](int patchIndex){

		Patch* patch = patches[patchIndex];

		// - check bounding box
		// - if too big, create 4 smaller patches
		// - if too small, delete children

		float4 p000 = {patch->min.x, patch->min.y, patch->min.z, 1.0f};
		float4 p001 = {patch->min.x, patch->min.y, patch->max.z, 1.0f};
		float4 p010 = {patch->min.x, patch->max.y, patch->min.z, 1.0f};
		float4 p011 = {patch->min.x, patch->max.y, patch->max.z, 1.0f};
		float4 p100 = {patch->max.x, patch->min.y, patch->min.z, 1.0f};
		float4 p101 = {patch->max.x, patch->min.y, patch->max.z, 1.0f};
		float4 p110 = {patch->max.x, patch->max.y, patch->min.z, 1.0f};
		float4 p111 = {patch->max.x, patch->max.y, patch->max.z, 1.0f};

		float4 ndc000 = uniforms.transform * p000;
		float4 ndc001 = uniforms.transform * p001;
		float4 ndc010 = uniforms.transform * p010;
		float4 ndc011 = uniforms.transform * p011;
		float4 ndc100 = uniforms.transform * p100;
		float4 ndc101 = uniforms.transform * p101;
		float4 ndc110 = uniforms.transform * p110;
		float4 ndc111 = uniforms.transform * p111;

		float4 s000 = ((ndc000 / ndc000.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
		float4 s001 = ((ndc001 / ndc001.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
		float4 s010 = ((ndc010 / ndc010.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
		float4 s011 = ((ndc011 / ndc011.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
		float4 s100 = ((ndc100 / ndc100.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
		float4 s101 = ((ndc101 / ndc101.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
		float4 s110 = ((ndc110 / ndc110.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
		float4 s111 = ((ndc111 / ndc111.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};

		float smin_x = min8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
		float smin_y = min8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

		float smax_x = max8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
		float smax_y = max8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

		// screen-space size
		float dx = smax_x - smin_x;
		float dy = smax_y - smin_y;

		float pixels = dx * dy;


		// drawSprite_w(float3{patch->min.x, patch->min.y, patch->min.z}, framebuffer, heatmap, 0x00ff00ff, uniforms);
		// drawSprite_w(float3{patch->min.x, patch->min.y, patch->max.z}, framebuffer, heatmap, 0x00ff00ff, uniforms);
		// drawSprite_w(float3{patch->min.x, patch->max.y, patch->min.z}, framebuffer, heatmap, 0x00ff00ff, uniforms);
		// drawSprite_w(float3{patch->min.x, patch->max.y, patch->max.z}, framebuffer, heatmap, 0x00ff00ff, uniforms);
		// drawSprite_w(float3{patch->max.x, patch->min.y, patch->min.z}, framebuffer, heatmap, 0x00ff00ff, uniforms);
		// drawSprite_w(float3{patch->max.x, patch->min.y, patch->max.z}, framebuffer, heatmap, 0x00ff00ff, uniforms);
		// drawSprite_w(float3{patch->max.x, patch->max.y, patch->min.z}, framebuffer, heatmap, 0x00ff00ff, uniforms);
		// drawSprite_w(float3{patch->max.x, patch->max.y, patch->max.z}, framebuffer, heatmap, 0x00ff00ff, uniforms);

		// printf("%.1f, %.1f, %.1f \n", patch->min.x, patch->min.y, patch->min.z);
		// printf("dx: %.1f, dy: %.1f \n", dx, dy);
		// printf("%.1f pixels \n", pixels);

		constexpr int LARGE_PATCH_THRESHOLD = 64 * 64; // need to generate new nodes
		constexpr int SMALL_PATCH_THRESHOLD = 32 * 32; // need to remove nodes
		
		bool isLarge = pixels > LARGE_PATCH_THRESHOLD;
		bool isSmall = pixels < SMALL_PATCH_THRESHOLD;

		if(patch->st_max.x - patch->st_min.x > 0.4){
			isLarge = true;
		}

		// if(isLarge && isSmall){
		// 	isSmall = false;
		// }

		bool needsExpansion = isLarge && patch->isLeaf();
		bool needsDeletion = isSmall && patch->isLeaf() && patch->parent != nullptr;

		if(needsExpansion && needsDeletion){
			printf("shit...");
			return;
		}

		if(!needsDeletion){
			// this patch is visible
			uint32_t next_index = atomicAdd(next_numNodes, 1);
			next_patches[next_index] = patch;
		}
		
		if(needsExpansion){
			// needs expansion - create children
			uint32_t index = atomicAdd(numNeedExpansion, 1);
			needExpansion[index] = patch;
		}else if(needsDeletion && patch == patch->parent->children[0]){
			uint32_t index = atomicAdd(numNeedDeletion, 1);
			needDeletion[index] = patch->parent;
		}
	});

	grid.sync();

	if(grid.thread_rank() == 0 && *numNeedDeletion > 0) 
	printf("numNeedExpansion: %i, numNeedDeletion: %i \n", *numNeedExpansion, *numNeedDeletion);

	// if(grid.thread_rank() == 0){
	// 	printf("%i \n", patchPool->offset);
	// }

	processRange(*numNeedDeletion, [&](int index){
		Patch* patch = needDeletion[index];

		int numToDelete = 0;
		if(patch->children[0] != nullptr) numToDelete++;
		if(patch->children[1] != nullptr) numToDelete++;
		if(patch->children[2] != nullptr) numToDelete++;
		if(patch->children[3] != nullptr) numToDelete++;

		uint32_t poolIndex = atomicSub(&patchPool->offset, numToDelete) - numToDelete;

		printf("deleting patch with %i children. Target pool index is %i \n", numToDelete, poolIndex);

		int numDeleted = 0;
		for(int i = 0; i < 4; i++){

			Patch* child = patch->children[i];

			if(child == nullptr) continue;

			patch->children[i] = nullptr;
			patchPool->pointers[poolIndex + numDeleted] = child;

			numDeleted++;
		}


	});

	grid.sync();

	__shared__ int sh_storageIndex;

	for_blockwise(*numNeedExpansion, [&](int index){

		Patch* parent = needExpansion[index];

		if(block.thread_rank() == 0){
			uint32_t storageIndex = atomicAdd(&patchPool->offset, 4);
			sh_storageIndex = storageIndex;
		}

		block.sync();


		float2 min = parent->st_min;
		float2 max = parent->st_max;
		float2 ctr = (min + max) * 0.5f;

		{
			Patch* child = &patchStorage[sh_storageIndex + 0];
			child->st_min = {min.x, min.y};
			child->st_max = {ctr.x, ctr.y};
			child->parent = parent;
			child->children[0] = nullptr;
			child->children[1] = nullptr;
			child->children[2] = nullptr;
			child->children[3] = nullptr;
			child->modelID = uniforms.model;
			child->numPoints = POINTS_PER_PATCH;
			parent->children[0] = child;

			generatePatchSamples(child, models);
		}

		{
			Patch* child = &patchStorage[sh_storageIndex + 1];
			child->st_min = {ctr.x, min.y};
			child->st_max = {max.x, ctr.y};
			child->parent = parent;
			child->children[0] = nullptr;
			child->children[1] = nullptr;
			child->children[2] = nullptr;
			child->children[3] = nullptr;
			child->modelID = uniforms.model;
			child->numPoints = POINTS_PER_PATCH;
			parent->children[1] = child;

			generatePatchSamples(child, models);
		}

		{
			Patch* child = &patchStorage[sh_storageIndex + 2];
			child->st_min = {min.x, ctr.y};
			child->st_max = {ctr.x, max.y};
			child->parent = parent;
			child->children[0] = nullptr;
			child->children[1] = nullptr;
			child->children[2] = nullptr;
			child->children[3] = nullptr;
			child->modelID = uniforms.model;
			child->numPoints = POINTS_PER_PATCH;
			parent->children[2] = child;

			generatePatchSamples(child, models);
		}

		{
			Patch* child = &patchStorage[sh_storageIndex + 3];
			child->st_min = {ctr.x, ctr.y};
			child->st_max = {max.x, max.y};
			child->parent = parent;
			child->children[0] = nullptr;
			child->children[1] = nullptr;
			child->children[2] = nullptr;
			child->children[3] = nullptr;
			child->modelID = uniforms.model;
			child->numPoints = POINTS_PER_PATCH;
			parent->children[3] = child;

			generatePatchSamples(child, models);
		}

		block.sync();

		if(block.thread_rank() == 0){
			uint32_t nextPatchIndex = atomicAdd(next_numNodes, 4);

			next_patches[nextPatchIndex + 0] = parent->children[0];
			next_patches[nextPatchIndex + 1] = parent->children[1];
			next_patches[nextPatchIndex + 2] = parent->children[2];
			next_patches[nextPatchIndex + 3] = parent->children[3];
		}
	});

	grid.sync();

	processRange(*next_numNodes, [&](int index){
		patches[index] = next_patches[index];
	});

	*numPatches = *next_numNodes;

	// if(grid.thread_rank() == 0){
	// 	printf("%i \n", *numPatches);
	// }

}

extern "C" __global__
void kernel_rasterize(
	const Uniforms _uniforms,
	unsigned int* buffer,
	uint64_t* framebuffer, uint64_t* heatmap,
	Model* models, uint32_t* numModels,
	Patch** patches, uint32_t* numPatches,
	PatchPool* patchPool, Patch* patchStorage,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	Allocator _allocator(buffer, 0);

	uniforms = _uniforms;
	allocator = &_allocator;


	for_blockwise(*numPatches, [&](int patchIndex){

		// if(patchIndex == 0) return;

		Patch* patch = patches[patchIndex];

		for(
			uint32_t pointIndex = block.thread_rank();
			pointIndex < patch->numPoints;
			pointIndex += block.num_threads()
		){
			Point point = patch->points[pointIndex];
			float3 pos = point.pos;

			float4 ndc = uniforms.transform * float4{pos.x, pos.y, pos.z, 1.0f};
			ndc.x = ndc.x / ndc.w;
			ndc.y = ndc.y / ndc.w;
			ndc.z = ndc.z / ndc.w;

			float4 screenCoord = float4{
				(ndc.x * 0.5f + 0.5f) * uniforms.width,
				(ndc.y * 0.5f + 0.5f) * uniforms.height,
				ndc.z, ndc.w
			};

			// if(block.thread_rank() == 0 && uniforms.frameCount % 500 == 0){
			// 	printf("sizeof(Patch): %i \n", sizeof(Patch));
			// 	// printf("rasterize block: %i, %llu \n", grid.block_rank(), patch);
			// }

			// if(pointIndex < 32)
			// if(uniforms.frameCount % 500 == 0)
			// {
			// 	printf("[%4i] (%.1f, %.1f, %.1f) - %i \n", pointIndex, point.pos.x, point.pos.y, point.pos.z, point.rgba[0]);
			// 	// printf("%.1f, %.1f, %.1f - %i \n", point.pos.x, point.pos.y, point.pos.z, point.rgba[0]);
			// }

			// if(pointIndex == 1){
			// 	printf("%.1f, %.1f, %.1f - %i \n", point.pos.x, point.pos.y, point.pos.z, point.rgba[0]);
			// }

			uint32_t color = point.color;
			drawPoint(screenCoord, framebuffer, heatmap, color, uniforms);

			// break;
		}
	});



	// __shared__ int sh_patchIndex;
	// uint32_t* numPatchesProcessed = allocator->alloc<uint32_t*>(4);
	// *numPatchesProcessed = 0;

	// grid.sync();

	// while(true){

	// 	if(block.thread_rank() == 0){
	// 		uint32_t patchIndex = atomicAdd(numPatchesProcessed, 1);
	// 		sh_patchIndex = patchIndex;
	// 	}

	// 	block.sync();

	// 	if(sh_patchIndex >= *numPatches) break;

	// 	Patch* patch = patches[sh_patchIndex];

	// 	for(
	// 		uint32_t pointIndex = block.thread_rank();
	// 		pointIndex < patch->numPoints;
	// 		pointIndex += block.num_threads()
	// 	){
	// 		Point point = patch->points[pointIndex];
	// 		float3 pos = point.pos;

	// 		float4 ndc = uniforms.transform * float4{pos.x, pos.y, pos.z, 1.0f};
	// 		ndc.x = ndc.x / ndc.w;
	// 		ndc.y = ndc.y / ndc.w;
	// 		ndc.z = ndc.z / ndc.w;

	// 		float4 screenCoord = float4{
	// 			(ndc.x * 0.5f + 0.5f) * uniforms.width,
	// 			(ndc.y * 0.5f + 0.5f) * uniforms.height,
	// 			ndc.z, ndc.w
	// 		};

	// 		uint32_t color = 0x000000ff;
	// 		drawSprite(screenCoord, framebuffer, heatmap, color, uniforms);
	// 	}

	// }


}


extern "C" __global__
void kernel_clear_framebuffer(
	const Uniforms _uniforms,
	unsigned int* buffer,
	uint64_t* framebuffer, uint64_t* heatmap,
	Model* models, uint32_t* numModels,
	Patch** patches, uint32_t* numPatches,
	PatchPool* patchPool, Patch* patchStorage,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){

	auto grid = cg::this_grid();

	processRange(0, _uniforms.width * _uniforms.height, [&](int pixelIndex){
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
	});

}

#define MAX_HEATMAP_COLORS 11
constexpr uint32_t heatmapColors[MAX_HEATMAP_COLORS] = {
    0x9e0142,
    0xd53e4f,
    0xf46d43,
    0xfdae61,
    0xfee08b,
    0xffffbf,
    0xe6f598,
    0xabdda4,
    0x66c2a5,
    0x3288bd,
    0x5e4fa2
};

extern "C" __global__
void kernel_framebuffer_to_OpenGL(
	const Uniforms _uniforms,
	unsigned int* buffer,
	uint64_t* framebuffer, uint64_t* heatmap,
	Model* models, uint32_t* numModels,
	Patch** patches, uint32_t* numPatches,
	PatchPool* patchPool, Patch* patchStorage,
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


