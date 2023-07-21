#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

namespace cg = cooperative_groups;

Uniforms uniforms;

constexpr float PI = 3.1415;

float4 operator*(const mat4& a, const float4& b){
	return make_float4(
		dot(a.rows[0], b),
		dot(a.rows[1], b),
		dot(a.rows[2], b),
		dot(a.rows[3], b)
	);
}


float3 sample(float u, float v, float level){
	
	float x = u;
	float y = v;

	float factor = pow(2.0f, level);
	float offset = (fmodf(level, 2.0) == 0) ? 0.0 : 1.0;

	u = factor * u;
	v = factor * v;

	float fu = fmodf(u, 1.0f) * 2.0 - 1.0;
	float fv = fmodf(v, 1.0f) * 2.0 - 1.0;

	float d = sqrt(fu * fu + fv * fv);

	float c = 0.3;
	float z = 0.3 * exp(-(d * d) / (2 * c * c)) / pow(2.0f, level);

	return {x, y, z};
};

float3 sample_1(float u, float v, float level){

	float epsilon = 0.001;
	float3 p_0 = sample(u, v, level);
	float3 tx_0 = sample(u + epsilon, v , level) - p_0;
	float3 ty_0 = sample(u, v + epsilon, level) - p_0;
	float3 N_0 = normalize(cross(tx_0, ty_0));

	float3 off_1 = sample(u, v, 3.0);
	float3 p_1 = p_0 + N_0 * off_1.z;

	return p_1;
}

float3 sample_2(float u, float v, float level){

	float epsilon = 0.001;
	float3 p_0 = sample_1(u, v, 0.0);
	float3 tx_0 = sample_1(u + epsilon, v , 0.0) - p_0;
	float3 ty_0 = sample_1(u, v + epsilon, 0.0) - p_0;
	float3 N_0 = normalize(cross(tx_0, ty_0));

	float3 off_1 = sample(u, v, 4.0);
	float3 p_1 = p_0 + N_0 * off_1.z;

	return p_1;
}

float3 sample(float u, float v){
	return sample_1(u, v, 0.0);

	// return {u, v, 0.0};
}

extern "C" __global__
void kernel(
	const Uniforms _uniforms,
	unsigned int* buffer,
	cudaSurfaceObject_t gl_colorbuffer
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uniforms = _uniforms;

	Allocator allocator(buffer, 0);

	// allocate framebuffer memory
	int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
	uint64_t* framebuffer = allocator.alloc<uint64_t*>(framebufferSize);

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		// depth:            7f800000 (Infinity)
		// background color: 00332211 (aabbggrr)
		framebuffer[pixelIndex] = 0x7f800000'00332211ull;;
	});

	grid.sync();

	

	// draw plane
	int cells = 2000;
	processRange(0, cells * cells, [&](int index){
		int ux = index % cells;
		int uy = index / cells;

		float u = float(ux) / float(cells - 1);
		float v = float(uy) / float(cells - 1);

		// u = 2.0 * u - 1.0;
		// v = 2.0 * v - 1.0;

		float3 pos = sample(u, v);

		float3 tx = sample(u + 0.01, v + 0.00) - pos;
		float3 ty = sample(u + 0.00, v + 0.01) - pos;

		float3 N = normalize(cross(tx, ty));

		float3 lightpos = {-10.0f, 10.0f, 10.0f};
		float3 L = normalize(lightpos - pos);

		float diffuse = dot(N, L);

		float4 ndc = uniforms.transform * float4{pos.x, pos.y, pos.z, 1.0};
		ndc.x = ndc.x / ndc.w;
		ndc.y = ndc.y / ndc.w;
		ndc.z = ndc.z / ndc.w;
		float depth = ndc.w;

		int x = (ndc.x * 0.5 + 0.5) * uniforms.width;
		int y = (ndc.y * 0.5 + 0.5) * uniforms.height;

		uint32_t R = 255.0f * u;
		uint32_t G = 255.0f * v;
		uint32_t B = 0;
		R = diffuse * 255.0 ;
		G = diffuse * 255.0 ;
		B = diffuse * 255.0 ;
		uint64_t color = R | (G << 8) | (B << 16);

		if(x > 1 && x < uniforms.width  - 2.0)
		if(y > 1 && y < uniforms.height - 2.0){

			// SINGLE PIXEL
			uint32_t pixelID = x + int(uniforms.width) * y;
			uint64_t udepth = *((uint32_t*)&depth);
			uint64_t encoded = (udepth << 32) | color;

			atomicMin(&framebuffer[pixelID], encoded);

			// POINT SPRITE
			// for(int ox : {-2, -1, 0, 1, 2})
			// for(int oy : {-2, -1, 0, 1, 2}){
			// 	uint32_t pixelID = (x + ox) + int(uniforms.width) * (y + oy);
			// 	uint64_t udepth = *((uint32_t*)&depth);
			// 	uint64_t encoded = (udepth << 32) | color;

			// 	atomicMin(&framebuffer[pixelID], encoded);
			// }
		}
	});

	grid.sync();

	// transfer framebuffer to opengl texture
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){

		int x = pixelIndex % int(uniforms.width);
		int y = pixelIndex / int(uniforms.width);

		uint64_t encoded = framebuffer[pixelIndex];
		uint32_t color = encoded & 0xffffffffull;

		surf2Dwrite(color, gl_colorbuffer, x * 4, y);
	});


}
