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
	int cells = 2'000;
	processRange(0, cells * cells, [&](int index){
		int ux = index % cells;
		int uy = index / cells;

		float u = float(ux) / float(cells - 1);
		float v = float(uy) / float(cells - 1);

		float4 pos = {
			5.0 * (u - 0.5), 
			5.0 * (v - 0.5), 
			0.0f, 
			1.0f};

		float4 ndc = uniforms.transform * pos;
		ndc.x = ndc.x / ndc.w;
		ndc.y = ndc.y / ndc.w;
		ndc.z = ndc.z / ndc.w;
		float depth = ndc.w;

		int x = (ndc.x * 0.5 + 0.5) * uniforms.width;
		int y = (ndc.y * 0.5 + 0.5) * uniforms.height;

		uint32_t R = 255.0f * u;
		uint32_t G = 255.0f * v;
		uint32_t B = 0;
		uint64_t color = R | (G << 8) | (B << 16);

		if(x > 1 && x < uniforms.width  - 2.0)
		if(y > 1 && y < uniforms.height - 2.0){

			for(int ox : {-2, -1, 0, 1, 2})
			for(int oy : {-2, -1, 0, 1, 2}){
				uint32_t pixelID = (x + ox) + int(uniforms.width) * (y + oy);
				uint64_t udepth = *((uint32_t*)&depth);
				uint64_t encoded = (udepth << 32) | color;

				atomicMin(&framebuffer[pixelID], encoded);
			}
		}
	});

	// draw sphere
	int s = 10'000;
	float rounds = 20.0;
	processRange(0, s, [&](int index){
		float u = float(index) / float(s - 1);

		float z = 2.0 * u - 1.0;
		float a = cos(0.5 * PI * z);
		a = sqrt(1.0f - abs(z * z));

		float r = 0.5;

		float4 pos = {
			r * a * sin(rounds * PI * u), 
			r * a * cos(rounds * PI * u), 
			r * z + 0.3, 
			1.0f};

		float4 ndc = uniforms.transform * pos;
		ndc.x = ndc.x / ndc.w;
		ndc.y = ndc.y / ndc.w;
		ndc.z = ndc.z / ndc.w;
		float depth = ndc.w;

		int x = (ndc.x * 0.5 + 0.5) * uniforms.width;
		int y = (ndc.y * 0.5 + 0.5) * uniforms.height;

		uint32_t R = 255.0f * u;
		uint32_t G = 0;
		uint32_t B = 0;
		uint64_t color = R | (G << 8) | (B << 16);

		if(x > 1 && x < uniforms.width  - 2.0)
		if(y > 1 && y < uniforms.height - 2.0){

			for(int ox : {-2, -1, 0, 1, 2})
			for(int oy : {-2, -1, 0, 1, 2}){
				uint32_t pixelID = (x + ox) + int(uniforms.width) * (y + oy);
				uint64_t udepth = *((uint32_t*)&depth);
				uint64_t encoded = (udepth << 32) | color;

				atomicMin(&framebuffer[pixelID], encoded);
			}
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
