#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

// ray tracing adapted from tutorial: https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
// author: Alan Wolfe
// (MIT LICENSE)

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

namespace cg = cooperative_groups;

Uniforms uniforms;
Allocator* allocator;
uint64_t nanotime_start;

constexpr float PI = 3.1415;
constexpr uint32_t BACKGROUND_COLOR = 0x00332211ull;

float intersect_sphere(float3 origin, float3 direction, float3 spherePos, float sphereRadius) {

	float3 CO = origin - spherePos;
	float a = dot(direction, direction);
	float b = 2.0f * dot(direction, CO);
	float c = dot(CO, CO) - sphereRadius * sphereRadius;
	float delta = b * b - 4.0f * a * c;
	
	if(delta < 0.0) {
		return -1.0;
	}

	float t = (-b - sqrt(delta)) / (2.0f * a);

	// return true;

	return t;
}

extern "C" __global__
void kernel(
	const Uniforms _uniforms,
	unsigned int* buffer,
	unsigned int* voxelBuffer,
	cudaSurfaceObject_t gl_colorbuffer
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime_start));

	uniforms = _uniforms;

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	// allocate framebuffer memory
	int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
	uint64_t* framebuffer = allocator->alloc<uint64_t*>(framebufferSize);
	uint64_t* framebuffer_2 = allocator->alloc<uint64_t*>(framebufferSize);

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
		framebuffer_2[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
	});

	grid.sync();

	// PROJECT VOXELS TO PIXELS
	uint8_t* voxelBuffer_u8 = (uint8_t*)voxelBuffer;
	processRange(uniforms.numVoxels, [&](int index){


		int X = voxelBuffer_u8[6 * index + 0];
		int Y = voxelBuffer_u8[6 * index + 1];
		int Z = voxelBuffer_u8[6 * index + 2];
		int r = voxelBuffer_u8[6 * index + 3];
		int g = voxelBuffer_u8[6 * index + 4];
		int b = voxelBuffer_u8[6 * index + 5];

		float x = float(X);
		float y = float(Y);
		float z = float(Z);

		mat4 transform = uniforms.proj * uniforms.view;

		float4 ndc = transform * float4{x, y, z, 1.0f};
		ndc.x = ndc.x / ndc.w;
		ndc.y = ndc.y / ndc.w;
		ndc.z = ndc.z / ndc.w;

		float2 imgPos = {
			(ndc.x * 0.5f + 0.5f) * uniforms.width, 
			(ndc.y * 0.5f + 0.5f) * uniforms.height,
		};

		uint32_t color = 0;
		uint8_t* rgba = (uint8_t*)&color;

		rgba[0] = r;
		rgba[1] = g;
		rgba[2] = b;

		float depth = ndc.w;
		uint64_t udepth = *((uint32_t*)&depth);
		uint64_t pixel = (udepth << 32ull) | color;


		int2 pixelCoords = make_int2(imgPos.x, imgPos.y);
		int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;
		pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

		atomicMin(&framebuffer[pixelID], pixel);


	});

	grid.sync();

	int window = 0;

	// WIDEN HORIZONTALY
	int numPixels = uniforms.width * uniforms.height;
	processRange(uniforms.width * uniforms.height, [&](int pixelIndex){

		int x = pixelIndex % int(uniforms.width);
		int y = pixelIndex / int(uniforms.width);
		int pixelID = x + uniforms.width * y;

		uint64_t closestFragment = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);

		for(int dx = -window; dx <= +window; dx++)
		{
			int neighborPixelID = (x + dx) + uniforms.width * y;
			neighborPixelID = clamp(neighborPixelID, 0, numPixels);

			uint64_t fragment = framebuffer[neighborPixelID];

			closestFragment = min(closestFragment, fragment);
		}

		framebuffer_2[pixelID] = closestFragment;
	});

	grid.sync();

	// WIDEN VERTICALLY
	processRange(uniforms.width * uniforms.height, [&](int pixelIndex){

		int x = pixelIndex % int(uniforms.width);
		int y = pixelIndex / int(uniforms.width);
		int pixelID = x + uniforms.width * y;

		uint64_t closestFragment = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);

		for(int dy = -window; dy <= +window; dy++)
		{
			int neighborPixelID = x + uniforms.width * (y + dy);
			neighborPixelID = clamp(neighborPixelID, 0, numPixels);

			uint64_t fragment = framebuffer_2[neighborPixelID];

			closestFragment = min(closestFragment, fragment);
		}

		framebuffer[pixelID] = closestFragment;
	});

	grid.sync();

	// WIDEN VERTICALLY
	processRange(uniforms.width * uniforms.height, [&](int pixelIndex){

		int x = pixelIndex % int(uniforms.width);
		int y = uniforms.height - (pixelIndex / int(uniforms.width));
		int pixelID = x + uniforms.width * y;

		float u = float(x) / uniforms.width;
		float v = float(y) / uniforms.height;

		float aspectRatio = uniforms.width / uniforms.height;

		float3 rayPosition = {0.0, 0.0, 0.0};
		float3 rayTarget = {
			2.0f * u - 1.0f, 
			1.0f,
			2.0f * v - 1.0f, 
		};
		// rayTarget.y /= aspectRatio;

		float4 rayPos4 = uniforms.viewInverse * float4{0.0, 0.0, 0.0, 1.0};
		rayPosition.x = rayPos4.x;
		rayPosition.y = rayPos4.y;
		rayPosition.z = rayPos4.z;

		float4 rayTarget4 = uniforms.viewInverse * float4{
			2.0f * u - 1.0f, 
			1.0f, 
			(2.0f * v - 1.0f) / aspectRatio, 
			1.0f};
		rayTarget.x = rayTarget4.x;
		rayTarget.y = rayTarget4.y;
		rayTarget.z = rayTarget4.z;


		// float4 rayWorldPos = uniforms.view * rayPosition;

		float3 spherePos = {0.0, 0.0, 0.0};
		float sphereRadius = 0.1f;

		float4 spherePosView = uniforms.view * float4{spherePos.x, spherePos.y, spherePos.z, 1.0};
		// spherePos.x = spherePosView.x;
		// spherePos.y = spherePosView.y;
		// spherePos.z = spherePosView.z;

		float3 rayDir = normalize(rayTarget - rayPosition);
		// rayDir.x *= 1.0;
		// rayDir.y *= 1.0;
		// rayDir.z *= 1.0;

		if(x == 1 && y == 500){
			// printf("rayPosition: %f, %f, %f \n", rayPosition.x, rayPosition.y, rayPosition.z);
			// printf("rayTarget: %f, %f, %f \n", rayTarget.x, rayTarget.y, rayTarget.z);
			printf("rayDir: %f, %f, %f \n", rayDir.x, rayDir.y, rayDir.z);
			// printf("spherePos: %f, %f, %f \n", spherePos.x, spherePos.y, spherePos.z);
		}



		float t = intersect_sphere(rayPosition, rayDir, spherePos, sphereRadius);


		uint32_t color;
		uint8_t* rgba = (uint8_t*)&color;

		rgba[0] = 255.0 * rayTarget.x;
		rgba[1] = 255.0 * rayTarget.y;
		rgba[2] = 255.0 * rayTarget.z;

		// rgba[0] = 255.0 * u;
		// rgba[1] = 255.0 * v;
		// rgba[2] = 0.0;

		// if(t > 0.0f){
		// 	color = 0x0000ff00;

		// 	uint64_t fragment = (uint64_t(Infinity) << 32ull) | uint64_t(color);
		// 	framebuffer[pixelID] = fragment;
		// }else{
		// 	color = 0x000000ff;
		// }

		uint64_t fragment = (uint64_t(Infinity) << 32ull) | uint64_t(color);
		framebuffer[pixelID] = fragment;

			// color = 0x000000ff;

		
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
