#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

// ray tracing adapted from tutorial: https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
// author: Alan Wolfe
// (MIT LICENSE)

struct Point{
	float x, y, z;
	uint32_t color;
};

float4 operator*(const mat4& a, const float4& b){
	return make_float4(
		dot(a.rows[0], b),
		dot(a.rows[1], b),
		dot(a.rows[2], b),
		dot(a.rows[3], b)
	);
}

float3 operator*(const mat4& a, const float3& b){
	// return make_float4(
	// 	dot(a.rows[0], b),
	// 	dot(a.rows[1], b),
	// 	dot(a.rows[2], b),
	// 	dot(a.rows[3], b)
	// );

	return float3{
		a.rows[0].x * b.x + a.rows[0].y * b.y + a.rows[0].z * b.z + a.rows[0].w,
		a.rows[1].x * b.x + a.rows[1].y * b.y + a.rows[1].z * b.z + a.rows[1].w,
		a.rows[2].x * b.x + a.rows[2].y * b.y + a.rows[2].z * b.z + a.rows[2].w
	};
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
constexpr uint64_t DEFAULT_FRAGMENT = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);

// float intersect_sphere(float3 origin, float3 direction, float3 spherePos, float sphereRadius) {

// 	float3 CO = origin - spherePos;
// 	float a = dot(direction, direction);
// 	float b = 2.0f * dot(direction, CO);
// 	float c = dot(CO, CO) - sphereRadius * sphereRadius;
// 	float delta = b * b - 4.0f * a * c;
	
// 	if(delta < 0.0) {
// 		return -1.0;
// 	}

// 	float t = (-b - sqrt(delta)) / (2.0f * a);

// 	return t;
// }

float intersect_sphere(float3 ro, float3 rd, float3 ce, float ra) {

	float3 oc = ro - ce;
	float b = dot( oc, rd );
	float c = dot( oc, oc ) - ra*ra;
	float h = b * b - c;

	if(h < 0.0 ){
		// no intersection
		return -1.0;
	} 

	h = sqrt(h);

	return min(-b - h, -b + h);

	// return vec2(-b - h, -b + h);
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

	uint32_t numPoints = 1'000'000;
	Point* points = allocator->alloc<Point*>(16 * numPoints);

	curandStateXORWOW_t thread_random_state;
	curand_init(grid.thread_rank(), 0, 0, &thread_random_state);



	processRange(numPoints, [&](int index){

		uint32_t X = curand(&thread_random_state);
		uint32_t Y = curand(&thread_random_state);
		uint32_t Z = curand(&thread_random_state);

		float x = float(X % 2'000'000) - 1'000'000.0;
		float y = float(Y % 2'000'000) - 1'000'000.0;
		float z = float(Z % 2'000'000) - 1'000'000.0;

		float3 spherePos = float3{x, y, z};
		spherePos = normalize(spherePos);

		float scale = 20.0;
		float3 pos = {40.0, 30.0, 100.0};

		points[index].x = scale * spherePos.x + pos.x;
		points[index].y = scale * spherePos.y + pos.y;
		points[index].z = scale * spherePos.z + pos.z;
		points[index].color = 0x000000ff;
	});

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
		framebuffer_2[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
	});

	grid.sync();

	// PROJECT POINTS TO PIXELS
	// if(false)
	processRange(numPoints, [&](int index){

		Point point = points[index];
		mat4 transform = uniforms.proj * uniforms.view;

		float4 ndc = transform * float4{point.x, point.y, point.z, 1.0f};
		ndc.x = ndc.x / ndc.w;
		ndc.y = ndc.y / ndc.w;
		ndc.z = ndc.z / ndc.w;

		if(ndc.w <= 0.0) return;
		if(ndc.x < -1.0) return;
		if(ndc.x >  1.0) return;
		if(ndc.y < -1.0) return;
		if(ndc.y >  1.0) return;

		float2 imgPos = {
			(ndc.x * 0.5f + 0.5f) * uniforms.width, 
			(ndc.y * 0.5f + 0.5f) * uniforms.height,
		};

		uint32_t color = point.color;
		uint8_t* rgba = (uint8_t*)&color;

		float depth = ndc.w;
		uint64_t udepth = *((uint32_t*)&depth);

		// uint64_t pixel = (udepth << 32ull) | index;
		uint64_t pixel = (udepth << 32ull) | color;


		int2 pixelCoords = make_int2(imgPos.x, imgPos.y);
		int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;
		pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

		atomicMin(&framebuffer_2[pixelID], pixel);


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

		if(ndc.w <= 0.0) return;
		if(ndc.x < -1.0) return;
		if(ndc.x >  1.0) return;
		if(ndc.y < -1.0) return;
		if(ndc.y >  1.0) return;

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

		uint64_t pixel = (udepth << 32ull) | index;
		// uint64_t pixel = (udepth << 32ull) | color;


		int2 pixelCoords = make_int2(imgPos.x, imgPos.y);
		int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;
		pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

		atomicMin(&framebuffer[pixelID], pixel);


	});

	grid.sync();

	int window = 1;

	// // WIDEN HORIZONTALY
	// int numPixels = uniforms.width * uniforms.height;
	// processRange(uniforms.width * uniforms.height, [&](int pixelIndex){

	// 	int x = pixelIndex % int(uniforms.width);
	// 	int y = pixelIndex / int(uniforms.width);
	// 	int pixelID = x + uniforms.width * y;

	// 	uint64_t closestFragment = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);

	// 	for(int dx = -window; dx <= +window; dx++)
	// 	{
	// 		int neighborPixelID = (x + dx) + uniforms.width * y;
	// 		neighborPixelID = clamp(neighborPixelID, 0, numPixels);

	// 		uint64_t fragment = framebuffer[neighborPixelID];

	// 		closestFragment = min(closestFragment, fragment);
	// 	}

	// 	framebuffer_2[pixelID] = closestFragment;
	// });

	// grid.sync();

	// // WIDEN VERTICALLY
	// processRange(uniforms.width * uniforms.height, [&](int pixelIndex){

	// 	int x = pixelIndex % int(uniforms.width);
	// 	int y = pixelIndex / int(uniforms.width);
	// 	int pixelID = x + uniforms.width * y;

	// 	uint64_t closestFragment = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);

	// 	for(int dy = -window; dy <= +window; dy++)
	// 	{
	// 		int neighborPixelID = x + uniforms.width * (y + dy);
	// 		neighborPixelID = clamp(neighborPixelID, 0, numPixels);

	// 		uint64_t fragment = framebuffer_2[neighborPixelID];

	// 		closestFragment = min(closestFragment, fragment);
	// 	}

	// 	framebuffer[pixelID] = closestFragment;
	// });

	// grid.sync();

	// // WIDEN VERTICALLY
	// processRange(uniforms.width * uniforms.height, [&](int pixelIndex){

	// 	int x = pixelIndex % int(uniforms.width);
	// 	int y = pixelIndex / int(uniforms.width);
	// 	int pixelID = x + uniforms.width * y;

	// 	framebuffer_2[pixelID] = framebuffer[pixelID];
	// 	// framebuffer[pixelID] = closestFragment;
	// });

	grid.sync();

	// try tracing points in vicinity
	if(false)
	processRange(uniforms.width * uniforms.height, [&](int pixelIndex){

		float aspectRatio = uniforms.width / uniforms.height;
		int numPixels = uniforms.width * uniforms.height;
		
		int x = pixelIndex % int(uniforms.width);
		int y = uniforms.height - (pixelIndex / int(uniforms.width));

		float closestT = Infinity;
		bool nonePicked = true;
		uint32_t pickedVoxel = -1;
		uint32_t pickedColor = 0;
		uint8_t* pickedRGBA  = (uint8_t*)&pickedColor;

		int window = 5;
		for(int dx = -window; dx <= window; dx++)
		for(int dy = -window; dy <= window; dy++)
		{
			int pixelID = (x + dx) + uniforms.width * (y + dy);

			if(pixelID >= numPixels) continue;

			uint64_t neighbor = framebuffer[pixelID];
			uint32_t udepth = (neighbor >> 32);
			uint32_t neighborIndex = neighbor & 0xffffffff;

			if(neighbor == DEFAULT_FRAGMENT) continue;
			if(neighborIndex >= uniforms.numVoxels) continue;

			int X = voxelBuffer_u8[6 * neighborIndex + 0];
			int Y = voxelBuffer_u8[6 * neighborIndex + 1];
			int Z = voxelBuffer_u8[6 * neighborIndex + 2];
			int r = voxelBuffer_u8[6 * neighborIndex + 3];
			int g = voxelBuffer_u8[6 * neighborIndex + 4];
			int b = voxelBuffer_u8[6 * neighborIndex + 5];

			float u = float(x + dx) / uniforms.width;
			float v = float(y + dy) / uniforms.height;

			float4 rayPos4     = uniforms.viewInverse * float4{0.0, 0.0, 0.0, 1.0};
			float3 rayPosition = make_float3(rayPos4);
			float3 rayTarget   = uniforms.viewInverse * float3{
				2.0f * u - 1.0f, 
				(2.0f * v - 1.0f) / aspectRatio, 
				-0.8,
			};

			// float3 spherePos = {
			// 	float(X) + 0.5, 
			// 	float(Y), 
			// 	float(Z),
			// };
			// float sphereRadius = 0.2f;

			float sphereRadius = 1.1f;
			float3 spherePos = {40.0, 30.0, 100.0};

			float3 rayDir = normalize(rayTarget - rayPosition);
			float t = intersect_sphere(rayPosition, rayDir, spherePos, sphereRadius);

			nonePicked = false;
			pickedVoxel = neighborIndex;
			pickedColor = r | (g << 8) | (b << 16);

			if(t > 0.0){

				// float3 I = rayPosition + t * rayDir;

				// mat4 transform = uniforms.proj * uniforms.view;

				// float4 ndc = transform * float4{I.x, I.y, I.z, 1.0f};
				// ndc.x = ndc.x / ndc.w;
				// ndc.y = ndc.y / ndc.w;
				// ndc.z = ndc.z / ndc.w;

				// float2 imgPos = {
				// 	(ndc.x * 0.5f + 0.5f) * uniforms.width, 
				// 	(ndc.y * 0.5f + 0.5f) * uniforms.height,
				// };



				closestT = min(closestT, t);
				pickedColor = 0x0000ffff;
			}

			// pickedRGBA[0] = 255.0 * u;
			// pickedRGBA[1] = 255.0 * v;
			// pickedRGBA[2] = 0;

			// if(u > 0.99){
			// 	pickedRGBA[2] = 255.0;
			// }

			// if(t > 0.0f){
			// 	uint32_t color = 0x0000ff00;

			// 	uint64_t fragment = (uint64_t(Infinity) << 32ull) | uint64_t(color);
			// 	framebuffer_2[pixelID] = fragment;
			// }else{
			// 	uint32_t color = 0x000000ff;
			// 	uint8_t* rgba = (uint8_t*)&color;

			// 	// rgba[0] = r;
			// 	// rgba[1] = g;
			// 	// rgba[2] = b;

			// 	// color = neighborIndex * 123;

			// 	uint64_t fragment = (uint64_t(Infinity) << 32ull) | uint64_t(color);
			// 	framebuffer_2[pixelID] = fragment;
			// }

		}

		int targetpixelID = x + uniforms.width * y;

		if(!nonePicked)
		{
			uint32_t color = 0x000000ff;
			uint8_t* rgba = (uint8_t*)&color;
			// rgba[0] = 255.0 * u;
			// rgba[1] = 255.0 * v;

			color = pickedColor;

			uint64_t udepth = *((uint32_t*)&closestT);
			uint64_t fragment = (uint64_t(Infinity) << 32ull) | uint64_t(color);
			framebuffer_2[targetpixelID] = fragment;
		}
	});

	grid.sync();

	// // Ray-Cast a sphere
	// if(false)
	// processRange(uniforms.width * uniforms.height, [&](int pixelIndex){

	// 	float aspectRatio = uniforms.width / uniforms.height;

	// 	int x = pixelIndex % int(uniforms.width);
	// 	int y = uniforms.height - (pixelIndex / int(uniforms.width));
	// 	int pixelID = x + uniforms.width * y;

	// 	float u = float(x) / uniforms.width;
	// 	float v = float(y) / uniforms.height;

	// 	float4 rayPos4     = uniforms.viewInverse * float4{0.0, 0.0, 0.0, 1.0};
	// 	float3 rayPosition = make_float3(rayPos4);
	// 	float3 rayTarget   = uniforms.viewInverse * float3{
	// 		2.0f * u - 1.0f, 
	// 		(2.0f * v - 1.0f) / aspectRatio, 
	// 		-0.8,
	// 	};

	// 	float4 cameraDir4 = uniforms.viewInverse * float4{0.0, 0.0, -1.0f, 1.0f};
	// 	float3 cameraDir = normalize(make_float3(cameraDir4));
		
	// 	float sphereRadius = 20.0f;
	// 	float3 spherePos = {40.0, 30.0, 100.0};

	// 	float3 rayDir = normalize(rayTarget - rayPosition);
	// 	float t = intersect_sphere(rayPosition, rayDir, spherePos, sphereRadius);

	// 	uint32_t color;
	// 	uint8_t* rgba = (uint8_t*)&color;

	// 	rgba[0] = 255.0 * rayTarget.x;
	// 	rgba[1] = 255.0 * rayTarget.y;
	// 	rgba[2] = 255.0 * rayTarget.z;

	// 	if(t > 0.0f){
	// 		color = 0x0000ff00;

	// 		float3 I = rayPosition + t * rayDir;

	// 		mat4 transform = uniforms.proj * uniforms.view;

	// 		float4 ndc = transform * float4{I.x, I.y, I.z, 1.0f};
	// 		ndc.x = ndc.x / ndc.w;
	// 		ndc.y = ndc.y / ndc.w;
	// 		ndc.z = ndc.z / ndc.w;

	// 		float2 imgPos = {
	// 			(ndc.x * 0.5f + 0.5f) * uniforms.width, 
	// 			(ndc.y * 0.5f + 0.5f) * uniforms.height,
	// 		};

	// 		float depth = ndc.w;
	// 		uint64_t udepth = *((uint32_t*)&depth);

	// 		uint64_t pixel = (udepth << 32ull) | 0x0000ff00;

	// 		int2 pixelCoords = make_int2(imgPos.x, imgPos.y);
	// 		int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;
	// 		pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

	// 		atomicMin(&framebuffer_2[pixelID], pixel);
	// 	}
	// });

	grid.sync();

	// Ray-Cast a sphere, method 2
	// if(false)
	processRange(uniforms.width * uniforms.height, [&](int pixelID){

		int x = pixelID % int(uniforms.width);
		int y = uniforms.height - (pixelID / int(uniforms.width));

		float4 ndc = {
			2.0f * float(x + 0.5f) / uniforms.width - 1.0f,
			2.0f * float(y + 0.5f) / uniforms.height - 1.0f,
			0.0f, 1.0f
		};

		float4 pixelViewDir = uniforms.projInverse * ndc;
		pixelViewDir = pixelViewDir / pixelViewDir.w;
		float4 pixelWorldDir = uniforms.viewInverse * pixelViewDir;

		float4 rayPos4     = uniforms.viewInverse * float4{0.0, 0.0, 0.0, 1.0};
		float3 rayPosition = make_float3(rayPos4);
		float3 rayTarget = make_float3(pixelWorldDir);

		float3 cameraTarget = uniforms.viewInverse * float3{0.0, 0.0, -1.0f};
		float3 cameraDir = normalize(cameraTarget - rayPosition);
		// float3 cameraDir = normalize(make_float3(cameraDir4));
		
		float sphereRadius = 20.0f;
		float3 spherePos = {40.0, 30.0, 100.0};

		float3 rayDir = normalize(rayTarget - rayPosition);
		float t = intersect_sphere(rayPosition, rayDir, spherePos, sphereRadius);

		uint32_t color;
		uint8_t* rgba = (uint8_t*)&color;

		// if(x == uniforms.width / 2.0 && y == uniforms.height / 2.0){
		// 	printf("rayDir: %f, %f, %f \n", rayDir.x, rayDir.y, rayDir.z);
		// 	printf("cameraDir: %f, %f, %f \n", cameraDir.x, cameraDir.y, cameraDir.z);
		// }


		if(t > 0.0f){
			color = 0x0000ff00;

			// t is actual distance to intersection.
			// However, because the rasterized scene uses distance on central view-dir "z",
			// we need to transform t to "z" by projecting t * rayDir onto cameraDir.

			// float depth = t;
			float z = dot(t * rayDir, cameraDir);
			float depth = z;

			uint64_t udepth = *((uint32_t*)&depth);

			uint64_t pixel = (udepth << 32ull) | 0x0000ffff;

			int pixelID = x + y * uniforms.width;
			pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

			atomicMin(&framebuffer_2[pixelID], pixel);
		}
	});

	grid.sync();

	// transfer framebuffer to opengl texture
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){

		int x = pixelIndex % int(uniforms.width);
		int y = pixelIndex / int(uniforms.width);

		struct Fragment{
			uint32_t color;
			float depth;
		};

		Fragment fragment = ((Fragment*)framebuffer_2)[pixelIndex];

		uint64_t encoded = framebuffer_2[pixelIndex];
		// uint32_t color = encoded & 0xffffffffull;
		uint32_t color = fragment.color;
		// color = fragment.depth * 0.5;

		surf2Dwrite(color, gl_colorbuffer, x * 4, y);
	});


}
