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

// see https://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/sld017.htm
float intersect_plane(float3 origin, float3 direction, float3 N, float d) {
	float t = -(dot(origin, N) + d) / dot(direction, N);

	return t;
}

float intersect_cube(float3 origin, float3 direction, float3 pos, float size){

	auto grid = cg::this_grid();

	float t0 = intersect_plane(origin, direction, { 1.0f,  0.0f,  0.0f}, -pos.x + 0.5f * size);
	float t1 = intersect_plane(origin, direction, { 1.0f,  0.0f,  0.0f}, -pos.x - 0.5f * size);
	float t2 = intersect_plane(origin, direction, { 0.0f,  1.0f,  0.0f}, -pos.y + 0.5f * size);
	float t3 = intersect_plane(origin, direction, { 0.0f,  1.0f,  0.0f}, -pos.y - 0.5f * size);
	float t4 = intersect_plane(origin, direction, { 0.0f,  0.0f,  1.0f}, -pos.z + 0.5f * size);
	float t5 = intersect_plane(origin, direction, { 0.0f,  0.0f,  1.0f}, -pos.z - 0.5f * size);

	float t01 = min(t0, t1);
	float t23 = min(t2, t3);
	float t45 = min(t4, t5);

	float txf, txb;
	float tyf, tyb;
	float tzf, tzb;

	if(direction.x < 0.0){
		txf = t1;
		txb = t0;
	}else{
		txf = t0;
		txb = t1;
	}

	if(direction.y < 0.0){
		tyf = t3;
		tzb = t2;
	}else{
		tyf = t2;
		tyb = t3;
	}

	if(direction.z < 0.0){
		tzf = t5;
		tzb = t4;
	}else{
		tzf = t4;
		tzb = t5;
	}

	float t = max(max(txf, tyf), tzf);

	float epsilon = 0.0001f;

	float3 I = origin + t * direction;

	if(I.x < pos.x - 0.5f * size - epsilon) t = 0.0;
	if(I.x > pos.x + 0.5f * size + epsilon) t = 0.0;
	if(I.y < pos.y - 0.5f * size - epsilon) t = 0.0;
	if(I.y > pos.y + 0.5f * size + epsilon) t = 0.0;
	if(I.z < pos.z - 0.5f * size - epsilon) t = 0.0;
	if(I.z > pos.z + 0.5f * size + epsilon) t = 0.0;


	return t;
}

// float intersect_cube(float3 origin, float3 direction, float3 pos, float size){

// 	float t0 = intersect_plane(origin, direction, { 1.0f,  0.0f,  0.0f}, 0.5f * size);
// 	float t1 = intersect_plane(origin, direction, {-1.0f,  0.0f,  0.0f}, 0.5f * size);
// 	float t2 = intersect_plane(origin, direction, { 0.0f,  1.0f,  0.0f}, 0.5f * size);
// 	float t3 = intersect_plane(origin, direction, { 0.0f, -1.0f,  0.0f}, 0.5f * size);
// 	float t4 = intersect_plane(origin, direction, { 0.0f,  0.0f,  1.0f}, 0.5f * size);
// 	float t5 = intersect_plane(origin, direction, { 0.0f,  0.0f, -1.0f}, 0.5f * size);

	
// 	float t01 = min(t0, t1);
// 	float t23 = min(t2, t3);
// 	float t45 = min(t4, t5);

// 	float t = min(min(t01, t23), min(t4, t5));

// 	float3 I = origin + t * direction;

// 	// float epsilon = 0.0001f;
// 	// bool insideX = (I.x + epsilon >= pos.x - 0.5f * size) && (I.x - epsilon <= pos.x + 0.5f * size);
// 	// bool insideY = (I.y + epsilon >= pos.y - 0.5f * size) && (I.y - epsilon <= pos.y + 0.5f * size);
// 	// bool insideZ = (I.z + epsilon >= pos.z - 0.5f * size) && (I.z - epsilon <= pos.z + 0.5f * size);

// 	// if(!insideX) t = 0.0;
// 	// if(!insideY) t = 0.0;
// 	// if(!insideZ) t = 0.0;


// 	return t;
// }

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

	return t;
}

// The iniquo quilez way:
// float intersect_sphere(float3 ro, float3 rd, float3 ce, float ra) {

// 	float3 oc = ro - ce;
// 	float b = dot( oc, rd );
// 	float c = dot( oc, oc ) - ra*ra;
// 	float h = b * b - c;

// 	if(h < 0.0 ){
// 		// no intersection
// 		return -1.0;
// 	} 

// 	h = sqrt(h);

// 	return min(-b - h, -b + h);

// 	// return vec2(-b - h, -b + h);
// }

Point* createPointCloudSphere(uint32_t numPoints){

	auto grid = cg::this_grid();
	curandStateXORWOW_t thread_random_state;
	curand_init(grid.thread_rank(), 0, 0, &thread_random_state);

	Point* points = allocator->alloc<Point*>(16 * numPoints);

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

	grid.sync();

	return points;
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


	return;

	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime_start));

	uniforms = _uniforms;

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	uint8_t* voxelBuffer_u8 = (uint8_t*)voxelBuffer;

	// allocate framebuffer memory
	int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
	uint64_t* framebuffer = allocator->alloc<uint64_t*>(framebufferSize);
	uint64_t* framebuffer_2 = allocator->alloc<uint64_t*>(framebufferSize);

	int numPixels = uniforms.width * uniforms.height;


	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
		framebuffer_2[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
	});

	grid.sync();


	// uint32_t numPoints = 1'000'000;
	// // Point* points = allocator->alloc<Point*>(16 * numPoints);
	// Point* points = createPointCloudSphere(numPoints);
	//
	// // PROJECT POINTS TO PIXELS
	// if(false)
	// processRange(numPoints, [&](int index){

	// 	Point point = points[index];
	// 	mat4 transform = uniforms.proj * uniforms.view;

	// 	float4 ndc = transform * float4{point.x, point.y, point.z, 1.0f};
	// 	ndc.x = ndc.x / ndc.w;
	// 	ndc.y = ndc.y / ndc.w;
	// 	ndc.z = ndc.z / ndc.w;

	// 	if(ndc.w <= 0.0) return;
	// 	if(ndc.x < -1.0) return;
	// 	if(ndc.x >  1.0) return;
	// 	if(ndc.y < -1.0) return;
	// 	if(ndc.y >  1.0) return;

	// 	float2 imgPos = {
	// 		(ndc.x * 0.5f + 0.5f) * uniforms.width, 
	// 		(ndc.y * 0.5f + 0.5f) * uniforms.height,
	// 	};

	// 	uint32_t color = point.color;
	// 	uint8_t* rgba = (uint8_t*)&color;

	// 	float depth = ndc.w;
	// 	uint64_t udepth = *((uint32_t*)&depth);

	// 	// uint64_t pixel = (udepth << 32ull) | index;
	// 	uint64_t pixel = (udepth << 32ull) | color;


	// 	int2 pixelCoords = make_int2(imgPos.x, imgPos.y);
	// 	int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;
	// 	pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

	// 	atomicMin(&framebuffer_2[pixelID], pixel);
	// });

	grid.sync();

	// PROJECT VOXELS TO PIXELS
	processRange(uniforms.numPoints, [&](int index){

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
	float4 rayPos4     = uniforms.viewInverse * float4{0.0, 0.0, 0.0, 1.0};
	float3 rayOrigin = make_float3(rayPos4);
	float3 cameraTarget = uniforms.viewInverse * float3{0.0, 0.0, -1.0f};
	float3 cameraDir = normalize(cameraTarget - rayOrigin);

	// // if(false)
	// processRange(uniforms.width * uniforms.height, [&](int pixelID){
		
	// 	int x = pixelID % int(uniforms.width);
	// 	int y = (pixelID / int(uniforms.width));

	// 	// float closestT = Infinity;
	// 	float closestZ = Infinity;
	// 	bool nonePicked = true;
	// 	uint32_t pickedVoxel = -1;
	// 	uint32_t pickedColor = 0;
	// 	uint8_t* pickedRGBA  = (uint8_t*)&pickedColor;

	// 	int window = 5;
	// 	for(int dx = -window; dx <= window; dx++)
	// 	for(int dy = -window; dy <= window; dy++)
	// 	{
	// 		int neighborPixelID = (x + dx) + uniforms.width * (y + dy);

	// 		if(neighborPixelID >= numPixels) continue;

	// 		uint64_t neighbor = framebuffer[neighborPixelID];
	// 		uint32_t udepth = (neighbor >> 32);
	// 		uint32_t neighborIndex = neighbor & 0xffffffff;

	// 		if(neighbor == DEFAULT_FRAGMENT) continue;
	// 		if(neighborIndex >= uniforms.numVoxels) continue;

	// 		int X = voxelBuffer_u8[6 * neighborIndex + 0];
	// 		int Y = voxelBuffer_u8[6 * neighborIndex + 1];
	// 		int Z = voxelBuffer_u8[6 * neighborIndex + 2];
	// 		int r = voxelBuffer_u8[6 * neighborIndex + 3];
	// 		int g = voxelBuffer_u8[6 * neighborIndex + 4];
	// 		int b = voxelBuffer_u8[6 * neighborIndex + 5];

	// 		float4 ndc = {
	// 			2.0f * float(x + 0.5f) / uniforms.width - 1.0f,
	// 			2.0f * float(y + 0.5f) / uniforms.height - 1.0f,
	// 			0.0f, 1.0f
	// 		};

	// 		float4 pixelViewDir = uniforms.projInverse * ndc;
	// 		pixelViewDir = pixelViewDir / pixelViewDir.w;
	// 		float4 pixelWorldDir = uniforms.viewInverse * pixelViewDir;
	// 		float3 rayTarget = make_float3(pixelWorldDir);
	// 		float3 rayDir = normalize(rayTarget - rayOrigin);

	// 		float3 spherePos = {float(X), float(Y), float(Z)};
	// 		float sphereRadius = 0.5f;

	// 		float t = intersect_sphere(rayOrigin, rayDir, spherePos, sphereRadius);
	// 		float z = dot(t * rayDir, cameraDir);

	// 		if(t > 0.0){

	// 			closestZ = min(closestZ, z);
	// 			pickedVoxel = neighborIndex;
	// 			pickedColor = r | (g << 8) | (b << 16);
	// 			nonePicked = false;

	// 			uint32_t color = r | (g << 8) | (b << 16);

	// 			uint64_t udepth = *((uint32_t*)&z);
	// 			uint64_t fragment = (udepth << 32ull) | uint64_t(color);

	// 			int targetpixelID = x + uniforms.width * y;
	// 			atomicMin(&framebuffer_2[targetpixelID], fragment);
	// 		}
	// 	}
	// });

	grid.sync();

	uint32_t& tileCounter = *allocator->alloc<uint32_t*>(4);

	// grid.sync();

	// if(grid.thread_rank() == 0){
	// 	printf("%i \n", tileCounter);
	// }

	// grid.sync();

	// uint32_t TILE_SIZE = 16;
	// uint32_t tiles_x = (uniforms.width + TILE_SIZE - 1) / TILE_SIZE;
	// uint32_t tiles_y = (uniforms.height + TILE_SIZE - 1) / TILE_SIZE;
	// __shared__ Point sh_voxels[256];
	// __shared__ int sh_numVoxels;
	// processTiles(tiles_x, tiles_y, TILE_SIZE, tileCounter, [&](int tileX, int tileY){

	// 	int lx = block.thread_rank() % TILE_SIZE;
	// 	int ly = block.thread_rank() / TILE_SIZE;

	// 	int x = tileX * TILE_SIZE + lx;
	// 	int y = tileY * TILE_SIZE + ly;
	// 	int tileID = tileX + tileY * 10;
	// 	int pixelID = x + y * uniforms.width;

	// 	if(block.thread_rank() == 0){
	// 		sh_numVoxels = 0;
	// 	}

	// 	block.sync();

	// 	bool isAlive = true;
	// 	isAlive = isAlive && x < uniforms.width;
	// 	isAlive = isAlive && y < uniforms.height;
	// 	isAlive = isAlive && pixelID < numPixels;

	// 	float4 ndc = {
	// 		2.0f * float(x + 0.5f) / uniforms.width - 1.0f,
	// 		2.0f * float(y + 0.5f) / uniforms.height - 1.0f,
	// 		0.0f, 1.0f
	// 	};
	// 	float4 pixelViewDir = uniforms.projInverse * ndc;
	// 	pixelViewDir = pixelViewDir / pixelViewDir.w;
	// 	float4 pixelWorldDir = uniforms.viewInverse * pixelViewDir;
	// 	float3 rayTarget = make_float3(pixelWorldDir);
	// 	float3 rayDir = normalize(rayTarget - rayOrigin);

	// 	if(isAlive){
	// 		uint64_t encoded = framebuffer[pixelID];
	// 		uint32_t udepth = (encoded >> 32);
	// 		uint32_t voxelIndex = encoded & 0xffffffff;

	// 		if(encoded != DEFAULT_FRAGMENT){
	// 			uint32_t idx = atomicAdd(&sh_numVoxels, 1);

	// 			Point voxel;
	// 			voxel.x = float(voxelBuffer_u8[6 * voxelIndex + 0]);
	// 			voxel.y = float(voxelBuffer_u8[6 * voxelIndex + 1]);
	// 			voxel.z = float(voxelBuffer_u8[6 * voxelIndex + 2]);
	// 			uint32_t r = voxelBuffer_u8[6 * voxelIndex + 3];
	// 			uint32_t g = voxelBuffer_u8[6 * voxelIndex + 4];
	// 			uint32_t b = voxelBuffer_u8[6 * voxelIndex + 5];
	// 			voxel.color = r | (g << 8) | (b << 16);

	// 			sh_voxels[idx] = voxel;
	// 		}
	// 	}

	// 	block.sync();

	// 	for(int i = 0; i < sh_numVoxels; i++){

	// 		Point voxel = sh_voxels[i];

	// 		float3 spherePos = {voxel.x, voxel.y, voxel.z};
	// 		float sphereRadius = 0.5f;

	// 		float t = intersect_sphere(rayOrigin, rayDir, spherePos, sphereRadius);
	// 		float z = dot(t * rayDir, cameraDir);

	// 		if(t > 0.0){

	// 			uint64_t udepth = *((uint32_t*)&z);
	// 			uint64_t fragment = (udepth << 32ull) | uint64_t(voxel.color);

	// 			int targetpixelID = x + uniforms.width * y;
	// 			atomicMin(&framebuffer_2[targetpixelID], fragment);
	// 		}

	// 	}
	// });

	grid.sync();

	uint64_t t_resolve_tiles_start;
	if(grid.thread_rank() == 0){
		t_resolve_tiles_start = nanotime();
	}

	uint32_t TILE_SIZE = 16;
	uint32_t tiles_x = (uniforms.width + TILE_SIZE - 1) / TILE_SIZE;
	uint32_t tiles_y = (uniforms.height + TILE_SIZE - 1) / TILE_SIZE;
	__shared__ Point sh_voxels[10 * 256];
	__shared__ int sh_numVoxels;
	// if(false)
	processTiles(tiles_x, tiles_y, TILE_SIZE, tileCounter, [&](int tileX, int tileY){

		int targetTileX = tileX;
		int targetTileY = tileY;

		int lx = block.thread_rank() % TILE_SIZE;
		int ly = block.thread_rank() / TILE_SIZE;

		int x = targetTileX * TILE_SIZE + lx;
		int y = targetTileY * TILE_SIZE + ly;
		int tileID = targetTileX + targetTileY * 10;
		int pixelID = x + y * uniforms.width;

		if(block.thread_rank() == 0){
			sh_numVoxels = 0;
		}

		block.sync();

		// for(int tox : {-2, -1, 0, 1, 2})
		// for(int toy : {-2, -1, 0, 1, 2})
		for(int tox : {-1, 0, 1})
		for(int toy : {-1, 0, 1})
		// for(int tox : {0})
		// for(int toy : {0})
		{
			int sourceTileX = tileX + tox;
			int sourceTileY = tileY + toy;

			if(sourceTileX < 0) continue;
			if(sourceTileY < 0) continue;
			if(sourceTileX >= tiles_x) continue;
			if(sourceTileY >= tiles_y) continue;

			int source_x = sourceTileX * TILE_SIZE + lx;
			int source_y = sourceTileY * TILE_SIZE + ly;

			if(source_x < 0) continue;
			if(source_y < 0) continue;
			if(source_x >= uniforms.width) continue;
			if(source_y >= uniforms.height) continue;

			int sourcePixelID = source_x + source_y * uniforms.width;

			uint64_t encoded = framebuffer[sourcePixelID];
			uint32_t udepth = (encoded >> 32);
			uint32_t voxelIndex = encoded & 0xffffffff;

			if(encoded != DEFAULT_FRAGMENT){
				uint32_t idx = atomicAdd(&sh_numVoxels, 1);

				Point voxel;
				voxel.x = float(voxelBuffer_u8[6 * voxelIndex + 0]);
				voxel.y = float(voxelBuffer_u8[6 * voxelIndex + 1]);
				voxel.z = float(voxelBuffer_u8[6 * voxelIndex + 2]);
				uint32_t r = voxelBuffer_u8[6 * voxelIndex + 3];
				uint32_t g = voxelBuffer_u8[6 * voxelIndex + 4];
				uint32_t b = voxelBuffer_u8[6 * voxelIndex + 5];
				voxel.color = r | (g << 8) | (b << 16);

				sh_voxels[idx] = voxel;
			}

		}

		block.sync();

		// if(block.thread_rank() == 0 && sh_numVoxels > 1000){
		// 	printf("sh_numVoxels: %i \n", sh_numVoxels);
		// }

		block.sync();

		bool isAlive = true;
		isAlive = isAlive && x < uniforms.width;
		isAlive = isAlive && y < uniforms.height;
		isAlive = isAlive && pixelID < numPixels;

		float4 ndc = {
			2.0f * float(x + 0.5f) / uniforms.width - 1.0f,
			2.0f * float(y + 0.5f) / uniforms.height - 1.0f,
			0.0f, 1.0f
		};
		float4 pixelViewDir = uniforms.projInverse * ndc;
		pixelViewDir = pixelViewDir / pixelViewDir.w;
		float4 pixelWorldDir = uniforms.viewInverse * pixelViewDir;
		float3 rayTarget = make_float3(pixelWorldDir);
		float3 rayDir = normalize(rayTarget - rayOrigin);

		if(isAlive){
			uint64_t encoded = framebuffer[pixelID];
			uint32_t udepth = (encoded >> 32);
			uint32_t voxelIndex = encoded & 0xffffffff;

			if(encoded != DEFAULT_FRAGMENT){
				uint32_t idx = atomicAdd(&sh_numVoxels, 1);

				Point voxel;
				voxel.x = float(voxelBuffer_u8[6 * voxelIndex + 0]);
				voxel.y = float(voxelBuffer_u8[6 * voxelIndex + 1]);
				voxel.z = float(voxelBuffer_u8[6 * voxelIndex + 2]);
				uint32_t r = voxelBuffer_u8[6 * voxelIndex + 3];
				uint32_t g = voxelBuffer_u8[6 * voxelIndex + 4];
				uint32_t b = voxelBuffer_u8[6 * voxelIndex + 5];
				voxel.color = r | (g << 8) | (b << 16);

				sh_voxels[idx] = voxel;
			}
		}

		block.sync();

		for(int i = 0; i < sh_numVoxels; i++){

			Point voxel = sh_voxels[i];

			float3 spherePos = {voxel.x, voxel.y, voxel.z};
			float sphereRadius = 0.5f;

			float t = intersect_cube(rayOrigin, rayDir, {voxel.x, voxel.y, voxel.z}, 1.0f);
			// float t = intersect_sphere(rayOrigin, rayDir, spherePos, sphereRadius);
			float z = dot(t * rayDir, cameraDir);

			if(t > 0.0){

				uint64_t udepth = *((uint32_t*)&z);
				uint64_t fragment = (udepth << 32ull) | uint64_t(voxel.color);

				int targetpixelID = x + uniforms.width * y;
				atomicMin(&framebuffer_2[targetpixelID], fragment);
			}

		}
	});

	grid.sync();

	if(grid.thread_rank() == 0){
		uint64_t micros = (nanotime() - t_resolve_tiles_start) / 1000;
	
		float millies = float(micros) / 1000.0f;

		printf("duration: %.1f ms \n", millies);
	}

	// Ray-Cast a sphere, method 2
	// float4 rayPos4     = uniforms.viewInverse * float4{0.0, 0.0, 0.0, 1.0};
	// float3 rayOrigin = make_float3(rayPos4);
	// float3 cameraTarget = uniforms.viewInverse * float3{0.0, 0.0, -1.0f};
	// float3 cameraDir = normalize(cameraTarget - rayOrigin);
	// float sphereRadius = 20.5f;
	// float3 spherePos = {40.0, 30.0, 115.0};

	// processRange(uniforms.width * uniforms.height, [&](int pixelID){

	// 	// compute ray from pixel coordinates
	// 	int x = pixelID % int(uniforms.width);
	// 	int y = (pixelID / int(uniforms.width));

	// 	float4 ndc = {
	// 		2.0f * float(x + 0.5f) / uniforms.width - 1.0f,
	// 		2.0f * float(y + 0.5f) / uniforms.height - 1.0f,
	// 		0.0f, 1.0f
	// 	};

	// 	float4 pixelViewDir = uniforms.projInverse * ndc;
	// 	pixelViewDir = pixelViewDir / pixelViewDir.w;
	// 	float4 pixelWorldDir = uniforms.viewInverse * pixelViewDir;
	// 	float3 rayTarget = make_float3(pixelWorldDir);
	// 	float3 rayDir = normalize(rayTarget - rayOrigin);

	// 	float t = intersect_sphere(rayOrigin, rayDir, spherePos, sphereRadius);

	// 	uint32_t color;
	// 	uint8_t* rgba = (uint8_t*)&color;

	// 	if(t > 0.0f){
	// 		color = 0x0000ff00;

	// 		// t is actual distance to intersection.
	// 		// However, because the rasterized scene uses distance on central view-dir "z",
	// 		// we need to transform t to "z" by projecting t * rayDir onto cameraDir.

	// 		// float depth = t;
	// 		float z = dot(t * rayDir, cameraDir);
	// 		float depth = z;

	// 		uint64_t udepth = *((uint32_t*)&depth);
	// 		uint64_t pixel = (udepth << 32ull) | 0x0000ffff;

	// 		atomicMin(&framebuffer_2[pixelID], pixel);
	// 	}
	// });

	grid.sync();

	// // Ray-Cast
	// float sphereRadius = 20.5f;
	// float3 spherePos = {40.0, 30.0, 115.0};
	// float3 normal = {0.0, 0.0, 1.0};
	// float pd = 10.0f;

	// processRange(uniforms.width * uniforms.height, [&](int pixelID){

	// 	// compute ray from pixel coordinates
	// 	int x = pixelID % int(uniforms.width);
	// 	int y = (pixelID / int(uniforms.width));

	// 	float4 ndc = {
	// 		2.0f * float(x + 0.5f) / uniforms.width - 1.0f,
	// 		2.0f * float(y + 0.5f) / uniforms.height - 1.0f,
	// 		0.0f, 1.0f
	// 	};

	// 	float4 pixelViewDir = uniforms.projInverse * ndc;
	// 	pixelViewDir = pixelViewDir / pixelViewDir.w;
	// 	float4 pixelWorldDir = uniforms.viewInverse * pixelViewDir;
	// 	float3 rayTarget = make_float3(pixelWorldDir);
	// 	float3 rayDir = normalize(rayTarget - rayOrigin);

	// 	// float t = intersect_plane(rayOrigin, rayDir, normal, pd);
	// 	float t = intersect_cube(rayOrigin, rayDir, {0.0, 0.0, 50.0}, 40.0);

	// 	uint32_t color;
	// 	uint8_t* rgba = (uint8_t*)&color;

	// 	if(t > 0.0f){
	// 		color = 0x0000ff00;
	// 		color = t;

	// 		// t is actual distance to intersection.
	// 		// However, because the rasterized scene uses distance on central view-dir "z",
	// 		// we need to transform t to "z" by projecting t * rayDir onto cameraDir.

	// 		// float depth = t;
	// 		float z = dot(t * rayDir, cameraDir);
	// 		float depth = z;

	// 		uint64_t udepth = *((uint32_t*)&depth);
	// 		uint64_t pixel = (udepth << 32ull) | color;

	// 		atomicMin(&framebuffer_2[pixelID], pixel);
	// 	}
	// });

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
