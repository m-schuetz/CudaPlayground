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

template<typename T>
T get(uint8_t* buffer, int64_t position) {

	T value;

	memcpy(&value, buffer + position, sizeof(T));

	return value;
}

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
	uint8_t* gaussians,
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

	int numPixels = uniforms.width * uniforms.height;

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
		framebuffer_2[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
	});

	grid.sync();
	
	// PROJECT POINTS TO PIXELS
	processRange(uniforms.numPoints, [&](int index){

		mat4 transform = uniforms.proj * uniforms.view;

		float x = get<float>(gaussians, uniforms.stride * index + 0);
		float y = get<float>(gaussians, uniforms.stride * index + 4);
		float z = get<float>(gaussians, uniforms.stride * index + 8);

		float nx = get<float>(gaussians, uniforms.stride * index + 12 + 12);
		float ny = get<float>(gaussians, uniforms.stride * index + 16 + 12);
		float nz = get<float>(gaussians, uniforms.stride * index + 20 + 12);

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

		// uint32_t color = point.color;
		uint32_t color = 0x0000ff00;
		uint8_t* rgba = (uint8_t*)&color;

		rgba[0] = 128.0 * nx;
		rgba[1] = 128.0 * ny;
		rgba[2] = 128.0 * nz;

		// if(!(1'000'000 <= index && index <= 2'000'000)){
		// 	return;
		// }

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
		uint32_t color = encoded & 0xffffffffull;
		// uint32_t color = fragment.color;
		// color = fragment.depth * 0.5;
		// color = 0x0000ff00;

		surf2Dwrite(color, gl_colorbuffer, x * 4, y);
	});


}
