
#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

#include "CudaPrint.cu"

// ray tracing adapted from tutorial: https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
// author: Alan Wolfe
// (MIT LICENSE)

uint32_t SPECIAL_IDX = 1298305;

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

float3 operator*(const mat3& a, const float3& b){
	return float3{
		a.rows[0].x * b.x + a.rows[0].y * b.y + a.rows[0].z * b.z,
		a.rows[1].x * b.x + a.rows[1].y * b.y + a.rows[1].z * b.z,
		a.rows[2].x * b.x + a.rows[2].y * b.y + a.rows[2].z * b.z
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

//===============================
//===============================
//===============================

// LICENSE!
// project: SIBR_viewers
// file: gaussian_surface.vert
// url: https://github.com/graphdeco-inria/gaussian-splatting
mat3 quatToMat3(float4 q) {
	float qx = q.y;
	float qy = q.z;
	float qz = q.w;
	float qw = q.x;

	float qxx = qx * qx;
	float qyy = qy * qy;
	float qzz = qz * qz;
	float qxz = qx * qz;
	float qxy = qx * qy;
	float qyw = qy * qw;
	float qzw = qz * qw;
	float qyz = qy * qz;
	float qxw = qx * qw;

	return mat3(
		float3{1.0 - 2.0 * (qyy + qzz), 2.0 * (qxy - qzw), 2.0 * (qxz + qyw)},
		float3{2.0 * (qxy + qzw), 1.0 - 2.0 * (qxx + qzz), 2.0 * (qyz - qxw)},
		float3{2.0 * (qxz - qyw), 2.0 * (qyz + qxw), 1.0 - 2.0 * (qxx + qyy)}
	);
}

const float3 boxVertices[8] = {
	float3{-1, -1.00, -1.00},
	float3{-1, -1.00,  1.00},
	float3{-1,  1.00, -1.00},
	float3{-1,  1.00,  1.00},

	float3{ 1, -1, -1},
	float3{ 1, -1,  1},
	float3{ 1,  1, -1},
	float3{ 1,  1,  1},
};

const int boxIndices[36] = {
    0, 1, 2, 1, 3, 2,
    4, 6, 5, 5, 6, 7,
    0, 2, 4, 4, 2, 6,
    1, 5, 3, 5, 7, 3,
    0, 4, 1, 4, 5, 1,
    2, 3, 6, 3, 7, 6
};


//===============================
//===============================
//===============================







namespace cg = cooperative_groups;

Uniforms uniforms;
Allocator* allocator;
uint64_t nanotime_start;
mat4 viewProj;

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

// int COLORMODE_TEXTURE          = 0;
// int COLORMODE_UV               = 1;
// int COLORMODE_TRIANGLE_ID      = 2;
// int COLORMODE_TIME             = 3;
// int COLORMODE_TIME_NORMALIZED  = 4;

struct Triangles{
	uint32_t numTriangles;
	float3* positions;
	float2* uvs;
	uint32_t* colors;
};

struct RasterizationSettings{
	int colorMode = COLORMODE_TRIANGLE_ID;
	mat4 world;
};

void rasterizeTriangles(Triangles* triangles, uint64_t* framebuffer, RasterizationSettings settings){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int colorMode = settings.colorMode;

	mat4 transform = viewProj * settings.world;

	uint32_t& processedTriangles = *allocator->alloc<uint32_t*>(4);
	if(grid.thread_rank() == 0){
		processedTriangles = 0;
	}
	grid.sync();

	{
		__shared__ int sh_triangleIndex;

		block.sync();

		// safety mechanism: each block draws at most <loop_max> triangles
		int loop_max = 50'000;
		for(int loop_i = 0; loop_i < loop_max; loop_i++){
			
			// grab the index of the next unprocessed triangle
			block.sync();
			if(block.thread_rank() == 0){
				sh_triangleIndex = atomicAdd(&processedTriangles, 1);
			}
			block.sync();

			if(sh_triangleIndex >= triangles->numTriangles) break;

			// project x/y to pixel coords
			// z: whatever 
			// w: linear depth
			auto toScreenCoord = [&](float3 p){
				float4 pos = transform * float4{p.x, p.y, p.z, 1.0f};

				pos.x = pos.x / pos.w;
				pos.y = pos.y / pos.w;

				float4 imgPos = {
					(pos.x * 0.5f + 0.5f) * uniforms.width, 
					(pos.y * 0.5f + 0.5f) * uniforms.height,
					pos.z, 
					pos.w
				};

				return imgPos;
			};

			int i0 = 3 * sh_triangleIndex + 0;
			int i1 = 3 * sh_triangleIndex + 1;
			int i2 = 3 * sh_triangleIndex + 2;
			
			float3 v0 = triangles->positions[i0];
			float3 v1 = triangles->positions[i1];
			float3 v2 = triangles->positions[i2];

			float4 p0 = toScreenCoord(v0);
			float4 p1 = toScreenCoord(v1);
			float4 p2 = toScreenCoord(v2);

			// cull a triangle if one of its vertices is closer than depth 0
			if(p0.w < 0.0 || p1.w < 0.0 || p2.w < 0.0) continue;

			float2 v01 = {p1.x - p0.x, p1.y - p0.y};
			float2 v02 = {p2.x - p0.x, p2.y - p0.y};

			auto cross = [](float2 a, float2 b){ return a.x * b.y - a.y * b.x; };

			// {// backface culling
			// 	float w = cross(v01, v02);
			// 	if(w < 0.0) continue;
			// }

			// compute screen-space bounding rectangle
			float min_x = min(min(p0.x, p1.x), p2.x);
			float min_y = min(min(p0.y, p1.y), p2.y);
			float max_x = max(max(p0.x, p1.x), p2.x);
			float max_y = max(max(p0.y, p1.y), p2.y);

			// clamp to screen
			min_x = clamp(min_x, 0.0f, uniforms.width);
			min_y = clamp(min_y, 0.0f, uniforms.height);
			max_x = clamp(max_x, 0.0f, uniforms.width);
			max_y = clamp(max_y, 0.0f, uniforms.height);

			int size_x = ceil(max_x) - floor(min_x);
			int size_y = ceil(max_y) - floor(min_y);
			int numFragments = size_x * size_y;

			// iterate through fragments in bounding rectangle and draw if within triangle
			int numProcessedSamples = 0;
			for(int fragOffset = 0; fragOffset < numFragments; fragOffset += block.num_threads()){

				// safety mechanism: don't draw more than <x> pixels per thread
				if(numProcessedSamples > 10'000) break;

				int fragID = fragOffset + block.thread_rank();
				int fragX = fragID % size_x;
				int fragY = fragID / size_x;

				float2 pFrag = {
					floor(min_x) + float(fragX), 
					floor(min_y) + float(fragY)
				};
				float2 sample = {pFrag.x - p0.x, pFrag.y - p0.y};

				// v: vertex[0], s: vertex[1], t: vertex[2]
				float s = cross(sample, v02) / cross(v01, v02);
				float t = cross(v01, sample) / cross(v01, v02);
				float v = 1.0 - (s + t);

				int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
				int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;
				pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

				if(s >= 0.0)
				if(t >= 0.0)
				if(s + t <= 1.0)
				{
					uint8_t* v0_rgba = (uint8_t*)&triangles->colors[i0];
					uint8_t* v1_rgba = (uint8_t*)&triangles->colors[i1];
					uint8_t* v2_rgba = (uint8_t*)&triangles->colors[i2];

					float2 v0_uv = triangles->uvs[i0];
					float2 v1_uv = triangles->uvs[i1];
					float2 v2_uv = triangles->uvs[i2];
					float2 uv = {
						v * v0_uv.x + s * v1_uv.x + t * v2_uv.x,
						v * v0_uv.y + s * v1_uv.y + t * v2_uv.y
					};

					uint32_t color;
					uint8_t* rgb = (uint8_t*)&color;

					// { // color by vertex color
					// 	rgb[0] = v * v0_rgba[0] + s * v1_rgba[0] + t * v2_rgba[0];
					// 	rgb[1] = v * v0_rgba[1] + s * v1_rgba[1] + t * v2_rgba[1];
					// 	rgb[2] = v * v0_rgba[2] + s * v1_rgba[2] + t * v2_rgba[2];
					// }

					if(colorMode == COLORMODE_UV && triangles->uvs != nullptr){
						// UV
						rgb[0] = 255.0f * uv.x;
						rgb[1] = 255.0f * uv.y;
						rgb[2] = 0;
					}else if(colorMode == COLORMODE_TRIANGLE_ID){
						// TRIANGLE INDEX
						color = sh_triangleIndex * 123456;
					}else{
						// WHATEVER
						color = sh_triangleIndex * 123456;
					}

					// color = 0x000000ff;

					float depth = v * p0.w + s * p1.w + t * p2.w;
					uint64_t udepth = *((uint32_t*)&depth);
					uint64_t pixel = (udepth << 32ull) | color;

					atomicMin(&framebuffer[pixelID], pixel);
				}

				numProcessedSamples++;
			}


		}
	}

	grid.sync();

	// if(grid.thread_rank() == 0){
	// 	printf("%i \n", processedTriangles);
	// }
}

// template <typename... Args>
// inline void printfmt(const char* str, const Args&... args) {

// 	constexpr uint32_t numargs{ sizeof...(Args) };

// 	int stringSize = strlen(str);
// 	int argsSize = 0;
// 	int numArgs = 0;

// 	for(const auto p : {args...}) {
// 		argsSize += sizeof(p);
// 		numArgs++;
// 	}

// 	printf("stringSize: %i, argsSize: %i, numArgs: %i \n", stringSize, argsSize, numArgs);


// }

extern "C" __global__
void kernel(
	const Uniforms _uniforms,
	unsigned int* buffer,
	uint8_t* gaussians,
	cudaSurfaceObject_t gl_colorbuffer,
	CudaPrint* cudaprint
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime_start));

	uniforms = _uniforms;

	viewProj = uniforms.proj * uniforms.view;

	viewProj.rows[0] = { 1.484, -0.024, -0.004, -0.004};
	viewProj.rows[1] = { 0.013,  2.193, -0.291, -0.291};
	viewProj.rows[2] = { 0.011,  0.666,  0.957,  0.957};
	viewProj.rows[3] = {-0.487, -4.415,  3.940,  3.958};

	viewProj = viewProj.transpose();
	viewProj.rows[1].x *= -1.0f;
	viewProj.rows[1].y *= -1.0f;
	viewProj.rows[1].z *= -1.0f;
	viewProj.rows[1].w *= -1.0f;

	if(grid.thread_rank() == 0){
		cudaprint->set("test u32", 123);
		cudaprint->set("test key", "abc value");

		if(int(uniforms.time * 100.0) % 100 < 1){
			cudaprint->print("-abc {:.2f} def- \n", 134.1f);
		}
	}


	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	// allocate framebuffer memory
	int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
	uint64_t* framebuffer = allocator->alloc<uint64_t*>(framebufferSize);
	uint64_t* framebuffer_2 = allocator->alloc<uint64_t*>(framebufferSize);

	int triangleCapacity    = 10'000'000;
	int vertexCapacity      = 3 * triangleCapacity;
	Triangles* triangles    = allocator->alloc<Triangles*>(sizeof(Triangles));
	triangles->positions    = allocator->alloc<float3*  >(sizeof(float3) * vertexCapacity);
	triangles->uvs          = allocator->alloc<float2*  >(sizeof(float2) * vertexCapacity);
	triangles->colors       = allocator->alloc<uint32_t*>(sizeof(uint32_t) * vertexCapacity);
	triangles->numTriangles = 0;
	

	int numPixels = uniforms.width * uniforms.height;

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
		framebuffer_2[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
	});

	// // clear triangle buffer
	// processRange(0, triangleCapacity, [&](int index){
	// 	triangles->positions[3 * index + 0] = {0.0, 0.0, 0.0};
	// 	triangles->positions[3 * index + 1] = {0.0, 0.0, 0.0};
	// 	triangles->positions[3 * index + 2] = {0.0, 0.0, 0.0};
	// 	triangles->uvs[3 * index + 0] = {0.0, 0.0};
	// 	triangles->uvs[3 * index + 1] = {0.0, 0.0};
	// 	triangles->uvs[3 * index + 2] = {0.0, 0.0};
	// });

	grid.sync();
	
	// PROJECT POINTS TO PIXELS
	processRange(uniforms.numPoints, [&](int index){

		// mat4 transform = uniforms.proj * uniforms.view;
		mat4 transform = viewProj;


		
		// if(index == 0){
		// 	printf("===========================\n");
		// 	printf("transform: \n");
		// 	printf("%.3f, %.3f, %.3f, %.3f \n", transform.rows[0].x, transform.rows[0].y, transform.rows[0].z, transform.rows[0].w);
		// 	printf("%.3f, %.3f, %.3f, %.3f \n", transform.rows[1].x, transform.rows[1].y, transform.rows[1].z, transform.rows[1].w);
		// 	printf("%.3f, %.3f, %.3f, %.3f \n", transform.rows[2].x, transform.rows[2].y, transform.rows[2].z, transform.rows[2].w);
		// 	printf("%.3f, %.3f, %.3f, %.3f \n", transform.rows[3].x, transform.rows[3].y, transform.rows[3].z, transform.rows[3].w);
		// }

		// transform.rows[0] = {1.000, -0.011, -0.004, -0.000};
		// transform.rows[1] = {0.009, 0.957, -0.291, 0.000};
		// transform.rows[2] = {0.007, 0.291, 0.957, 0.000};
		// transform.rows[3] = {-0.328, -1.926, 3.958, 1.000};

		float3 center = {
			get<float>(gaussians, uniforms.stride * index + 0),
			get<float>(gaussians, uniforms.stride * index + 4),
			get<float>(gaussians, uniforms.stride * index + 8),
		};

		if(center.x < -1.2 | center.x > 1.7) return;
		if(center.y < 0.1 | center.y > 3.1) return;
		if(center.z < 0.0 | center.z > 2.0) return;

		float3 n = {
			get<float>(gaussians, uniforms.stride * index + 12),
			get<float>(gaussians, uniforms.stride * index + 16),
			get<float>(gaussians, uniforms.stride * index + 20),
		};

		float3 dc = {
			get<float>(gaussians, uniforms.stride * index + 24),
			get<float>(gaussians, uniforms.stride * index + 28),
			get<float>(gaussians, uniforms.stride * index + 32),
		};
		
		float opacity = get<float>(gaussians, uniforms.stride * index + 54 * 4);

		float3 scale = {
			get<float>(gaussians, uniforms.stride * index + 55 * 4 + 0),
			get<float>(gaussians, uniforms.stride * index + 55 * 4 + 4),
			get<float>(gaussians, uniforms.stride * index + 55 * 4 + 8),
		};

		float4 quaternion = {
			get<float>(gaussians, uniforms.stride * index + 58 * 4 +  0),
			get<float>(gaussians, uniforms.stride * index + 58 * 4 +  4),
			get<float>(gaussians, uniforms.stride * index + 58 * 4 +  8),
			get<float>(gaussians, uniforms.stride * index + 58 * 4 + 12),
		};

		{
			// printf("quaternion: %f, %f, %f, %f \n", quaternion.x, quaternion.y, quaternion.z, quaternion.w);
			float l = length(quaternion);
			quaternion = quaternion / l;
			// printf("quaternion: %f, %f, %f, %f \n", quaternion.x, quaternion.y, quaternion.z, quaternion.w);

			scale.x = exp(scale.x);
			scale.y = exp(scale.y);
			scale.z = exp(scale.z);
			scale = scale * 2.0;

			// printf("%f, %f, %f \n", scale.x, scale.y, scale.z);

			opacity = 1.0f / (1.0f + exp(-opacity));
			// printf("%f \n", opacity);
		}

		mat3 rotation = quatToMat3(quaternion);

		// if(index == SPECIAL_IDX){

		// 	float4 p_hom = transform.transpose() * float4{center.x, center.y, center.z, 1.0};
		// 	float p_w = 1.0f / (p_hom.w + 0.0000001f);
		// 	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

		// 	printf("===========================\n");
		// 	printf("point: %f, %f, %f \n", center.x, center.y, center.z);
		// 	printf("p_hom: %f, %f, %f, %f \n", p_hom.x, p_hom.y, p_hom.z, p_hom.w);
		// 	printf("p_proj: %f, %f, %f \n", p_proj.x, p_proj.y, p_proj.z);
		// 	// printf("scale: %f, %f, %f \n", scale.x, scale.y, scale.z);

		// 	printf("transform: \n");
		// 	printf("%.3f, %.3f, %.3f, %.3f \n", transform.rows[0].x, transform.rows[0].y, transform.rows[0].z, transform.rows[0].w);
		// 	printf("%.3f, %.3f, %.3f, %.3f \n", transform.rows[1].x, transform.rows[1].y, transform.rows[1].z, transform.rows[1].w);
		// 	printf("%.3f, %.3f, %.3f, %.3f \n", transform.rows[2].x, transform.rows[2].y, transform.rows[2].z, transform.rows[2].w);
		// 	printf("%.3f, %.3f, %.3f, %.3f \n", transform.rows[3].x, transform.rows[3].y, transform.rows[3].z, transform.rows[3].w);
		// }

		// if(false)
		// if(index < 100'000)
		{
			{
				uint32_t quadID = atomicAdd(&triangles->numTriangles, 2) / 2;

				triangles->positions[6 * quadID + 0] = center + rotation * (scale * float3{-1.0, -1.0, -1.0});
				triangles->positions[6 * quadID + 1] = center + rotation * (scale * float3{ 1.0, -1.0, -1.0});
				triangles->positions[6 * quadID + 2] = center + rotation * (scale * float3{ 1.0,  1.0, -1.0});
				triangles->positions[6 * quadID + 3] = center + rotation * (scale * float3{-1.0, -1.0, -1.0});
				triangles->positions[6 * quadID + 4] = center + rotation * (scale * float3{ 1.0,  1.0, -1.0});
				triangles->positions[6 * quadID + 5] = center + rotation * (scale * float3{-1.0,  1.0, -1.0});

				triangles->uvs[6 * quadID + 0] = float2{0.0, 0.0};
				triangles->uvs[6 * quadID + 1] = float2{1.0, 0.0};
				triangles->uvs[6 * quadID + 2] = float2{1.0, 1.0};
				triangles->uvs[6 * quadID + 3] = float2{0.0, 0.0};
				triangles->uvs[6 * quadID + 4] = float2{1.0, 1.0};
				triangles->uvs[6 * quadID + 5] = float2{0.0, 1.0};
			}

			{
				uint32_t quadID = atomicAdd(&triangles->numTriangles, 2) / 2;

				triangles->positions[6 * quadID + 0] = center + rotation * (scale * float3{-1.0, -1.0, -1.0});
				triangles->positions[6 * quadID + 1] = center + rotation * (scale * float3{ 1.0, -1.0, -1.0});
				triangles->positions[6 * quadID + 2] = center + rotation * (scale * float3{ 1.0, -1.0,  1.0});
				triangles->positions[6 * quadID + 3] = center + rotation * (scale * float3{-1.0, -1.0, -1.0});
				triangles->positions[6 * quadID + 4] = center + rotation * (scale * float3{ 1.0, -1.0,  1.0});
				triangles->positions[6 * quadID + 5] = center + rotation * (scale * float3{-1.0, -1.0,  1.0});

				triangles->uvs[6 * quadID + 0] = float2{0.0, 0.0};
				triangles->uvs[6 * quadID + 1] = float2{1.0, 0.0};
				triangles->uvs[6 * quadID + 2] = float2{1.0, 1.0};
				triangles->uvs[6 * quadID + 3] = float2{0.0, 0.0};
				triangles->uvs[6 * quadID + 4] = float2{1.0, 1.0};
				triangles->uvs[6 * quadID + 5] = float2{0.0, 1.0};
			}

			{
				uint32_t quadID = atomicAdd(&triangles->numTriangles, 2) / 2;

				triangles->positions[6 * quadID + 0] = center + rotation * (scale * float3{-1.0, -1.0, -1.0});
				triangles->positions[6 * quadID + 1] = center + rotation * (scale * float3{-1.0,  1.0, -1.0});
				triangles->positions[6 * quadID + 2] = center + rotation * (scale * float3{-1.0,  1.0,  1.0});
				triangles->positions[6 * quadID + 3] = center + rotation * (scale * float3{-1.0, -1.0, -1.0});
				triangles->positions[6 * quadID + 4] = center + rotation * (scale * float3{-1.0,  1.0,  1.0});
				triangles->positions[6 * quadID + 5] = center + rotation * (scale * float3{-1.0, -1.0,  1.0});

				triangles->uvs[6 * quadID + 0] = float2{0.0, 0.0};
				triangles->uvs[6 * quadID + 1] = float2{1.0, 0.0};
				triangles->uvs[6 * quadID + 2] = float2{1.0, 1.0};
				triangles->uvs[6 * quadID + 3] = float2{0.0, 0.0};
				triangles->uvs[6 * quadID + 4] = float2{1.0, 1.0};
				triangles->uvs[6 * quadID + 5] = float2{0.0, 1.0};
			}
		}


		
		// worldPos = ellipsoidRotation * (scale * center);
		// worldPos += ellipsoidCenter;

		// float4 ndc = transform * float4{center.x, center.y, center.z, 1.0f};
		// ndc.x = ndc.x / ndc.w;
		// ndc.y = ndc.y / ndc.w;
		// ndc.z = ndc.z / ndc.w;

		// if(ndc.w <= 0.0) return;
		// if(ndc.x < -1.0) return;
		// if(ndc.x >  1.0) return;
		// if(ndc.y < -1.0) return;
		// if(ndc.y >  1.0) return;

		{
			// float sd = max(abs(scale.x), max(abs(scale.y), abs(scale.z)));
			// float splatSize = max(min(0.5f * sd, 20.0f), 1.0);

			for(int boxVertexIndex = 0; boxVertexIndex < 8; boxVertexIndex++){

				float3 boxVertex = boxVertices[boxVertexIndex];

			// float steps = 10.0;
			// for(float us = 0.0; us <= steps; us += 1.0)
			// for(float vs = 0.0; vs <= steps; vs += 1.0)
			// {

			// 	float3 boxVertex = float3{
			// 		2.0 * us / steps - 1.0f,
			// 		0.0f,
			// 		2.0 * vs / steps - 1.0f,
			// 	};

				float3 worldPos = rotation * boxVertex * 10.0;
				// float3 worldPos = rotation * scale * boxVertex;
				// float3 worldPos = rotation * scale * boxVertex;
				// worldPos = 0.002 * worldPos + center;
				worldPos = 0.0005 * (0.5 * cos(5.0 * uniforms.time) + 1.0f) * worldPos + center;
				// worldPos = worldPos + center;
				// float3 worldPos = center;

				float4 ndc = transform * float4{worldPos.x, worldPos.y, worldPos.z, 1.0f};
				ndc.x = ndc.x / ndc.w;
				ndc.y = ndc.y / ndc.w;
				ndc.z = ndc.z / ndc.w;

				if(ndc.w <= 0.0) continue;
				if(ndc.x < -1.0) continue;
				if(ndc.x >  1.0) continue;
				if(ndc.y < -1.0) continue;
				if(ndc.y >  1.0) continue;

				float2 imgPos = {
					(ndc.x * 0.5f + 0.5f) * uniforms.width, 
					(ndc.y * 0.5f + 0.5f) * uniforms.height,
				};

				// uint32_t color = point.color;
				uint32_t color = 0x0000ff00;
				uint8_t* rgba = (uint8_t*)&color;

				rgba[0] = 128.0 * dc.x;
				rgba[1] = 128.0 * dc.y;
				rgba[2] = 128.0 * dc.z;

				float depth = ndc.w;
				uint64_t udepth = *((uint32_t*)&depth);

				// uint64_t pixel = (udepth << 32ull) | index;
				uint64_t pixel = (udepth << 32ull) | color;

				int2 pixelCoords = make_int2(imgPos.x, imgPos.y);
				int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;
				pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

				// atomicMin(&framebuffer_2[pixelID], pixel);
			}
		}





		// float sd = max(abs(scale.x), max(abs(scale.y), abs(scale.z)));
		// float splatSize = max(min(0.5f * sd, 20.0f), 1.0);

		// for(int ox = 0; ox <= splatSize; ox++)
		// for(int oy = 0; oy <= splatSize; oy++)
		// {
		// 	float2 imgPos = {
		// 		ox + (ndc.x * 0.5f + 0.5f) * uniforms.width, 
		// 		oy + (ndc.y * 0.5f + 0.5f) * uniforms.height,
		// 	};

		// 	// uint32_t color = point.color;
		// 	uint32_t color = 0x0000ff00;
		// 	uint8_t* rgba = (uint8_t*)&color;

		// 	rgba[0] = 128.0 * dc.x;
		// 	rgba[1] = 128.0 * dc.y;
		// 	rgba[2] = 128.0 * dc.z;

		// 	float depth = ndc.w;
		// 	uint64_t udepth = *((uint32_t*)&depth);

		// 	// uint64_t pixel = (udepth << 32ull) | index;
		// 	uint64_t pixel = (udepth << 32ull) | color;

		// 	int2 pixelCoords = make_int2(imgPos.x, imgPos.y);
		// 	int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;
		// 	pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

		// 	atomicMin(&framebuffer_2[pixelID], pixel);
		// }


		// float2 imgPos = {
		// 	(ndc.x * 0.5f + 0.5f) * uniforms.width, 
		// 	(ndc.y * 0.5f + 0.5f) * uniforms.height,
		// };

		// // uint32_t color = point.color;
		// uint32_t color = 0x0000ff00;
		// uint8_t* rgba = (uint8_t*)&color;

		// rgba[0] = 128.0 * dc.x;
		// rgba[1] = 128.0 * dc.y;
		// rgba[2] = 128.0 * dc.z;

		// float depth = ndc.w;
		// uint64_t udepth = *((uint32_t*)&depth);

		// // uint64_t pixel = (udepth << 32ull) | index;
		// uint64_t pixel = (udepth << 32ull) | color;

		// int2 pixelCoords = make_int2(imgPos.x, imgPos.y);
		// int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;
		// pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

		// atomicMin(&framebuffer_2[pixelID], pixel);

		
	});

	grid.sync();

	{
		// int numTriangles        = 1;
		// int numVertices         = 3 * numTriangles;
		// Triangles* triangles    = allocator->alloc<Triangles*>(sizeof(Triangles));
		// triangles->positions    = allocator->alloc<float3*  >(sizeof(float3) * numVertices);
		// triangles->uvs          = allocator->alloc<float2*  >(sizeof(float2) * numVertices);
		// triangles->colors       = allocator->alloc<uint32_t*>(sizeof(uint32_t) * numVertices);
		// triangles->numTriangles = numTriangles;

		// if(grid.thread_rank() == 0){
		// 	triangles->positions[0] = {0.0, 0.0, 0.0};
		// 	triangles->positions[1] = {1.0, 0.0, 0.0};
		// 	triangles->positions[2] = {1.0, 1.0, 0.0};
		// 	atomicAdd(&triangles->numTriangles, 1);
		// }

		RasterizationSettings settings;
		settings.colorMode = COLORMODE_UV;
		settings.world = mat4::identity();
		// settings.world = mat4::translate(
		// 	2.0 * cos(3.0 * uniforms.time),
		// 	0.0, 
		// 	0.0
		// );

		// if(grid.thread_rank() == 0){
		// 	printf("%i \n", triangles->numTriangles);
		// }

		if(grid.thread_rank() == 0){
			triangles->numTriangles = min(triangles->numTriangles, triangleCapacity);
		}
		// triangles->numTriangles = 1;

		rasterizeTriangles(triangles, framebuffer_2, settings);
	}

	grid.sync();

	// grid.sync();

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
