#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

#include "math.cuh"

constexpr int MAX_PATCHES = 1'000'000;

namespace cg = cooperative_groups;

Uniforms uniforms;
Allocator* allocator;
uint64_t nanotime_start;

constexpr float PI = 3.1415;
constexpr uint32_t BACKGROUND_COLOR = 0x00332211ull;

struct Patch{
	float s_min;
	float s_max;
	float t_min;
	float t_max;
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

	if(x > 1 && x < uniforms.width  - 2.0)
	if(y > 1 && y < uniforms.height - 2.0){

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

	if(x > 1 && x < uniforms.width  - 2.0)
	if(y > 1 && y < uniforms.height - 2.0){

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
	float4 ndc = uniforms.transform * float4{p.x, p.y, p.z, 1.0};

	ndc.x = ndc.x / ndc.w;
	ndc.y = ndc.y / ndc.w;
	ndc.z = ndc.z / ndc.w;

	ndc.x = (ndc.x * 0.5 + 0.5) * uniforms.width;
	ndc.y = (ndc.y * 0.5 + 0.5) * uniforms.height;
	// ndc.z = (ndc.z * 0.5 + 0.5) * uniforms.width;

	return ndc;
};

// s, t in range 0 to 1!
float3 sampleSphere(float s, float t){

	float u = 2.0 * 3.14 * s;
	float v = 3.14 * t;
	
	float3 xyz = {
		cos(u) * sin(v),
		sin(u) * sin(v),
		cos(v)
	};

	return xyz;
};

// s, t in range 0 to 1!
float3 samplePlane(float s, float t){
	return float3{2.0 * s - 1.0, 0.0, 2.0 * t - 1.0};
};

float3 sampleSinCos(float s, float t){

	float scale = 10.0;
	float height = 0.105;


	float time = uniforms.time;
	float su = s - 0.5;
	float tu = t - 0.5;
	float d = (su * su + tu * tu);

	float z = height * sin(scale * s + time) * cos(scale * t + time) + cos(2.0 * time) * 10.0 * height * exp(-1000.0 * d);

	return float3{
		2.0 * (-s + 0.5), 
		z, 
		2.0 * (-t + 0.5)
	};
};

// sampleSinCos, samplePlane, sampleSphere;
// auto sample = sampleSinCos;

auto getSampler(int model){
	if(model == MODEL_FUNKY_PLANE){
		return sampleSinCos;
	}else if(model == MODEL_SPHERE){
		return sampleSphere;
	}else{
		return samplePlane;
	}
};

void generatePatches2(Patch* patches, uint32_t* numPatches, int threshold, Uniforms& uniforms, Stats* stats){

	auto grid = cg::this_grid();

	if(grid.thread_rank() < 30){
		stats->numPatches[grid.thread_rank()] = 0;
	}

	auto sample = getSampler(uniforms.model);
	
	Patch* patches_finished = allocator->alloc<Patch*>(MAX_PATCHES * sizeof(Patch));
	Patch* patches_tmp_0 = allocator->alloc<Patch*>(MAX_PATCHES * sizeof(Patch));
	Patch* patches_tmp_1 = allocator->alloc<Patch*>(MAX_PATCHES * sizeof(Patch));
	uint32_t* numPatches_finished = allocator->alloc<uint32_t*>(4);
	uint32_t* numPatches_tmp_0 = allocator->alloc<uint32_t*>(4);
	uint32_t* numPatches_tmp_1 = allocator->alloc<uint32_t*>(4);

	struct PatchData{
		Patch* patch;
		uint32_t* counter;
	};

	PatchData* pingpong = allocator->alloc<PatchData*>(2 * sizeof(PatchData));
	pingpong[0].patch = patches_tmp_0;
	pingpong[0].counter = numPatches_tmp_0;
	pingpong[1].patch = patches_tmp_1;
	pingpong[1].counter = numPatches_tmp_1;

	if(grid.thread_rank() == 0){
		*numPatches_finished = 0;
		*numPatches_tmp_0 = 0;
		*numPatches_tmp_1 = 0;
	}

	grid.sync();

	// if(grid.thread_rank() == 0) 
	// 	printf("target.counter: %llu \n", pingpong[2].counter);

	Patch root;
	root.s_min = 0;
	root.s_max = 1;
	root.t_min = 0;
	root.t_max = 1;

	patches_tmp_0[0] = root;
	*numPatches_tmp_0 = 1;

	int level = 0;

	// SUBDIVIDE LARGE PATCHES
	// - if too large, divide and store in target
	// - if not too large, store in <patches>
	auto subdivide = [&](Patch* source, uint32_t* sourceCounter, Patch* target, uint32_t* targetCounter){

		processRange(*sourceCounter, [&](int index){
			Patch patch = source[index];

			float s_c = (patch.s_min + patch.s_max) * 0.5;
			float t_c = (patch.t_min + patch.t_max) * 0.5;

			float3 p_00 = sample(patch.s_min, patch.t_min);
			float3 p_01 = sample(patch.s_min, patch.t_max);
			float3 p_10 = sample(patch.s_max, patch.t_min);
			float3 p_11 = sample(patch.s_max, patch.t_max);
			float3 p_c = sample(s_c, t_c);


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
			bool isIntersectingFrustum = intersectsFrustum(uniforms.transform, nodeMin, nodeMax);

			if(!isIntersectingFrustum){
				return;
			}

			float4 ps_00 = toScreen(p_00, uniforms);
			float4 ps_01 = toScreen(p_01, uniforms);
			float4 ps_10 = toScreen(p_10, uniforms);
			float4 ps_11 = toScreen(p_11, uniforms);
			float4 ps_c = toScreen(p_c, uniforms);

			float min_x = min(min(min(ps_00.x, ps_01.x), min(ps_10.x, ps_11.x)), ps_c.x);
			float min_y = min(min(min(ps_00.y, ps_01.y), min(ps_10.y, ps_11.y)), ps_c.y);
			float max_x = max(max(max(ps_00.x, ps_01.x), max(ps_10.x, ps_11.x)), ps_c.x);
			float max_y = max(max(max(ps_00.y, ps_01.y), max(ps_10.y, ps_11.y)), ps_c.y);

			float s_x = max_x - min_x;
			float s_y = max_y - min_y;
			float area = s_x * s_y;

			if(area > threshold * threshold)
			// if(area > 64 * 64)
			{
				// too large, subdivide
				uint32_t targetIndex = atomicAdd(targetCounter, 4);

				float s_center = (patch.s_min + patch.s_max) / 2.0;
				float t_center = (patch.t_min + patch.t_max) / 2.0;

				Patch patch_00;
				patch_00.s_min = patch.s_min;
				patch_00.s_max = s_center;
				patch_00.t_min = patch.t_min;
				patch_00.t_max = t_center;
				target[targetIndex + 0] = patch_00;

				Patch patch_01;
				patch_01.s_min = patch.s_min;
				patch_01.s_max = s_center;
				patch_01.t_min = t_center;
				patch_01.t_max = patch.t_max;
				target[targetIndex + 1] = patch_01;

				Patch patch_10;
				patch_10.s_min = s_center;
				patch_10.s_max = patch.s_max;
				patch_10.t_min = patch.t_min;
				patch_10.t_max = t_center;
				target[targetIndex + 2] = patch_10;

				Patch patch_11;
				patch_11.s_min = s_center;
				patch_11.s_max = patch.s_max;
				patch_11.t_min = t_center;
				patch_11.t_max = patch.t_max;
				target[targetIndex + 3] = patch_11;


			}else{

				// float3 t_01 = p_01 - p_00;
				// float3 t_10 = p_10 - p_00;
				// float3 N = normalize(cross(t_01, t_10));
				// float3 N_v = make_float3(uniforms.view * float4{N.x, N.y, N.z, 0.0});
				
				// float a = dot(N_v, float3{0.0, 0.0, 1.0});
				// if(a < 0.0) return;

				// small enough, add to final list
				uint32_t targetIndex = atomicAdd(numPatches, 1);
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

		level++;
	}

}

void rasterizePatches_32x32(Patch* patches, uint32_t* numPatches, uint64_t* framebuffer, Uniforms& uniforms){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint32_t& processedPatches = *allocator->alloc<uint32_t*>(4);
	if(grid.thread_rank() == 0){
		processedPatches = 0;
	}

	auto sample = getSampler(uniforms.model);

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

		float s_min = patch.s_min;
		float s_max = patch.s_max;
		float t_min = patch.t_min;
		float t_max = patch.t_max;

		int index_t = block.thread_rank();
		int index_tx = index_t % 32;
		int index_ty = index_t / 32;


		float uts = float(index_tx) / 32.0f;
		float vts = float(index_ty) / 32.0f;

		float s = (1.0 - uts) * s_min + uts * s_max;
		float t = (1.0 - vts) * t_min + vts * t_max;

		float3 p = sample(s, t);

		block.sync();

		sh_samples[3 * index_t + 0] = p.x;
		sh_samples[3 * index_t + 1] = p.y;
		sh_samples[3 * index_t + 2] = p.z;

		block.sync();

		int inx = index_t + (index_tx < 31 ?  1 :  -1);
		int iny = index_t + (index_ty < 31 ? 32 : -32);

		float3 pnx = {sh_samples[3 * inx + 0], sh_samples[3 * inx + 1], sh_samples[3 * inx + 2]};
		float3 pny = {sh_samples[3 * iny + 0], sh_samples[3 * iny + 1], sh_samples[3 * iny + 2]};

		float3 tx = normalize(pnx - p);
		float3 ty = normalize(pny - p);
		float3 N = normalize(cross(ty, tx));

		float4 ps = toScreen(p, uniforms);

		uint32_t color = 0;
		// uint32_t color = patch.dbg * 12345678;
		uint8_t* rgba = (uint8_t*)&color;
		rgba[0] = 200.0 * N.x;
		rgba[1] = 200.0 * N.y;
		rgba[2] = 200.0 * N.z;
		rgba[3] = 255;

		// color = (patch.x + 1) * (patch.y + 13) * 1234567;

		// drawSprite(ps, framebuffer, color, uniforms);
		drawPoint(ps, framebuffer, color, uniforms);

		block.sync();
	}
}

void rasterizePatches_runnin_thru(Patch* patches, uint32_t* numPatches, uint64_t* framebuffer, Uniforms& uniforms){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint32_t& processedPatches = *allocator->alloc<uint32_t*>(4);
	if(grid.thread_rank() == 0){
		processedPatches = 0;
	}

	auto sample = getSampler(uniforms.model);

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

		float s_min = patch.s_min;
		float s_max = patch.s_max;
		float t_min = patch.t_min;
		float t_max = patch.t_max;

		int index_t = block.thread_rank();
		float ut = float(index_t) / float(block.num_threads());

		float s = (1.0 - ut) * s_min + ut * s_max;
		float t = t_min;

		
		float steps = 64.0;
		for(float i = 0.0f; i < steps; i = i + 1.0f){
			float vt = i / steps;
			float t = (1.0 - vt) * t_min + vt * t_max;

			float3 p = sample(s, t);
			uint32_t color = 0x000000ff;
			float4 ps = toScreen(p, uniforms);


			color = 1234567.0 * (123.0 + patch.s_min * patch.t_min);

			drawPoint(ps, framebuffer, color, uniforms);

		}


		

		block.sync();
	}
}

extern "C" __global__
void kernel_generate_patches(
	const Uniforms _uniforms,
	unsigned int* buffer,
	Patch* patches, uint32_t* numPatches,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime_start));

	uniforms = _uniforms;

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	grid.sync();
	if(grid.thread_rank() == 0){
		*numPatches = 0;
	}
	grid.sync();

	// generatePatches(patches, numPatches, uniforms);

	int threshold = 32;
	if(uniforms.method == METHOD_32X32){
		threshold = 32;
	}else if(uniforms.method == METHOD_RUNNIN_THRU){
		threshold = 64;
	}

	generatePatches2(patches, numPatches, threshold, uniforms, stats);
}

extern "C" __global__
void kernel_rasterize_patches_32x32(
	const Uniforms _uniforms,
	unsigned int* buffer,
	Patch* patches, uint32_t* numPatches,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	Allocator _allocator(buffer, 0);

	uniforms = _uniforms;
	allocator = &_allocator;

	// allocate framebuffer memory
	int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
	uint64_t* framebuffer = allocator->alloc<uint64_t*>(framebufferSize);

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
	});

	grid.sync();

	uint64_t t_00 = nanotime();

	rasterizePatches_32x32(patches, numPatches, framebuffer, uniforms);
	grid.sync();

	uint64_t t_20 = nanotime();

	if(grid.thread_rank() == 0 && (stats->frameID % 100) == 0){
		stats->time_1 = float((t_20 - t_00) / 1000llu) / 1000.0f;
	}

	grid.sync();

	// transfer framebuffer to opengl texture
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){

		int x = pixelIndex % int(uniforms.width);
		int y = pixelIndex / int(uniforms.width);

		uint64_t encoded = framebuffer[pixelIndex];
		uint32_t color = encoded & 0xffffffffull;

		surf2Dwrite(color, gl_colorbuffer, x * 4, y);
	});

	if(grid.thread_rank() == 0){
		stats->frameID++;
	}

}

extern "C" __global__
void kernel_rasterize_patches_runnin_thru(
	const Uniforms _uniforms,
	unsigned int* buffer,
	Patch* patches, uint32_t* numPatches,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	Allocator _allocator(buffer, 0);

	uniforms = _uniforms;
	allocator = &_allocator;

	// allocate framebuffer memory
	int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
	uint64_t* framebuffer = allocator->alloc<uint64_t*>(framebufferSize);

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
	});

	grid.sync();

	uint64_t t_00 = nanotime();

	rasterizePatches_runnin_thru(patches, numPatches, framebuffer, uniforms);
	grid.sync();

	uint64_t t_20 = nanotime();

	if(grid.thread_rank() == 0 && (stats->frameID % 100) == 0){
		stats->time_1 = float((t_20 - t_00) / 1000llu) / 1000.0f;
	}

	grid.sync();

	// transfer framebuffer to opengl texture
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){

		int x = pixelIndex % int(uniforms.width);
		int y = pixelIndex / int(uniforms.width);

		uint64_t encoded = framebuffer[pixelIndex];
		uint32_t color = encoded & 0xffffffffull;

		surf2Dwrite(color, gl_colorbuffer, x * 4, y);
	});

	if(grid.thread_rank() == 0){
		stats->frameID++;
	}

}
