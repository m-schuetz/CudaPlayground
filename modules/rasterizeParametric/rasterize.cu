#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

#include "math.cuh"

constexpr int MAX_PATCHES = 1'000'000;

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

namespace cg = cooperative_groups;

Uniforms uniforms;
Allocator* allocator;
uint64_t nanotime_start;

constexpr float PI = 3.1415;
constexpr uint32_t BACKGROUND_COLOR = 0x00332211ull;

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

// struct Patch{
// 	int gridsize;
// 	int x;
// 	int y;
// 	int dbg;
// };

struct Patch{
	float s_min;
	float s_max;
	float t_min;
	float t_max;
};

void generatePatches2(Patch* patches, uint32_t* numPatches, Uniforms& uniforms){

	auto grid = cg::this_grid();

	
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

	// SUBDIVIDE LARGE PATCHES
	// - if too large, divide and store in target
	// - if not too large, store in <patches>
	auto subdivide = [&](Patch* source, uint32_t* sourceCounter, Patch* target, uint32_t* targetCounter){

		processRange(*sourceCounter, [&](int index){
			Patch patch = source[index];

			float3 p_00 = sample(patch.s_min, patch.t_min);
			float3 p_01 = sample(patch.s_min, patch.t_max);
			float3 p_10 = sample(patch.s_max, patch.t_min);
			float3 p_11 = sample(patch.s_max, patch.t_max);

			float3 nodeMin = {
				min(min(p_00.x, p_01.x), min(p_10.x, p_11.x)),
				min(min(p_00.y, p_01.y), min(p_10.y, p_11.y)),
				min(min(p_00.z, p_01.z), min(p_10.z, p_11.z)),
			};
			float3 nodeMax = {
				max(max(p_00.x, p_01.x), max(p_10.x, p_11.x)),
				max(max(p_00.y, p_01.y), max(p_10.y, p_11.y)),
				max(max(p_00.z, p_01.z), max(p_10.z, p_11.z)),
			};
			bool isIntersectingFrustum = intersectsFrustum(uniforms.transform, nodeMin, nodeMax);

			if(!isIntersectingFrustum){
				return;
			}

			float4 ps_00 = toScreen(p_00, uniforms);
			float4 ps_01 = toScreen(p_01, uniforms);
			float4 ps_10 = toScreen(p_10, uniforms);
			float4 ps_11 = toScreen(p_11, uniforms);

			float min_x = min(min(ps_00.x, ps_01.x), min(ps_10.x, ps_11.x));
			float max_x = max(max(ps_00.x, ps_01.x), max(ps_10.x, ps_11.x));
			float min_y = min(min(ps_00.y, ps_01.y), min(ps_10.y, ps_11.y));
			float max_y = max(max(ps_00.y, ps_01.y), max(ps_10.y, ps_11.y));

			float s_x = max_x - min_x;
			float s_y = max_y - min_y;
			float area = s_x * s_y;

			if(area > 64 * 64){
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
				// small enough, add to final list
				uint32_t targetIndex = atomicAdd(numPatches, 4);
				patches[targetIndex] = patch;
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

	}

}

void generatePatches(Patch* patches, uint32_t* numPatches, Uniforms& uniforms){

	int gridsize = 64;

	processRange(0, gridsize * gridsize, [&](int index){

		int patch_x = index % gridsize;
		int patch_y = index / gridsize;

		int ux_0 = patch_x + 0;
		int ux_1 = patch_x + 1;
		int uy_0 = patch_y + 0;
		int uy_1 = patch_y + 1;

		float u_0 = float(ux_0) / float(gridsize);
		float u_1 = float(ux_1) / float(gridsize);
		float v_0 = float(uy_0) / float(gridsize);
		float v_1 = float(uy_1) / float(gridsize);

		float3 p_00 = sample(u_0, v_0);
		float3 p_01 = sample(u_0, v_1);
		float3 p_10 = sample(u_1, v_0);
		float3 p_11 = sample(u_1, v_1);

		float3 nodeMin = {
			min(min(p_00.x, p_01.x), min(p_10.x, p_11.x)),
			min(min(p_00.y, p_01.y), min(p_10.y, p_11.y)),
			min(min(p_00.z, p_01.z), min(p_10.z, p_11.z)),
		};
		float3 nodeMax = {
			max(max(p_00.x, p_01.x), max(p_10.x, p_11.x)),
			max(max(p_00.y, p_01.y), max(p_10.y, p_11.y)),
			max(max(p_00.z, p_01.z), max(p_10.z, p_11.z)),
		};
		bool isIntersectingFrustum = intersectsFrustum(uniforms.transform, nodeMin, nodeMax);

		if(!isIntersectingFrustum){
			return;
		}

		float4 ps_00 = toScreen(p_00, uniforms);
		float4 ps_01 = toScreen(p_01, uniforms);
		float4 ps_10 = toScreen(p_10, uniforms);
		float4 ps_11 = toScreen(p_11, uniforms);

		float min_x = min(min(ps_00.x, ps_01.x), min(ps_10.x, ps_11.x));
		float max_x = max(max(ps_00.x, ps_01.x), max(ps_10.x, ps_11.x));
		float min_y = min(min(ps_00.y, ps_01.y), min(ps_10.y, ps_11.y));
		float max_y = max(max(ps_00.y, ps_01.y), max(ps_10.y, ps_11.y));

		float s_x = max_x - min_x;
		float s_y = max_y - min_y;
		float area = s_x * s_y;

		float4 p = ps_00;
		int x = p.x;
		int y = p.y;
		float depth = p.w;

		uint32_t R = 255.0f * u_0 / (2.0 * 3.14);
		uint32_t G = 255.0f * v_0 / 3.14;
		uint32_t B = 0;

		if(area < 32 * 32){
			R = 0;
			G = 255;
			B = 0;

			Patch patch;
			patch.s_min = u_0;
			patch.s_max = u_1;
			patch.t_min = v_0;
			patch.t_max = v_1;
			// patch.gridsize = gridsize;
			// patch.x = patch_x;
			// patch.y = patch_y;
			// patch.dbg = 0;

			uint32_t patchIndex = atomicAdd(numPatches, 1);
			patches[patchIndex] = patch;

		}else if(area < 64 * 64){
			R = 0;
			G = 0;
			B = 255;

			uint32_t patchIndex = atomicAdd(numPatches, 4);
			
			int i = 0; 
			for(int px : {0, 1})
			for(int py : {0, 1})
			{
				Patch patch;
				patch.s_min = float(2 * patch_x + px + 0) / float(2 * gridsize);
				patch.s_max = float(2 * patch_x + px + 1) / float(2 * gridsize);
				patch.t_min = float(2 * patch_y + py + 0) / float(2 * gridsize);
				patch.t_max = float(2 * patch_y + py + 1) / float(2 * gridsize);

				patches[patchIndex + i] = patch;

				i++;
			}
		}else if(area < 128 * 128){
			R = 255;
			G = 255;
			B = 0;

			uint32_t patchIndex = atomicAdd(numPatches, 16);
			
			int i = 0; 
			for(int px : {0, 1, 2, 3})
			for(int py : {0, 1, 2, 3})
			{
				Patch patch;
				// patch.gridsize = 4 * gridsize;
				// patch.x = 4 * patch_x + px;
				// patch.y = 4 * patch_y + py;
				// patch.dbg = 2;
				patch.s_min = float(4 * patch_x + px + 0) / float(4 * gridsize);
				patch.s_max = float(4 * patch_x + px + 1) / float(4 * gridsize);
				patch.t_min = float(4 * patch_y + py + 0) / float(4 * gridsize);
				patch.t_max = float(4 * patch_y + py + 1) / float(4 * gridsize);

				patches[patchIndex + i] = patch;

				i++;
			}
		}else if(area < 256 * 256){
			R = 255;
			G = 255;
			B = 255;

			uint32_t patchIndex = atomicAdd(numPatches, 64);
			
			int i = 0; 
			for(int px : {0, 1, 2, 3, 4, 5, 6, 7})
			for(int py : {0, 1, 2, 3, 4, 5, 6, 7})
			{
				Patch patch;
				patch.s_min = float(8 * patch_x + px + 0) / float(8 * gridsize);
				patch.s_max = float(8 * patch_x + px + 1) / float(8 * gridsize);
				patch.t_min = float(8 * patch_y + py + 0) / float(8 * gridsize);
				patch.t_max = float(8 * patch_y + py + 1) / float(8 * gridsize);

				patches[patchIndex + i] = patch;

				i++;
			}
		}else if(area < 512 * 512){
			R = 255;
			G = 0;
			B = 0;

			uint32_t patchIndex = atomicAdd(numPatches, 4 * 64);
			
			int i = 0; 
			for(int px : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
			for(int py : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
			{
				Patch patch;
				patch.s_min = float(16 * patch_x + px + 0) / float(16 * gridsize);
				patch.s_max = float(16 * patch_x + px + 1) / float(16 * gridsize);
				patch.t_min = float(16 * patch_y + py + 0) / float(16 * gridsize);
				patch.t_max = float(16 * patch_y + py + 1) / float(16 * gridsize);

				patches[patchIndex + i] = patch;

				i++;
			}
		}else{
			
			R = 255;
			G = 0;
			B = 0;

			uint32_t patchIndex = atomicAdd(numPatches, 16 * 64);
			
			int i = 0; 
			for(int px = 0; px < 32; px++)
			for(int py = 0; py < 32; py++)
			{
				Patch patch;
				patch.s_min = float(32 * patch_x + px + 0) / float(32 * gridsize);
				patch.s_max = float(32 * patch_x + px + 1) / float(32 * gridsize);
				patch.t_min = float(32 * patch_y + py + 0) / float(32 * gridsize);
				patch.t_max = float(32 * patch_y + py + 1) / float(32 * gridsize);

				patches[patchIndex + i] = patch;

				i++;
			}
		}
	});
}

void rasterizePatches(Patch* patches, uint32_t* numPatches, uint64_t* framebuffer, Uniforms& uniforms){

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

extern "C" __global__
void kernel(
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

	// allocate framebuffer memory
	int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
	uint64_t* framebuffer = allocator->alloc<uint64_t*>(framebufferSize);

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
	});
	
	// Patch* patches = allocator->alloc<Patch*>(MAX_PATCHES);
	// uint32_t& numPatches = *allocator->alloc<uint32_t*>(MAX_PATCHES);

	grid.sync();

	// if(grid.thread_rank() == 0){
	// 	*numPatches = 0;
	// }

	grid.sync();

	uint64_t t_00 = nanotime();

	rasterizePatches(patches, numPatches, framebuffer, uniforms);

	grid.sync();

	uint64_t t_20 = nanotime();

	if(grid.thread_rank() == 0 && (stats->frameID % 100) == 0){

		stats->time_1 = float((t_20 - t_00) / 1000llu) / 1000.0f;
		// stats->time_1 = float((t_20 - t_10) / 1000llu) / 1000.0f;

		stats->time_2 = float(*numPatches);
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
