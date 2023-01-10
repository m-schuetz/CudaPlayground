#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

namespace cg = cooperative_groups;

Uniforms uniforms;
Allocator* allocator;

constexpr float PI = 3.1415;

float4 operator*(const mat4& a, const float4& b){
	return make_float4(
		dot(a.rows[0], b),
		dot(a.rows[1], b),
		dot(a.rows[2], b),
		dot(a.rows[3], b)
	);
}

struct Point{
	float x;
	float y;
	float z;
	unsigned int color;
};

struct Points{
	unsigned int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	Point points[10'000'000];
};

struct Lines{
	unsigned int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	Point vertices[10'000'000];
};

struct Triangles{
	int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	float3* positions;
	float2* uvs;
	uint32_t* colors;
};

struct Texture{
	int width;
	int height;
	uint32_t* data;
};

void rasterizeTriangles(Triangles* triangles, uint64_t* framebuffer, Texture* texture = nullptr){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint32_t& processedTriangles = *allocator->alloc<uint32_t*>(4);
	if(grid.thread_rank() == 0){
		processedTriangles = 0;
	}
	grid.sync();

	{
		__shared__ int sh_triangleIndex;

		block.sync();

		int loop_max = 10'000;
		for(int loop_i = 0; loop_i < loop_max; loop_i++){
			
			block.sync();
			if(block.thread_rank() == 0){
				sh_triangleIndex = atomicAdd(&processedTriangles, 1);
			}
			block.sync();

			if(sh_triangleIndex >= triangles->count / 3) break;

			// project x/y to pixel coords
			// z is whatever, w is the linear depth
			auto toScreenCoord = [&](float3 p){
				float4 pos = uniforms.transform * float4{p.x, p.y, p.z, 1.0f};

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

			auto isInside = [&](float4 p){
				if(p.x < 0 || p.x >= uniforms.width) return false;
				if(p.y < 0 || p.y >= uniforms.height) return false;

				return true;
			};

			if(!isInside(p0) || !isInside(p1) || !isInside(p2)) continue;

			float2 v01 = float2{p1.x - p0.x, p1.y - p0.y};
			float2 v02 = float2{p2.x - p0.x, p2.y - p0.y};

			auto cross = [](float2 a, float2 b){ return a.x * b.y - a.y * b.x; };

			{// backface culling
				float w = cross(v01, v02);
				if(w < 0.0) continue;
			}

			// compute screen-space bounding rectangle
			float min_x = min(min(p0.x, p1.x), p2.x);
			float min_y = min(min(p0.y, p1.y), p2.y);
			float max_x = max(max(p0.x, p1.x), p2.x);
			float max_y = max(max(p0.y, p1.y), p2.y);
			int size_x = ceil(max_x) - floor(min_x);
			int size_y = ceil(max_y) - floor(min_y);
			int numFragments = size_x * size_y;

			// iterate through fragments in bounding rectangle and draw if within triangle
			int numProcessedSamples = 0;
			for(int fragOffset = 0; fragOffset < numFragments; fragOffset += block.num_threads()){

				// safety mechanism: don't draw more than 1k pixels per triangle
				if(numProcessedSamples > 10'000) break;

				int fragID = fragOffset + block.thread_rank();
				int fragX = fragID % size_x;
				int fragY = fragID / size_x;

				float2 pFrag = {
					floor(min_x) + float(fragX), 
					floor(min_y) + float(fragY)
				};
				float2 sample = {pFrag.x - p0.x, pFrag.y - p0.y};

				float s = cross(sample, v02) / cross(v01, v02);
				float t = cross(v01, sample) / cross(v01, v02);
				float v = 1.0 - (s + t);

				// v: vertex[0], s: vertex[1], t: vertex[2]

				int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
				int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;

				pixelID = max(pixelID, 0);
				pixelID = min(pixelID, int(uniforms.width * uniforms.height));

				if( (s >= 0.0) && (t >= 0.0) && (s + t <= 1.0) )
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

					// { // color by uv
					// 	rgb[0] = 255.0f * uv.x;
					// 	rgb[1] = 255.0f * uv.y;
					// 	rgb[2] = 0;
					// 	rgb[3] = 255;
					// }

					if(texture)
					{ // color by texture
						int tx = int(uv.x * texture->width) % texture->width;
						int ty = int(uv.y * texture->height) % texture->height;
						ty = texture->height - ty;

						int texelIndex = tx + texture->width * ty;
						uint32_t texel = texture->data[texelIndex];
						uint8_t* texel_rgb = (uint8_t*)&texel;

						rgb[0] = texel_rgb[0];
						rgb[1] = texel_rgb[1];
						rgb[2] = texel_rgb[2];
					}else{
						rgb[0] = v * v0_rgba[0] + s * v1_rgba[0] + t * v2_rgba[0];
						rgb[1] = v * v0_rgba[1] + s * v1_rgba[1] + t * v2_rgba[1];
						rgb[2] = v * v0_rgba[2] + s * v1_rgba[2] + t * v2_rgba[2];
					}

					float depth = v * p0.w + s * p1.w + t * p2.w;
					uint64_t udepth = *((uint32_t*)&depth);
					uint64_t pixel = (udepth << 32ull) | color;

					atomicMin(&framebuffer[pixelID], pixel);
				}

				numProcessedSamples++;
			}


		}
	}
}

extern "C" __global__
void kernel(
	const Uniforms _uniforms,
	unsigned int* buffer,
	cudaSurfaceObject_t gl_colorbuffer,
	uint32_t numTriangles,
	float3* positions,
	float2* uvs,
	uint32_t* colors,
	uint32_t* textureData
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uniforms = _uniforms;

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	// if(grid.thread_rank() == 0){
	// 	printf("%i \n", numTriangles);
	// }

	// allocate framebuffer memory
	int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
	uint64_t* framebuffer = allocator->alloc<uint64_t*>(framebufferSize);

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		// depth:            7f800000 (Infinity)
		// background color: 00332211 (aabbggrr)
		framebuffer[pixelIndex] = 0x7f800000'00332211ull;;
	});

	grid.sync();

	// draw some custom defined triangles
	{
		// make a ground plane
		int cells = 20;
		int numTriangles     = cells * cells * 2;
		int numVertices      = 3 * numTriangles;
		Triangles* triangles = allocator->alloc<Triangles*>(sizeof(Triangles));
		triangles->positions = allocator->alloc<float3*  >(sizeof(float3) * numVertices);
		triangles->uvs       = allocator->alloc<float2*  >(sizeof(float2) * numVertices);
		triangles->colors    = allocator->alloc<uint32_t*>(sizeof(uint32_t) * numVertices);

		triangles->count = numVertices;
		triangles->instanceCount = 1;
		
		processRange(0, cells * cells, [&](int cellIndex){

			int cx = cellIndex % cells;
			int cy = cellIndex / cells;

			float u0 = float(cx + 0) / float(cells);
			float v0 = float(cy + 0) / float(cells);
			float u1 = float(cx + 1) / float(cells);
			float v1 = float(cy + 1) / float(cells);

			int offset = 6 * cellIndex;

			uint32_t color = 0;
			uint8_t* rgb = (uint8_t*)&color;
			rgb[0] = 255.0f * u0;
			rgb[1] = 255.0f * v0;
			rgb[2] = 0;

			triangles->positions[offset + 0] = {2.0 * u0 - 1.0, -0.7, 2.0 * v0 - 1.0};
			triangles->positions[offset + 2] = {2.0 * u1 - 1.0, -0.7, 2.0 * v0 - 1.0};
			triangles->positions[offset + 1] = {2.0 * u1 - 1.0, -0.7, 2.0 * v1 - 1.0};
			triangles->colors[offset + 0]    = color;
			triangles->colors[offset + 2]    = color;
			triangles->colors[offset + 1]    = color;

			triangles->positions[offset + 3] = {2.0 * u0 - 1.0, -0.7, 2.0 * v0 - 1.0};
			triangles->positions[offset + 5] = {2.0 * u1 - 1.0, -0.7, 2.0 * v1 - 1.0};
			triangles->positions[offset + 4] = {2.0 * u0 - 1.0, -0.7, 2.0 * v1 - 1.0};
			triangles->colors[offset + 3]    = color;
			triangles->colors[offset + 5]    = color;
			triangles->colors[offset + 4]    = color;
		});

		rasterizeTriangles(triangles, framebuffer);
	}

	{
		Triangles* triangles = allocator->alloc<Triangles*>(sizeof(Triangles));
		triangles->count = numTriangles * 3;
		triangles->instanceCount = 1;

		triangles->positions = positions;
		triangles->uvs = uvs;
		triangles->colors = colors;

		Texture texture;
		texture.width  = 1024;
		texture.height = 1024;
		texture.data   = textureData;

		rasterizeTriangles(triangles, framebuffer, &texture);
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


}
