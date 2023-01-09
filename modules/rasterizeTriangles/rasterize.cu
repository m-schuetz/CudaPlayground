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
	Point* points;
};

struct Lines{
	unsigned int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	Point* vertices;
};

struct Triangles{
	int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	Point* vertices;
};

void rasterizeTriangles(Triangles* triangles, uint64_t* framebuffer){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	
	// if(false)
	{
		__shared__ int shared_triangleIndex[1];

		block.sync();

		int loop_max = 10'000;
		for(int loop_i = 0; loop_i < loop_max; loop_i++){
			
			block.sync();
			if(block.thread_rank() == 0){
				shared_triangleIndex[0] = atomicAdd(&triangles->count, -3) / 3;
			}
			block.sync();

			int triangleIndex = shared_triangleIndex[0];

			if(triangleIndex < 0) break;

			auto toScreenCoord = [&](Point p){
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

			Point v0 = triangles->vertices[3 * triangleIndex + 0];
			Point v1 = triangles->vertices[3 * triangleIndex + 1];
			Point v2 = triangles->vertices[3 * triangleIndex + 2];

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
				if(w > 0.0) continue;
			}

			float min_x = min(min(p0.x, p1.x), p2.x);
			float min_y = min(min(p0.y, p1.y), p2.y);
			float max_x = max(max(p0.x, p1.x), p2.x);
			float max_y = max(max(p0.y, p1.y), p2.y);
			int size_x = ceil(max_x) - floor(min_x);
			int size_y = ceil(max_y) - floor(min_y);
			int numFragments = size_x * size_y;

			int numProcessedSamples = 0;
			for(int fragOffset = 0; fragOffset < numFragments; fragOffset += block.num_threads()){

				if(numProcessedSamples > 1'000) break;

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

				int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
				int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;

				pixelID = max(pixelID, 0);
				pixelID = min(pixelID, int(uniforms.width * uniforms.height));

				if( (s >= 0.0) && (t >= 0.0) && (s + t <= 1.0) )
				{
					
					unsigned int depth = *((int*)&p0.w);
					uint64_t pixel = (((unsigned long long int)depth) << 32) | v0.color;

					// uint64_t pixel = v0.color;

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

	{
		int numTriangles = 1;
		int numVertices = 3 * numTriangles;
		Triangles* triangles = allocator.alloc<Triangles*>(sizeof(Triangles) + sizeof(Point) * numVertices);

		triangles->count = numVertices;
		triangles->instanceCount = 1;

		rasterizeTriangles(triangles, framebuffer);
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
