#pragma once

#include "globals.cuh"
#include "texture.cuh"
#include "more_math.cuh"

struct Triangles{
	int numTriangles;
	float3* positions;
	float2* uvs;
	uint32_t* colors;
};


struct RasterizationSettings{
	Texture* texture = nullptr;
	int colorMode = COLORMODE_TRIANGLE_ID;
	mat4 world;
	mat4 view;
	mat4 proj;
	mat4 transform;
	float width; 
	float height;
};

uint32_t sample_nearest(float2 uv, Texture* texture){
	int tx = int(uv.x * texture->width) % texture->width;
	int ty = int(uv.y * texture->height) % texture->height;
	ty = texture->height - ty;

	int texelIndex = tx + texture->width * ty;
	uint32_t texel = texture->data[texelIndex];

	return texel;
}

uint32_t sample_linear(float2 uv, Texture* texture){
	float width = texture->width;
	float height = texture->height;

	float tx = uv.x * width;
	float ty = height - uv.y * height;

	int x0 = clamp(floor(tx), 0.0f, width - 1.0f);
	int x1 = clamp(ceil(tx) , 0.0f, width - 1.0f);
	int y0 = clamp(floor(ty), 0.0f, height - 1.0f);
	int y1 = clamp(ceil(ty) , 0.0f, height - 1.0f);
	float wx = tx - floor(tx);
	float wy = ty - floor(ty);

	float w00 = (1.0 - wx) * (1.0 - wy);
	float w10 = wx * (1.0 - wy);
	float w01 = (1.0 - wx) * wy;
	float w11 = wx * wy;

	uint8_t* c00 = (uint8_t*)&texture->data[x0 + y0 * texture->width];
	uint8_t* c10 = (uint8_t*)&texture->data[x1 + y0 * texture->width];
	uint8_t* c01 = (uint8_t*)&texture->data[x0 + y1 * texture->width];
	uint8_t* c11 = (uint8_t*)&texture->data[x1 + y1 * texture->width];

	uint32_t color;
	uint8_t* rgb = (uint8_t*)&color;

	rgb[0] = c00[0] * w00 + c10[0] * w10 + c01[0] * w01 + c11[0] * w11;
	rgb[1] = c00[1] * w00 + c10[1] * w10 + c01[1] * w01 + c11[1] * w11;
	rgb[2] = c00[2] * w00 + c10[2] * w10 + c01[2] * w01 + c11[2] * w11;

	return color;
}

// rasterizes triangles in a block-wise fashion
// - each block grabs a triangle
// - all threads of that block process different fragments of the triangle
// - <framebuffer> stores interleaved 32bit depth and color values
// - The closest fragments are rendered via atomicMin on a combined 64bit depth&color integer
//   atomicMin(&framebuffer[pixelIndex], (depth << 32 | color)); 
// see http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html#algo3
void rasterizeTriangles(Triangles* triangles, uint64_t* framebuffer, RasterizationSettings settings){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	Texture* texture = settings.texture;
	int colorMode = settings.colorMode;
	
	mat4 transform = settings.proj * settings.view * settings.world;

	uint32_t& processedTriangles = *allocator->alloc<uint32_t*>(4);
	if(grid.thread_rank() == 0){
		processedTriangles = 0;
	}
	grid.sync();

	float3 lightPos = {-10.0f, 10.0f, 10.0f};
	// float3 L = {10.0f, 10.0f, 10.0f};

	// if(uniforms.vrEnabled){
	// 	float4 leftPos = uniforms.vr_left_view_inv * float4{0.0, 0.0f, 0.0f, 1.0f};

	// 	lightPos = {leftPos.x, leftPos.y, leftPos.z};
	// }

	auto toScreenCoord = [&](float3 p){
		float4 pos = transform * float4{p.x, p.y, p.z, 1.0f};

		pos.x = pos.x / pos.w;
		pos.y = pos.y / pos.w;
		// pos.z = pos.z / pos.w;

		float4 imgPos = {
			(pos.x * 0.5f + 0.5f) * settings.width, 
			(pos.y * 0.5f + 0.5f) * settings.height,
			pos.z, 
			pos.w
		};

		return imgPos;
	};

	for_blockwise(triangles->numTriangles, [&](int sh_triangleIndex){
		// project x/y to pixel coords
		// z: whatever 
		// w: linear depth
		

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
		if(p0.w < 0.0 || p1.w < 0.0 || p2.w < 0.0) return;


		float3 v01 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
		float3 v02 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
		float3 N = normalize(cross(v02, v01));

		// if(sh_triangleIndex == 0 && block.thread_rank() == 0){
		// 	printf("%f, %f, %f \n", N.x, N.y, N.z);
		// }

		float2 p01 = {p1.x - p0.x, p1.y - p0.y};
		float2 p02 = {p2.x - p0.x, p2.y - p0.y};

		// auto cross = [](float2 a, float2 b){ return a.x * b.y - a.y * b.x; };
		// auto cross = [](float3 a, float3 b){ return a.y * b.z - a.y * b.x; };

		{// backface culling
			float w = cross(p01, p02);
			// if(w < 0.0) return;
		}

		// compute screen-space bounding rectangle
		float min_x = min(min(p0.x, p1.x), p2.x);
		float min_y = min(min(p0.y, p1.y), p2.y);
		float max_x = max(max(p0.x, p1.x), p2.x);
		float max_y = max(max(p0.y, p1.y), p2.y);

		// clamp to screen
		min_x = clamp(min_x, 0.0f, settings.width);
		min_y = clamp(min_y, 0.0f, settings.height);
		max_x = clamp(max_x, 0.0f, settings.width);
		max_y = clamp(max_y, 0.0f, settings.height);

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
			float s = cross(sample, p02) / cross(p01, p02);
			float t = cross(p01, sample) / cross(p01, p02);
			float v = 1.0 - (s + t);

			int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
			int pixelID = pixelCoords.x + pixelCoords.y * settings.width;
			pixelID = clamp(pixelID, 0, int(settings.width * settings.height) - 1);

			if(s >= 0.0)
			if(t >= 0.0)
			if(s + t <= 1.0)
			{
				uint8_t* v0_rgba = (uint8_t*)&triangles->colors[i0];
				uint8_t* v1_rgba = (uint8_t*)&triangles->colors[i1];
				uint8_t* v2_rgba = (uint8_t*)&triangles->colors[i2];

				float2 v0_uv = triangles->uvs[i0] / p0.z;
				float2 v1_uv = triangles->uvs[i1] / p1.z;
				float2 v2_uv = triangles->uvs[i2] / p2.z;
				float2 uv = {
					v * v0_uv.x + s * v1_uv.x + t * v2_uv.x,
					v * v0_uv.y + s * v1_uv.y + t * v2_uv.y
				};
				float repz = v * (1.0f / p0.z) + s * (1.0f / p1.z) + t * (1.0f / p2.z);
				uv.x = uv.x / repz;
				uv.y = uv.y / repz;

				uint32_t color;
				uint8_t* rgb = (uint8_t*)&color;

				// { // color by vertex color
				// 	rgb[0] = v * v0_rgba[0] + s * v1_rgba[0] + t * v2_rgba[0];
				// 	rgb[1] = v * v0_rgba[1] + s * v1_rgba[1] + t * v2_rgba[1];
				// 	rgb[2] = v * v0_rgba[2] + s * v1_rgba[2] + t * v2_rgba[2];
				// }

				// colorMode = COLORMODE_UV;
				// colorMode = COLORMODE_DEPTH;

				float depth = v * p0.w + s * p1.w + t * p2.w;

				if(colorMode == COLORMODE_TEXTURE && texture != nullptr){
					// TEXTURE
					int tx = int(uv.x * texture->width) % texture->width;
					int ty = int(uv.y * texture->height) % texture->height;
					ty = texture->height - ty;

					int texelIndex = tx + texture->width * ty;
					uint32_t texel = texture->data[texelIndex];
					uint8_t* texel_rgb = (uint8_t*)&texel;

					if(uniforms.sampleMode == SAMPLEMODE_NEAREST){
						color = sample_nearest(uv, texture);
					}else if(uniforms.sampleMode == SAMPLEMODE_LINEAR){
						color = sample_linear(uv, texture);
					}
				}else if(colorMode == COLORMODE_UV && triangles->uvs != nullptr){
					// UV
					rgb[0] = 255.0f * uv.x;
					rgb[1] = 255.0f * uv.y;
					rgb[2] = 0;
				}else if(colorMode == COLORMODE_TRIANGLE_ID){
					// TRIANGLE INDEX
					color = sh_triangleIndex * 123456;
				}else if(colorMode == COLORMODE_TIME || colorMode == COLORMODE_TIME_NORMALIZED){
					// TIME
					uint64_t nanotime;
					asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime));
					color = (nanotime - nanotime_start) % 0x00ffffffull;
				}else if(colorMode == COLORMODE_VERTEXCOLOR){
					color = triangles->colors[i0];
				}
				else if(colorMode == COLORMODE_DEPTH){
					color = 0;
					float w = log2f(depth);
					w = 80.0f * w;

					// w = 200.0f * (s + t + v);

					rgb[0] = w;
					rgb[1] = w;
					rgb[2] = w;

					// rgb[0] = 255 * s;
					// rgb[1] = 255 * t;
					// rgb[2] = 255 * v;
				}
				else{
					// WHATEVER
					color = sh_triangleIndex * 123456;
				}

				// float3 L = {10.0f, 10.0f, 10.0f};
				float3 L = normalize(v0 - lightPos);

				float lambertian = max(dot(N, L), 0.0);

				// rgb[0] = 255.0f *  N.x;
				// rgb[1] = 255.0f *  N.y;
				// rgb[2] = 255.0f *  N.z;

				float3 diffuse = {0.8f, 0.8f, 0.8f};
				float3 ambient = {0.5f, 0.5f, 0.5f};
				rgb[0] = clamp(rgb[0] * (lambertian * diffuse.x + ambient.x), 0.0f, 255.0f);
				rgb[1] = clamp(rgb[1] * (lambertian * diffuse.y + ambient.y), 0.0f, 255.0f);
				rgb[2] = clamp(rgb[2] * (lambertian * diffuse.z + ambient.z), 0.0f, 255.0f);

				
				uint64_t udepth = *((uint32_t*)&depth);
				uint64_t pixel = (udepth << 32ull) | color;

				atomicMin(&framebuffer[pixelID], pixel);
			}

			numProcessedSamples++;
		}
	});

	// {
	// 	__shared__ int sh_triangleIndex;

	// 	block.sync();

		// safety mechanism: each block draws at most <loop_max> triangles
		// int loop_max = 10'000;
		// for(int loop_i = 0; loop_i < loop_max; loop_i++){
			
		// 	// grab the index of the next unprocessed triangle
		// 	block.sync();
		// 	if(block.thread_rank() == 0){
		// 		sh_triangleIndex = atomicAdd(&processedTriangles, 1);
		// 	}
		// 	block.sync();

		// 	if(sh_triangleIndex >= triangles->numTriangles) break;

		// 	// project x/y to pixel coords
		// 	// z: whatever 
		// 	// w: linear depth
		// 	auto toScreenCoord = [&](float3 p){
		// 		float4 pos = transform * float4{p.x, p.y, p.z, 1.0f};

		// 		pos.x = pos.x / pos.w;
		// 		pos.y = pos.y / pos.w;
		// 		// pos.z = pos.z / pos.w;

		// 		float4 imgPos = {
		// 			(pos.x * 0.5f + 0.5f) * settings.width, 
		// 			(pos.y * 0.5f + 0.5f) * settings.height,
		// 			pos.z, 
		// 			pos.w
		// 		};

		// 		return imgPos;
		// 	};

		// 	int i0 = 3 * sh_triangleIndex + 0;
		// 	int i1 = 3 * sh_triangleIndex + 1;
		// 	int i2 = 3 * sh_triangleIndex + 2;
			
		// 	float3 v0 = triangles->positions[i0];
		// 	float3 v1 = triangles->positions[i1];
		// 	float3 v2 = triangles->positions[i2];

		// 	float4 p0 = toScreenCoord(v0);
		// 	float4 p1 = toScreenCoord(v1);
		// 	float4 p2 = toScreenCoord(v2);

		// 	// cull a triangle if one of its vertices is closer than depth 0
		// 	if(p0.w < 0.0 || p1.w < 0.0 || p2.w < 0.0) continue;


		// 	float3 v01 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
		// 	float3 v02 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
		// 	float3 N = normalize(cross(v02, v01));

		// 	// if(sh_triangleIndex == 0 && block.thread_rank() == 0){
		// 	// 	printf("%f, %f, %f \n", N.x, N.y, N.z);
		// 	// }

		// 	float2 p01 = {p1.x - p0.x, p1.y - p0.y};
		// 	float2 p02 = {p2.x - p0.x, p2.y - p0.y};

		// 	// auto cross = [](float2 a, float2 b){ return a.x * b.y - a.y * b.x; };
		// 	// auto cross = [](float3 a, float3 b){ return a.y * b.z - a.y * b.x; };

		// 	{// backface culling
		// 		float w = cross(p01, p02);
		// 		if(w < 0.0) continue;
		// 	}

		// 	// compute screen-space bounding rectangle
		// 	float min_x = min(min(p0.x, p1.x), p2.x);
		// 	float min_y = min(min(p0.y, p1.y), p2.y);
		// 	float max_x = max(max(p0.x, p1.x), p2.x);
		// 	float max_y = max(max(p0.y, p1.y), p2.y);

		// 	// clamp to screen
		// 	min_x = clamp(min_x, 0.0f, settings.width);
		// 	min_y = clamp(min_y, 0.0f, settings.height);
		// 	max_x = clamp(max_x, 0.0f, settings.width);
		// 	max_y = clamp(max_y, 0.0f, settings.height);

		// 	int size_x = ceil(max_x) - floor(min_x);
		// 	int size_y = ceil(max_y) - floor(min_y);
		// 	int numFragments = size_x * size_y;

		// 	// iterate through fragments in bounding rectangle and draw if within triangle
		// 	int numProcessedSamples = 0;
		// 	for(int fragOffset = 0; fragOffset < numFragments; fragOffset += block.num_threads()){

		// 		// safety mechanism: don't draw more than <x> pixels per thread
		// 		if(numProcessedSamples > 5'000) break;

		// 		int fragID = fragOffset + block.thread_rank();
		// 		int fragX = fragID % size_x;
		// 		int fragY = fragID / size_x;

		// 		float2 pFrag = {
		// 			floor(min_x) + float(fragX), 
		// 			floor(min_y) + float(fragY)
		// 		};
		// 		float2 sample = {pFrag.x - p0.x, pFrag.y - p0.y};

		// 		// v: vertex[0], s: vertex[1], t: vertex[2]
		// 		float s = cross(sample, p02) / cross(p01, p02);
		// 		float t = cross(p01, sample) / cross(p01, p02);
		// 		float v = 1.0 - (s + t);

		// 		int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
		// 		int pixelID = pixelCoords.x + pixelCoords.y * settings.width;
		// 		pixelID = clamp(pixelID, 0, int(settings.width * settings.height) - 1);

		// 		if(s >= 0.0)
		// 		if(t >= 0.0)
		// 		if(s + t <= 1.0)
		// 		{
		// 			uint8_t* v0_rgba = (uint8_t*)&triangles->colors[i0];
		// 			uint8_t* v1_rgba = (uint8_t*)&triangles->colors[i1];
		// 			uint8_t* v2_rgba = (uint8_t*)&triangles->colors[i2];

		// 			float2 v0_uv = triangles->uvs[i0] / p0.z;
		// 			float2 v1_uv = triangles->uvs[i1] / p1.z;
		// 			float2 v2_uv = triangles->uvs[i2] / p2.z;
		// 			float2 uv = {
		// 				v * v0_uv.x + s * v1_uv.x + t * v2_uv.x,
		// 				v * v0_uv.y + s * v1_uv.y + t * v2_uv.y
		// 			};
		// 			float repz = v * (1.0f / p0.z) + s * (1.0f / p1.z) + t * (1.0f / p2.z);
		// 			uv.x = uv.x / repz;
		// 			uv.y = uv.y / repz;

		// 			uint32_t color;
		// 			uint8_t* rgb = (uint8_t*)&color;

		// 			// { // color by vertex color
		// 			// 	rgb[0] = v * v0_rgba[0] + s * v1_rgba[0] + t * v2_rgba[0];
		// 			// 	rgb[1] = v * v0_rgba[1] + s * v1_rgba[1] + t * v2_rgba[1];
		// 			// 	rgb[2] = v * v0_rgba[2] + s * v1_rgba[2] + t * v2_rgba[2];
		// 			// }

		// 			if(colorMode == COLORMODE_TEXTURE && texture != nullptr){
		// 				// TEXTURE
		// 				int tx = int(uv.x * texture->width) % texture->width;
		// 				int ty = int(uv.y * texture->height) % texture->height;
		// 				ty = texture->height - ty;

		// 				int texelIndex = tx + texture->width * ty;
		// 				uint32_t texel = texture->data[texelIndex];
		// 				uint8_t* texel_rgb = (uint8_t*)&texel;

		// 				if(uniforms.sampleMode == SAMPLEMODE_NEAREST){
		// 					color = sample_nearest(uv, texture);
		// 				}else if(uniforms.sampleMode == SAMPLEMODE_LINEAR){
		// 					color = sample_linear(uv, texture);
		// 				}
		// 			}else if(colorMode == COLORMODE_UV && triangles->uvs != nullptr){
		// 				// UV
		// 				rgb[0] = 255.0f * uv.x;
		// 				rgb[1] = 255.0f * uv.y;
		// 				rgb[2] = 0;
		// 			}else if(colorMode == COLORMODE_TRIANGLE_ID){
		// 				// TRIANGLE INDEX
		// 				color = sh_triangleIndex * 123456;
		// 			}else if(colorMode == COLORMODE_TIME || colorMode == COLORMODE_TIME_NORMALIZED){
		// 				// TIME
		// 				uint64_t nanotime;
		// 				asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime));
		// 				color = (nanotime - nanotime_start) % 0x00ffffffull;
		// 			}else if(colorMode == COLORMODE_VERTEXCOLOR){
		// 				color = triangles->colors[i0];
		// 			}else{
		// 				// WHATEVER
		// 				color = sh_triangleIndex * 123456;
		// 			}

		// 			// float3 L = {10.0f, 10.0f, 10.0f};
		// 			float3 L = normalize(v0 - lightPos);

		// 			float lambertian = max(dot(N, L), 0.0);

		// 			// rgb[0] = 255.0f *  N.x;
		// 			// rgb[1] = 255.0f *  N.y;
		// 			// rgb[2] = 255.0f *  N.z;

		// 			float3 diffuse = {0.8f, 0.8f, 0.8f};
		// 			float3 ambient = {0.2f, 0.2f, 0.2f};
		// 			rgb[0] = rgb[0] * (lambertian * diffuse.x + ambient.x);
		// 			rgb[1] = rgb[1] * (lambertian * diffuse.y + ambient.y);
		// 			rgb[2] = rgb[2] * (lambertian * diffuse.z + ambient.z);

		// 			float depth = v * p0.w + s * p1.w + t * p2.w;
		// 			uint64_t udepth = *((uint32_t*)&depth);
		// 			uint64_t pixel = (udepth << 32ull) | color;

		// 			atomicMin(&framebuffer[pixelID], pixel);
		// 		}

		// 		numProcessedSamples++;
		// 	}
		// }
	// }
}

Triangles* createGroundPlane(int cells){
	// int cells = 50;
	int numTriangles     = cells * cells * 2;
	int numVertices      = 3 * numTriangles;
	Triangles* triangles = allocator->alloc<Triangles*>(sizeof(Triangles));
	triangles->positions = allocator->alloc<float3*  >(sizeof(float3) * numVertices);
	triangles->uvs       = allocator->alloc<float2*  >(sizeof(float2) * numVertices);
	triangles->colors    = allocator->alloc<uint32_t*>(sizeof(uint32_t) * numVertices);

	triangles->numTriangles = numTriangles;
	
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

		float s = 10.0f;
		float height = 0.0f;
		
		triangles->positions[offset + 0] = {s * u0 - s * 0.5f, s * v0 - s * 0.5f, height};
		triangles->positions[offset + 1] = {s * u1 - s * 0.5f, s * v0 - s * 0.5f, height};
		triangles->positions[offset + 2] = {s * u1 - s * 0.5f, s * v1 - s * 0.5f, height};
		triangles->positions[offset + 3] = {s * u0 - s * 0.5f, s * v0 - s * 0.5f, height};
		triangles->positions[offset + 4] = {s * u1 - s * 0.5f, s * v1 - s * 0.5f, height};
		triangles->positions[offset + 5] = {s * u0 - s * 0.5f, s * v1 - s * 0.5f, height};

		// triangles->uvs[offset + 0] = {0.0f, 0.0f};
		// triangles->uvs[offset + 1] = {1.0f, 0.0f};
		// triangles->uvs[offset + 2] = {1.0f, 1.0f};
		// triangles->uvs[offset + 3] = {0.0f, 0.0f};
		// triangles->uvs[offset + 4] = {1.0f, 1.0f};
		// triangles->uvs[offset + 5] = {0.0f, 1.0f};
		triangles->uvs[offset + 0] = {u0, v0};
		triangles->uvs[offset + 1] = {u1, v0};
		triangles->uvs[offset + 2] = {u1, v1};
		triangles->uvs[offset + 3] = {u0, v0};
		triangles->uvs[offset + 4] = {u1, v1};
		triangles->uvs[offset + 5] = {u0, v1};
	});

	return triangles;
}

Triangles* createSphere(int segments){

	auto grid = cg::this_grid();

	int numTriangles     = 2 * segments * (segments);
	int numVertices      = 3 * numTriangles;
	Triangles* triangles = allocator->alloc<Triangles*>(sizeof(Triangles));
	triangles->positions = allocator->alloc<float3*  >(sizeof(float3) * numVertices);
	triangles->uvs       = allocator->alloc<float2*  >(sizeof(float2) * numVertices);
	triangles->colors    = allocator->alloc<uint32_t*>(sizeof(uint32_t) * numVertices);
	triangles->numTriangles = numTriangles;

	// s, t between [0, 1.0]
	// s horizontal
	// t vertical
	auto sampleSphere = [](float s, float t){
		constexpr float PI = 3.1415f;

		float x = cos(2.0f * PI * s) * cos(PI * (t - 0.5f));
		float z = sin(2.0f * PI * s) * cos(PI * (t - 0.5f));
		float y = sin(PI * (t - 0.5f));

		return float3{x, y, z};
	};


	int numTrianglesAdded = 0;

	if(grid.thread_rank() == 0)
	for(int seg_vert = 0; seg_vert < segments; seg_vert++)
	for(int seg_hori = 0; seg_hori < segments; seg_hori++)
	{

		float s_0 = float(seg_hori + 0) / segments;
		float s_1 = float(seg_hori + 1) / segments;
		float t_0 = float(seg_vert + 0) / segments;
		float t_1 = float(seg_vert + 1) / segments;

		float r = 1.0f;
		float3 p00 = r * sampleSphere(s_0, t_0);
		float3 p01 = r * sampleSphere(s_0, t_1);
		float3 p10 = r * sampleSphere(s_1, t_0);
		float3 p11 = r * sampleSphere(s_1, t_1);

		int targetIndex = numTrianglesAdded;
	
		triangles->positions[3 * targetIndex + 0] = p00;
		triangles->positions[3 * targetIndex + 1] = p10;
		triangles->positions[3 * targetIndex + 2] = p11;
		triangles->positions[3 * targetIndex + 3] = p00;
		triangles->positions[3 * targetIndex + 4] = p11;
		triangles->positions[3 * targetIndex + 5] = p01;

		triangles->uvs[3 * targetIndex + 0] = {s_0, t_0};
		triangles->uvs[3 * targetIndex + 1] = {s_1, t_0};
		triangles->uvs[3 * targetIndex + 2] = {s_1, t_1};
		triangles->uvs[3 * targetIndex + 3] = {s_0, t_0};
		triangles->uvs[3 * targetIndex + 4] = {s_1, t_1};
		triangles->uvs[3 * targetIndex + 5] = {s_0, t_1};

		triangles->colors[3 * targetIndex + 0] = 0x00aaaaaa;
		triangles->colors[3 * targetIndex + 1] = 0x00aaaaaa;
		triangles->colors[3 * targetIndex + 2] = 0x00aaaaaa;
		triangles->colors[3 * targetIndex + 3] = 0x00aaaaaa;
		triangles->colors[3 * targetIndex + 4] = 0x00aaaaaa;
		triangles->colors[3 * targetIndex + 5] = 0x00aaaaaa;

		numTrianglesAdded += 2;
	}

	return triangles;
}