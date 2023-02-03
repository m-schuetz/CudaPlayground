#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

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

struct Intersection{
	float3 position;
	float distance;
	
	bool intersects(){
		return distance > 0.0f && distance != Infinity;
	}
};

Intersection rayPlane(float3 origin, float3 direction, float3 planeNormal, float planeDistance){

	float denominator = dot(planeNormal, direction);

	if(denominator == 0){
		Intersection I;
		I.distance = Infinity;

		return I;
	}else{
		float distance = - (dot(origin, planeNormal) + planeDistance ) / denominator;

		Intersection I;
		I.distance = distance;
		I.position = origin + direction * distance;

		return I;
	}

}

namespace cg = cooperative_groups;

Uniforms uniforms;
Allocator* allocator;
uint64_t nanotime_start;

constexpr float PI = 3.1415;
constexpr uint32_t BACKGROUND_COLOR = 0x00332211ull;

struct Triangles{
	int numTriangles;
	float3* positions;
	float2* uvs;
	uint32_t* colors;
};

struct Texture{
	int width;
	int height;
	uint32_t* data;
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

	{
		__shared__ int sh_triangleIndex;

		block.sync();

		// safety mechanism: each block draws at most <loop_max> triangles
		int loop_max = 10'000;
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
				// pos.z = pos.z / pos.w;

				float4 imgPos = {
					(pos.x * 0.5f + 0.5f) * settings.width, 
					(pos.y * 0.5f + 0.5f) * settings.height,
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

			{// backface culling
				float w = cross(v01, v02);
				if(w < 0.0) continue;
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
				if(numProcessedSamples > 5'000) break;

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
					}else{
						// WHATEVER
						color = sh_triangleIndex * 123456;
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
	cudaSurfaceObject_t gl_colorbuffer_main,
	cudaSurfaceObject_t gl_colorbuffer_vr_left,
	cudaSurfaceObject_t gl_colorbuffer_vr_right,
	uint32_t numTriangles,
	float3* positions,
	float2* uvs,
	uint32_t* colors,
	uint32_t* textureData
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
	uint64_t* fb_vr_left = allocator->alloc<uint64_t*>(int(uniforms.vr_left_width) * int(uniforms.vr_left_height) * sizeof(uint64_t));
	uint64_t* fb_vr_right = allocator->alloc<uint64_t*>(int(uniforms.vr_right_width) * int(uniforms.vr_right_height) * sizeof(uint64_t));

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
	});

	if(uniforms.vrEnabled){
		processRange(0, uniforms.vr_left_width * uniforms.vr_left_height, [&](int pixelIndex){
			fb_vr_left[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
		});

		processRange(0, uniforms.vr_right_width * uniforms.vr_right_height, [&](int pixelIndex){
			fb_vr_right[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
		});
	}
	
	grid.sync();

	{ // generate and draw a ground plane
		int cells = 50;
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
			float height = -0.5f;
			
			triangles->positions[offset + 0] = {s * u0 - s * 0.5f, s * v0 - s * 0.5f, height};
			triangles->positions[offset + 1] = {s * u1 - s * 0.5f, s * v0 - s * 0.5f, height};
			triangles->positions[offset + 2] = {s * u1 - s * 0.5f, s * v1 - s * 0.5f, height};
			triangles->positions[offset + 3] = {s * u0 - s * 0.5f, s * v0 - s * 0.5f, height};
			triangles->positions[offset + 4] = {s * u1 - s * 0.5f, s * v1 - s * 0.5f, height};
			triangles->positions[offset + 5] = {s * u0 - s * 0.5f, s * v1 - s * 0.5f, height};

			triangles->uvs[offset + 0] = {u0, v0};
			triangles->uvs[offset + 1] = {u1, v0};
			triangles->uvs[offset + 2] = {u1, v1};
			triangles->uvs[offset + 3] = {u0, v0};
			triangles->uvs[offset + 4] = {u1, v1};
			triangles->uvs[offset + 5] = {u0, v1};
		});

		Texture texture;
		texture.width = 512;
		texture.height = 512;
		texture.data = allocator->alloc<uint32_t*>(4 * texture.width * texture.height);

		grid.sync();

		processRange(0, texture.width * texture.height, [&](int index){
			
			int x = index % texture.width;
			int y = index / texture.width;

			uint32_t color;
			uint8_t* rgba = (uint8_t*)&color;

			if((x % 16) == 0 || (y % 16) == 0){
				color = 0x00000000;
			}else{
				color = 0x00aaaaaa;
			}

			// rgba[0] = 255.0f * float(x) / float(texture.width);
			// rgba[1] = 255.0f * float(y) / float(texture.height);
			// rgba[2] = 0;
			// rgba[3] = 255;

			texture.data[index] = color;

		});

		grid.sync();

		
		RasterizationSettings settings;
		settings.texture = nullptr;
		settings.colorMode = COLORMODE_TRIANGLE_ID;
		settings.world = mat4::identity();
		settings.view = uniforms.view;
		settings.proj = uniforms.proj;
		settings.width = uniforms.width;
		settings.height = uniforms.height;
		settings.texture = &texture;

		// when drawing time, due to normalization, everything needs to be colored by time
		// lets draw the ground with non-normalized time as well for consistency
		if(uniforms.colorMode == COLORMODE_TIME){
			settings.colorMode = COLORMODE_TIME_NORMALIZED;
		}else if(uniforms.colorMode == COLORMODE_TIME_NORMALIZED){
			settings.colorMode = COLORMODE_TIME_NORMALIZED;
		}

		settings.colorMode = COLORMODE_TEXTURE;

		// rasterizeTriangles(triangles, framebuffer, settings);

		if(uniforms.vrEnabled){
			settings.view = uniforms.vr_left_view;
			settings.proj = uniforms.vr_left_proj;
			settings.width = uniforms.vr_left_width;
			settings.height = uniforms.vr_left_height;
			rasterizeTriangles(triangles, fb_vr_left, settings);

			grid.sync();

			settings.view = uniforms.vr_right_view;
			settings.proj = uniforms.vr_right_proj;
			settings.width = uniforms.vr_right_width;
			settings.height = uniforms.vr_right_height;
			rasterizeTriangles(triangles, fb_vr_right, settings);
		}else{
			settings.view = uniforms.view;
			settings.proj = uniforms.proj;
			settings.width = uniforms.width;
			settings.height = uniforms.height;

			rasterizeTriangles(triangles, framebuffer, settings);
		}
	}

	grid.sync();

	// if(false)
	{ // draw the triangle mesh that was passed to this kernel
		Triangles* triangles = allocator->alloc<Triangles*>(sizeof(Triangles));
		triangles->numTriangles = numTriangles;

		triangles->positions = positions;
		triangles->uvs = uvs;
		triangles->colors = colors;

		Texture texture;
		texture.width  = 1024;
		texture.height = 1024;
		texture.data   = textureData;

		RasterizationSettings settings;
		settings.texture = &texture;
		settings.colorMode = uniforms.colorMode;
		settings.world = uniforms.world;

		// rasterizeTriangles(triangles, framebuffer, settings);
		{
			float s = 0.8f;
			mat4 rot = mat4::rotate(0.5f * PI, {1.0f, 0.0f, 0.0f}).transpose();
			mat4 translate = mat4::identity();
			mat4 scale = mat4::scale(s, s, s);
			mat4 wiggle = mat4::rotate(cos(5.0f * uniforms.time) * 0.1f, {0.0f, 1.0f, 0.0f}).transpose();
			mat4 wiggle_yaw = mat4::rotate(cos(5.0f * uniforms.time) * 0.1f, {0.0f, 0.0f, 1.0f}).transpose();
			
			settings.world = translate * wiggle * wiggle_yaw * rot * scale;

			
			if(uniforms.vrEnabled){

				if(uniforms.vr_left_controller_active){
					settings.world = rot * uniforms.vr_left_controller_pose.transpose() * mat4::scale(0.1f, 0.1f, 0.1f);

					settings.view = uniforms.vr_left_view;
					settings.proj = uniforms.vr_left_proj;
					settings.width = uniforms.vr_left_width;
					settings.height = uniforms.vr_left_height;
					rasterizeTriangles(triangles, fb_vr_left, settings);

					grid.sync();

					settings.view = uniforms.vr_right_view;
					settings.proj = uniforms.vr_right_proj;
					settings.width = uniforms.vr_right_width;
					settings.height = uniforms.vr_right_height;
					rasterizeTriangles(triangles, fb_vr_right, settings);
				}

				if(uniforms.vr_right_controller_active){
					settings.world = rot * uniforms.vr_right_controller_pose.transpose() * mat4::scale(0.1f, 0.1f, 0.1f);

					settings.view = uniforms.vr_left_view;
					settings.proj = uniforms.vr_left_proj;
					settings.width = uniforms.vr_left_width;
					settings.height = uniforms.vr_left_height;
					rasterizeTriangles(triangles, fb_vr_left, settings);

					grid.sync();

					settings.view = uniforms.vr_right_view;
					settings.proj = uniforms.vr_right_proj;
					settings.width = uniforms.vr_right_width;
					settings.height = uniforms.vr_right_height;
					rasterizeTriangles(triangles, fb_vr_right, settings);
				}
			}else{
				settings.view = uniforms.view;
				settings.proj = uniforms.proj;
				settings.width = uniforms.width;
				settings.height = uniforms.height;

				rasterizeTriangles(triangles, framebuffer, settings);
			}

			grid.sync();
		}
	}

	// grid.sync();

	// if(uniforms.vrEnabled)
	// {
	// 	uniforms.vr_left_view
	// }

	grid.sync();

	uint32_t& maxNanos = *allocator->alloc<uint32_t*>(4);

	// if colored by normalized time, we compute the max time for normalization
	if(uniforms.colorMode == COLORMODE_TIME_NORMALIZED){
		if(grid.thread_rank() == 0){
			maxNanos = 0;
		}
		grid.sync();

		processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){

			int x = pixelIndex % int(uniforms.width);
			int y = pixelIndex / int(uniforms.width);

			uint64_t encoded = framebuffer[pixelIndex];
			uint32_t color = encoded & 0xffffffffull;

			if(color != BACKGROUND_COLOR){
				atomicMax(&maxNanos, color);
			}
		});

		grid.sync();
	}

	// transfer framebuffer to opengl texture
	if(uniforms.vrEnabled){
		
		// left
		processRange(0, uniforms.vr_left_width * uniforms.vr_left_height, [&](int pixelIndex){
			int x = pixelIndex % int(uniforms.vr_left_width);
			int y = pixelIndex / int(uniforms.vr_left_width);

			uint64_t encoded = fb_vr_left[pixelIndex];
			uint32_t color = encoded & 0xffffffffull;

			surf2Dwrite(color, gl_colorbuffer_vr_left, x * 4, y);
		});

		// right
		processRange(0, uniforms.vr_right_width * uniforms.vr_right_height, [&](int pixelIndex){
			int x = pixelIndex % int(uniforms.vr_right_width);
			int y = pixelIndex / int(uniforms.vr_right_width);

			uint64_t encoded = fb_vr_right[pixelIndex];
			uint32_t color = encoded & 0xffffffffull;

			surf2Dwrite(color, gl_colorbuffer_vr_right, x * 4, y);
		});

		// blit vr displays to main window
		processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){

			int x = pixelIndex % int(uniforms.width);
			int y = pixelIndex / int(uniforms.width);

			float u = fmodf(2.0 * float(x) / uniforms.width, 1.0f);
			float v = float(y) / uniforms.height;

			uint32_t color = 0x000000ff;
			if(x < uniforms.width / 2.0){
				int vr_x = u * uniforms.vr_left_width;
				int vr_y = v * uniforms.vr_left_height;
				int vr_pixelIndex = vr_x + vr_y * uniforms.vr_left_width;

				uint64_t encoded = fb_vr_left[vr_pixelIndex];
				color = encoded & 0xffffffffull;
			}else{
				int vr_x = u * uniforms.vr_right_width;
				int vr_y = v * uniforms.vr_right_height;
				int vr_pixelIndex = vr_x + vr_y * uniforms.vr_right_width;

				uint64_t encoded = fb_vr_right[vr_pixelIndex];
				color = encoded & 0xffffffffull;
			}

			if(uniforms.colorMode == COLORMODE_TIME_NORMALIZED)
			if(color != BACKGROUND_COLOR)
			{
				color = color / (maxNanos / 255);
			}

			surf2Dwrite(color, gl_colorbuffer_main, x * 4, y);
		});

	}else{
		// blit custom cuda framebuffer to opengl texture
		processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){

			int x = pixelIndex % int(uniforms.width);
			int y = pixelIndex / int(uniforms.width);

			uint64_t encoded = framebuffer[pixelIndex];
			uint32_t color = encoded & 0xffffffffull;

			if(uniforms.colorMode == COLORMODE_TIME_NORMALIZED)
			if(color != BACKGROUND_COLOR)
			{
				color = color / (maxNanos / 255);
			}

			surf2Dwrite(color, gl_colorbuffer_main, x * 4, y);
		});
	}


}
