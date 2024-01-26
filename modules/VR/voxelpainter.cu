#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.cuh"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

#include "triangles.cuh"
#include "lines.cuh"
#include "points.cuh"
#include "texture.cuh"
#include "voxels.cuh"
#include "globals.cuh"
#include "skybox.cuh"

namespace cg = cooperative_groups;


constexpr bool EDL_ENABLED = false;
constexpr uint32_t gridSize = 128;
constexpr float fGridSize = gridSize;
constexpr uint32_t numCells = gridSize * gridSize * gridSize;
constexpr float3 gridMin = { -1.0f, -1.0f, 0.0f};
constexpr float3 gridMax = { 1.0f, 1.0f, 2.0f};
constexpr float PI = 3.1415;
constexpr uint32_t BACKGROUND_COLOR = 0x00332211ull;

// struct Particle{
// 	float3 pos;
// 	uint32_t color;
// 	float3 velocity;
// };

constexpr int MAX_PARTICLES = 10'000'000;
// Particle particles[MAX_PARTICLES];

// struct{
// 	float3   position[MAX_PARTICLES];
// 	uint32_t color[MAX_PARTICLES];
// 	float    age[MAX_PARTICLES];
// 	float3   velocity[MAX_PARTICLES];
// } g_particles;


uint32_t SPECTRAL[11] = {
	0x42019e,
	0x4f3ed5,
	0x436df4,
	0x61aefd,
	0x8be0fe,
	0xbfffff,
	0x98f5e6,
	0xa4ddab,
	0xa5c266,
	0xbd8832,
	0xa24f5e,
};

extern "C" __global__
void kernel(
	const Uniforms _uniforms,
	uint32_t* buffer,
	cudaSurfaceObject_t gl_colorbuffer_main,
	cudaSurfaceObject_t gl_colorbuffer_vr_left,
	cudaSurfaceObject_t gl_colorbuffer_vr_right,
	uint64_t* framebuffer,
	uint64_t* fb_vr_left,
	uint64_t* fb_vr_right,
	uint32_t numTriangles,
	float3* positions,
	float2* uvs,
	uint32_t* colors,
	uint32_t* textureData,
	const Skybox skybox
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime_start));

	uniforms = _uniforms;

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	curandStateXORWOW_t thread_random_state;
	curand_init(grid.thread_rank(), 0, 0, &thread_random_state);

	{ // clear framebuffer
		uint64_t clearValue = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
		clearBuffer_u64(framebuffer, clearValue, uniforms.width * uniforms.height);

		if(uniforms.vrEnabled){
			clearBuffer_u64(fb_vr_left, clearValue, uniforms.vr_left_width * uniforms.vr_left_height);
			clearBuffer_u64(fb_vr_right, clearValue, uniforms.vr_right_width * uniforms.vr_right_height);
		}
	}

	uint64_t* fb_points = allocator->alloc<uint64_t*>(uniforms.width * uniforms.height * sizeof(uint64_t));
	{
		uint64_t clearValue = (uint64_t(Infinity) << 32ull) | uint64_t(0);
		clearBuffer_u64(fb_points, clearValue, uniforms.width * uniforms.height);
	}

	struct{
		float3*   position;
		uint32_t* color;
		float*    age;
		float*    lifetime;
		float3*   velocity;
	} particles;

	particles.position = allocator->alloc<float3*  >(MAX_PARTICLES * sizeof(float3));
	particles.color    = allocator->alloc<uint32_t*>(MAX_PARTICLES * sizeof(uint32_t));
	particles.age      = allocator->alloc<float*   >(MAX_PARTICLES * sizeof(float));
	particles.lifetime = allocator->alloc<float*   >(MAX_PARTICLES * sizeof(float));
	particles.velocity = allocator->alloc<float3*  >(MAX_PARTICLES * sizeof(float3));



	// clear particles
	processRange(MAX_PARTICLES, [&](int index){
		uint32_t X = curand(&thread_random_state) >> 16;
		uint32_t Y = curand(&thread_random_state) >> 16;
		uint32_t Z = curand(&thread_random_state) >> 16;
		uint32_t upper = 1 << 16;

		uint32_t color = curand(&thread_random_state);

		color = SPECTRAL[color % 11];

		float x = 10.0f * (float(X) / float(upper) - 0.5f);
		float y =  1.0f * (float(Y) / float(upper) - 0.5f) - 0.8f;
		float z = 10.0f * (float(Z) / float(upper) - 0.5f);

		uint32_t lifetime_ms = curand(&thread_random_state) % 2000 + 1000;

		particles.position[index] = float3{x, y, z};
		particles.color[index] = color;
		particles.age[index] = 0.0f;
		particles.lifetime[index] = float(lifetime_ms) / 1000.0f;

		// g_particles.position[index] = float3{0.0f, 0.0f, 0.0f};
		// g_particles.color[index] = 0;
		// g_particles.age[index] = -1.0f;
		particles.velocity[index] = float3{0.0f, 0.0f, 0.0f};
	});


	g_lines.count = 0;

	RasterizationSettings rs_main;
	RasterizationSettings rs_left;
	RasterizationSettings rs_right;

	rs_main.texture   = nullptr;
	rs_main.colorMode = COLORMODE_VERTEXCOLOR;
	rs_main.world     = mat4::identity();
	rs_main.view      = uniforms.view;
	rs_main.proj      = uniforms.proj;
	rs_main.width     = uniforms.width;
	rs_main.height    = uniforms.height;

	if(uniforms.vrEnabled){
		rs_left = rs_main;
		rs_right = rs_main;

		rs_left.view    = uniforms.vr_left_view;
		rs_left.proj    = uniforms.vr_left_proj;
		rs_left.width   = uniforms.vr_left_width;
		rs_left.height  = uniforms.vr_left_height;

		rs_right.view   = uniforms.vr_right_view;
		rs_right.proj   = uniforms.vr_right_proj;
		rs_right.width  = uniforms.vr_right_width;
		rs_right.height = uniforms.vr_right_height;
	}
	
	grid.sync();

	{ // generate and draw a ground plane
		Triangles* triangles = createGroundPlane(50);
		Texture texture = createGridTexture();

		grid.sync();
		
		rs_main.colorMode = COLORMODE_TEXTURE;
		rs_main.texture = &texture;

		if(uniforms.vrEnabled){
			rs_left.colorMode  = COLORMODE_TEXTURE;
			rs_left.texture    = &texture;
			rs_right.colorMode = COLORMODE_TEXTURE;
			rs_right.texture   = &texture;

			rasterizeTriangles(triangles, fb_vr_left, rs_left);
			grid.sync();
			rasterizeTriangles(triangles, fb_vr_right, rs_right);
		}else{
			rasterizeTriangles(triangles, framebuffer, rs_main);
		}
	}

	grid.sync();

	{
		Triangles* sphere = createSphere(24);


		for(int i = 0; i <= 10; i++)
		for(int j = -10; j <= 10; j += 2)
		// for(int j : {0, 1, 2})
		// for(int i : {1})
		// for(int j : {-5})
		{
			// uint32_t X = curand(&thread_random_state) >> 16;
			// uint32_t Y = curand(&thread_random_state) >> 16;
			// uint32_t Z = curand(&thread_random_state) >> 16;
			// uint32_t upper = 1 << 16;

			// float x = float(X) / float(upper) - 0.5f;
			// float y = float(Y) / float(upper) - 0.5f;
			// float z = float(Z) / float(upper) - 0.5f;

			float x = i - 5;

			rs_main.texture = nullptr;
			rs_main.colorMode = COLORMODE_UV;
			rs_main.world = mat4::translate(x, -j - 1.0f, -0.3f) * mat4::scale(0.2f, 0.2f, 0.2f);
			

			if(uniforms.vrEnabled){
				RasterizationSettings rs_left;
				rs_left.texture   = nullptr;
				rs_left.colorMode = COLORMODE_UV;
				rs_left.world     = mat4::translate(x, -j - 1.0f, -0.3f) * mat4::scale(0.2f, 0.2f, 0.2f);
				rs_left.view      = uniforms.vr_left_view;
				rs_left.proj      = uniforms.vr_left_proj;
				rs_left.transform = uniforms.vr_left_transform;
				rs_left.width     = uniforms.vr_left_width;
				rs_left.height    = uniforms.vr_left_height;
				
				rasterizeTriangles(sphere, fb_vr_left, rs_left);
				grid.sync();

				RasterizationSettings rs_right;
				rs_right.texture   = nullptr;
				rs_right.colorMode = COLORMODE_VERTEXCOLOR;
				rs_right.world     = mat4::translate(x, -j - 1.0f, -0.3f) * mat4::scale(0.2f, 0.2f, 0.2f);
				rs_right.view      = uniforms.vr_right_view;
				rs_right.proj      = uniforms.vr_right_proj;
				rs_right.transform = uniforms.vr_right_transform;
				rs_right.width     = uniforms.vr_right_width;
				rs_right.height    = uniforms.vr_right_height;
				rasterizeTriangles(sphere, fb_vr_right, rs_right);

				// mat4 transform = uniforms.vr_right_transform;
				// // mat4 transform = uniforms.vr_right_proj * uniforms.vr_right_view;
				// float3 position = {x, -0.3f, -j - 1.0f};
				// rasterizeSprite(fb_vr_right, position, 0x000000ff, 10, transform, uniforms.vr_right_width, uniforms.vr_right_height);

				grid.sync();
			}else{
				// rasterizeTriangles(sphere, framebuffer, rs_main);
				float3 position = {x, -0.3f, -j - 1.0f};
				if(grid.thread_rank() == 0)
				rasterizeSprite(framebuffer, position, 0x000000ff, 10, uniforms.transform, uniforms.width, uniforms.height);
			}

			grid.sync();
		}
	}

	grid.sync();

	{ // DRAW CONTROLLERS
		Triangles* triangles    = allocator->alloc<Triangles*>(sizeof(Triangles));
		triangles->numTriangles = numTriangles;
		triangles->positions    = positions;
		triangles->uvs          = uvs;
		triangles->colors       = colors;

		Texture texture;
		texture.width  = 1024;
		texture.height = 1024;
		texture.data   = textureData;

		rs_main.texture    = &texture;
		rs_left.texture    = &texture;
		rs_right.texture   = &texture;
		
		rs_main.colorMode  = uniforms.colorMode;
		rs_left.colorMode  = uniforms.colorMode;
		rs_right.colorMode = uniforms.colorMode;

		{
			float s = 0.8f;
			mat4 rot = mat4::rotate(0.5f * PI, {1.0f, 0.0f, 0.0f}).transpose();
			mat4 translate = mat4::translate(0.0f, 0.0f, 0.0f);
			mat4 scale = mat4::scale(s, s, s);
			mat4 wiggle = mat4::rotate(cos(5.0f * uniforms.time) * 0.1f, {0.0f, 1.0f, 0.0f}).transpose();
			mat4 wiggle_yaw = mat4::rotate(cos(5.0f * uniforms.time) * 0.1f, {0.0f, 0.0f, 1.0f}).transpose();
			
			if(uniforms.vrEnabled){
				float sController = 0.05f;

				rs_left.world = rot * uniforms.vr_left_controller_pose.transpose() 
					* mat4::scale(sController, sController, sController);
				rs_right.world = rs_left.world;

				if(uniforms.vr_left_controller_active){
					rasterizeTriangles(triangles, fb_vr_left, rs_left);
					rasterizeTriangles(triangles, fb_vr_right, rs_right);
				}

				grid.sync();

				rs_left.world = rot * uniforms.vr_right_controller_pose.transpose() 
					* mat4::scale(sController, sController, sController);
				rs_right.world = rs_left.world;

				if(uniforms.vr_right_controller_active){
					rasterizeTriangles(triangles, fb_vr_left, rs_left);
					rasterizeTriangles(triangles, fb_vr_right, rs_right);
				}
			}else{
				rs_main.world = translate * wiggle * wiggle_yaw * rot * scale;
				rasterizeTriangles(triangles, framebuffer, rs_main);
			}

			grid.sync();
		}
	}

	grid.sync();

	// draw random points
	// for(int i = 0; i < 1; i++){
	// 	uint32_t X = curand(&thread_random_state) >> 16;
	// 	uint32_t Y = curand(&thread_random_state) >> 16;
	// 	uint32_t Z = curand(&thread_random_state) >> 16;
	// 	uint32_t upper = 1 << 16;

	// 	uint32_t color = curand(&thread_random_state);

	// 	color = SPECTRAL[color % 11];

	// 	float x = 10.0f * (float(X) / float(upper) - 0.5f);
	// 	float y =  1.0f * (float(Y) / float(upper) - 0.5f) - 0.8f;
	// 	float z = 10.0f * (float(Z) / float(upper) - 0.5f);

	// 	float3 position = {x, y, z};

	// 	rasterizePoint(fb_points, position, color, 
	// 		uniforms.transform, uniforms.width, uniforms.height);

	// 	// rasterizeSprite(fb_points, position, color, 5,
	// 	// 	uniforms.transform, uniforms.width, uniforms.height);
	// }

	// draw particles
	processRange(1'000'000, [&](int index){

		float age = particles.age[index];

		if(age == -1.0f) return;

		float3 position = particles.position[index];
		uint32_t color = particles.color[index];

		rasterizePoint(fb_points, position, color, 
			uniforms.transform, uniforms.width, uniforms.height);

	});

	grid.sync();

	// transfer points to main framebuffer
	processRange(uniforms.width * uniforms.height, [&](int pixelID){

		// atomicMin(&framebuffer[pixelID], fb_points[pixelID]);
		uint64_t closest = uint64_t(Infinity) << 32;

		int x = pixelID % int(uniforms.width);
		int y = pixelID / int(uniforms.width);

		int radius = 3;
		for(int ox = -radius; ox <= radius; ox++)
		for(int oy = -radius; oy <= radius; oy++)
		// for(int ox : {0})
		// for(int oy : {0})
		{
			int px = x + ox;
			int py = y + oy;

			if(px < 0 || px >= uniforms.width) continue;
			if(py < 0 || py >= uniforms.width) continue;

			int pid = px + int(uniforms.width) * py;

			uint64_t value = fb_points[pid];

			closest = min(closest, value);
		}

		atomicMin(&framebuffer[pixelID], closest);
		// fb_points[pixelID]

	});

	// if(grid.thread_rank() == 0){
	// 	drawBoundingBox({-2.0f, 0.0f, 0.0f}, {0.5f, 1.0f, 1.5f}, 0x000000ff);
	// }

	// grid.sync();

	// rasterizeLines(framebuffer, uniforms.transform, uniforms.width, uniforms.height);

}

extern "C" __global__
void kernel_draw_skybox(
	const Uniforms _uniforms,
	uint32_t* buffer,
	cudaSurfaceObject_t gl_colorbuffer_main,
	cudaSurfaceObject_t gl_colorbuffer_vr_left,
	cudaSurfaceObject_t gl_colorbuffer_vr_right,
	uint64_t* framebuffer,
	uint64_t* fb_vr_left,
	uint64_t* fb_vr_right,
	uint32_t numTriangles,
	float3* positions,
	float2* uvs,
	uint32_t* colors,
	uint32_t* textureData,
	const Skybox skybox
){
	if(uniforms.vrEnabled){

		// TODO
		drawSkybox(
			uniforms.vr_left_proj, uniforms.vr_left_view, 
			uniforms.vr_left_proj_inv, uniforms.vr_left_view_inv, 
			fb_vr_left, 
			uniforms.vr_left_width, uniforms.vr_left_height, 
			skybox
		);

		drawSkybox(
			uniforms.vr_right_proj, uniforms.vr_right_view, 
			uniforms.vr_right_proj_inv, uniforms.vr_right_view_inv, 
			fb_vr_right, 
			uniforms.vr_right_width, uniforms.vr_right_height, 
			skybox
		);
		
		// if(grid.thread_rank() == 0){
		// 	mat4 mat = uniforms.vr_right_proj_inv;
		// 	printf("===========\n");
		// 	printf("%5.1f, %5.1f, %5.1f, %5.1f \n", mat[0].x, mat[0].y, mat[0].z, mat[0].w);
		// 	printf("%5.1f, %5.1f, %5.1f, %5.1f \n", mat[1].x, mat[1].y, mat[1].z, mat[1].w);
		// 	printf("%5.1f, %5.1f, %5.1f, %5.1f \n", mat[2].x, mat[2].y, mat[2].z, mat[2].w);
		// 	printf("%5.1f, %5.1f, %5.1f, %5.1f \n", mat[3].x, mat[3].y, mat[3].z, mat[3].w);
		// }

	}else{
		drawSkybox(
			uniforms.proj, uniforms.view, 
			uniforms.proj_inv, uniforms.view_inv, 
			framebuffer, 
			uniforms.width, uniforms.height, 
			skybox
		);
	}
}


extern "C" __global__
void kernel_toOpenGL(
	const Uniforms _uniforms,
	uint32_t* buffer,
	cudaSurfaceObject_t gl_colorbuffer_main,
	cudaSurfaceObject_t gl_colorbuffer_vr_left,
	cudaSurfaceObject_t gl_colorbuffer_vr_right,
	uint64_t* framebuffer,
	uint64_t* fb_vr_left,
	uint64_t* fb_vr_right,
	uint32_t numTriangles,
	float3* positions,
	float2* uvs,
	uint32_t* colors,
	uint32_t* textureData,
	const Skybox skybox
){
	// TRANSFER TO OPENGL TEXTURE
	if(uniforms.vrEnabled){
		
		// left
		processRange(0, uniforms.vr_left_width * uniforms.vr_left_height, [&](int pixelIndex){
			int x = pixelIndex % int(uniforms.vr_left_width);
			int y = pixelIndex / int(uniforms.vr_left_width);

			uint64_t encoded = fb_vr_left[pixelIndex];
			uint32_t color = encoded & 0xffffffffull;
			uint8_t* rgba = (uint8_t*)&color;
			uint32_t idepth = (encoded >> 32);
			float depth = *((float*)&idepth);

			if(EDL_ENABLED){
				float edlRadius = 2.0f;
				float edlStrength = 0.4f;
				float2 edlSamples[4] = {
					{-1.0f,  0.0f},
					{ 1.0f,  0.0f},
					{ 0.0f,  1.0f},
					{ 0.0f, -1.0f}
				};

				float sum = 0.0f;
				for(int i = 0; i < 4; i++){
					float2 samplePos = {
						x + edlSamples[i].x,
						y + edlSamples[i].y
					};

					int sx = clamp(samplePos.x, 0.0f, uniforms.vr_left_width - 1.0f);
					int sy = clamp(samplePos.y, 0.0f, uniforms.vr_left_height - 1.0f);
					int samplePixelIndex = sx + sy * uniforms.vr_left_width;

					uint64_t sampleEncoded = fb_vr_left[samplePixelIndex];
					uint32_t iSampledepth = (sampleEncoded >> 32);
					float sampleDepth = *((float*)&iSampledepth);

					sum += max(0.0, depth - sampleDepth);
				}

				float shade = exp(-sum * 300.0 * edlStrength);

				rgba[0] = float(rgba[0]) * shade;
				rgba[1] = float(rgba[1]) * shade;
				rgba[2] = float(rgba[2]) * shade;
			}

			// color = 0x000000ff;
			surf2Dwrite(color, gl_colorbuffer_vr_left, x * 4, y);
		});

		// right
		processRange(0, uniforms.vr_right_width * uniforms.vr_right_height, [&](int pixelIndex){
			int x = pixelIndex % int(uniforms.vr_right_width);
			int y = pixelIndex / int(uniforms.vr_right_width);

			uint64_t encoded = fb_vr_right[pixelIndex];
			uint32_t color = encoded & 0xffffffffull;
			uint8_t* rgba = (uint8_t*)&color;
			uint32_t idepth = (encoded >> 32);
			float depth = *((float*)&idepth);

			if(EDL_ENABLED){
				float edlRadius = 2.0f;
				float edlStrength = 0.4f;
				float2 edlSamples[4] = {
					{-1.0f,  0.0f},
					{ 1.0f,  0.0f},
					{ 0.0f,  1.0f},
					{ 0.0f, -1.0f}
				};

				float sum = 0.0f;
				for(int i = 0; i < 4; i++){
					float2 samplePos = {
						x + edlSamples[i].x,
						y + edlSamples[i].y
					};

					int sx = clamp(samplePos.x, 0.0f, uniforms.vr_right_width - 1.0f);
					int sy = clamp(samplePos.y, 0.0f, uniforms.vr_right_height - 1.0f);
					int samplePixelIndex = sx + sy * uniforms.vr_right_width;

					uint64_t sampleEncoded = fb_vr_right[samplePixelIndex];
					uint32_t iSampledepth = (sampleEncoded >> 32);
					float sampleDepth = *((float*)&iSampledepth);

					sum += max(0.0, depth - sampleDepth);
				}

				float shade = exp(-sum * 300.0 * edlStrength);

				rgba[0] = float(rgba[0]) * shade;
				rgba[1] = float(rgba[1]) * shade;
				rgba[2] = float(rgba[2]) * shade;
			}

			// color = 0x000000ff;
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

			// color = 0x000000ff;
			surf2Dwrite(color, gl_colorbuffer_main, x * 4, y);
		});

	}else{
		// blit custom cuda framebuffer to opengl texture
		processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){

			int x = pixelIndex % int(uniforms.width);
			int y = pixelIndex / int(uniforms.width);

			uint64_t encoded = framebuffer[pixelIndex];
			uint32_t color = encoded & 0xffffffffull;

			surf2Dwrite(color, gl_colorbuffer_main, x * 4, y);
		});
	}
}