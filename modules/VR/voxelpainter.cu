#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.cuh"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"


#include "triangles.cuh"
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

	uint32_t* voxelGrid = allocator->alloc<uint32_t*>(numCells * sizeof(uint32_t));

	// clear/initialize voxel grid, if necessary. Note that this is done every frame.
	// processRange(0, numCells, [&](int voxelIndex){
	// 	int x = voxelIndex % gridSize;
	// 	int y = voxelIndex % (gridSize * gridSize) / gridSize;
	// 	int z = voxelIndex / (gridSize * gridSize);

	// 	float fx = 2.0f * float(x) / fGridSize - 1.0f;
	// 	float fy = 2.0f * float(y) / fGridSize - 1.0f;
	// 	float fz = 2.0f * float(z) / fGridSize - 1.0f;

	// 	// clear and make sphere and ground plane
	// 	if(fx * fx + fy * fy + fz * fz < 0.1f){
	// 		voxelGrid[voxelIndex] = 123;
	// 	}else if(x > 10 && x < gridSize - 10 && z < 4){
	// 		voxelGrid[voxelIndex] = 123;

	// 	}else{
	// 		voxelGrid[voxelIndex] = 0;
	// 	}

	// 	// clear everything
	// 	// voxelGrid[voxelIndex] = 0;
	// });

	{ // clear framebuffer
		uint64_t clearValue = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
		clearBuffer_u64(framebuffer, clearValue, uniforms.width * uniforms.height);

		if(uniforms.vrEnabled){
			clearBuffer_u64(fb_vr_left, clearValue, uniforms.vr_left_width * uniforms.vr_left_height);
			clearBuffer_u64(fb_vr_left, clearValue, uniforms.vr_right_width * uniforms.vr_right_height);
		}
	}

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
		Triangles* triangles = createGroundPlane();
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

	// VOXEL PAINTING / CONTROLLER INPUT
	if(grid.thread_rank() == 0){

		// see openvr.h: EVRButtonId and ButtonMaskFromId
		uint64_t triggerMask = 1ull << 33ull;
		bool leftTriggerButtonDown = (uniforms.vr_left_controller_state.buttonPressedMask & triggerMask) != 0;
		bool rightTriggerButtonDown = (uniforms.vr_right_controller_state.buttonPressedMask & triggerMask) != 0;
		bool isTriggerDown = leftTriggerButtonDown || rightTriggerButtonDown;

		// printf("%llu \n", leftTriggerButtonDown);
		
		if(rightTriggerButtonDown){
			float brushRadius = 3.0;
			int iBrushRadius = ceil(brushRadius);
			float brushRadius2 = brushRadius * brushRadius;
			
			mat4 rot = mat4::rotate(0.5f * PI, {1.0f, 0.0f, 0.0f}).transpose();
			float4 pos = rot * uniforms.vr_right_controller_pose.transpose() * float4{0.0f, 0.0f, 0.0f, 1.0f};
			float3 boxSize = gridMax - gridMin;

			float fx = gridSize * (pos.x - gridMin.x) / boxSize.x;
			float fy = gridSize * (pos.y - gridMin.y) / boxSize.y;
			float fz = gridSize * (pos.z - gridMin.z) / boxSize.z;

			for(int ox = -iBrushRadius; ox <= brushRadius; ox++)
			for(int oy = -iBrushRadius; oy <= brushRadius; oy++)
			for(int oz = -iBrushRadius; oz <= brushRadius; oz++)
			{

				int ix = fx + float(ox);
				int iy = fy + float(oy);
				int iz = fz + float(oz);

				if(ix < 0 || ix >= gridSize) continue;
				if(iy < 0 || iy >= gridSize) continue;
				if(iz < 0 || iz >= gridSize) continue;

				int voxelIndex = ix + iy * gridSize + iz * gridSize * gridSize;

				float vcx = float(ix) + 0.5f;
				float vcy = float(iy) + 0.5f;
				float vcz = float(iz) + 0.5f;
				float dx = vcx - fx;
				float dy = vcy - fy;
				float dz = vcz - fz;
				float dd = dx * dx + dy * dy + dz * dz;

				if(dd < brushRadius2){
					voxelGrid[voxelIndex] = 123;
				}
			}
		}

		if(leftTriggerButtonDown){
			float brushRadius = 5.0;
			int iBrushRadius = ceil(brushRadius);
			float brushRadius2 = brushRadius * brushRadius;
			
			mat4 rot = mat4::rotate(0.5f * PI, {1.0f, 0.0f, 0.0f}).transpose();
			float4 pos = rot * uniforms.vr_left_controller_pose.transpose() * float4{0.0f, 0.0f, 0.0f, 1.0f};
			float3 boxSize = gridMax - gridMin;

			float fx = gridSize * (pos.x - gridMin.x) / boxSize.x;
			float fy = gridSize * (pos.y - gridMin.y) / boxSize.y;
			float fz = gridSize * (pos.z - gridMin.z) / boxSize.z;

			for(int ox = -iBrushRadius; ox <= brushRadius; ox++)
			for(int oy = -iBrushRadius; oy <= brushRadius; oy++)
			for(int oz = -iBrushRadius; oz <= brushRadius; oz++)
			{

				int ix = fx + float(ox);
				int iy = fy + float(oy);
				int iz = fz + float(oz);

				if(ix < 0 || ix >= gridSize) continue;
				if(iy < 0 || iy >= gridSize) continue;
				if(iz < 0 || iz >= gridSize) continue;

				int voxelIndex = ix + iy * gridSize + iz * gridSize * gridSize;

				float vcx = float(ix) + 0.5f;
				float vcy = float(iy) + 0.5f;
				float vcz = float(iz) + 0.5f;
				float dx = vcx - fx;
				float dy = vcy - fy;
				float dz = vcz - fz;
				float dd = dx * dx + dy * dy + dz * dz;

				if(dd < brushRadius2){
					voxelGrid[voxelIndex] = 0;
				}
			}
		}
	}

	grid.sync();

	{ // DRAW VOXEL GRID
		
		Triangles* triangles = marchingCubes(gridMin, gridMax, gridSize, voxelGrid);

		grid.sync();
		
		rs_main.texture = nullptr;
		rs_main.colorMode = COLORMODE_VERTEXCOLOR;

		if(uniforms.vrEnabled){
			rs_left.texture   = nullptr;
			rs_left.colorMode  = COLORMODE_VERTEXCOLOR;
			rs_right.texture   = nullptr;
			rs_right.colorMode = COLORMODE_VERTEXCOLOR;
			rasterizeTriangles(triangles, fb_vr_left, rs_left);
			rasterizeTriangles(triangles, fb_vr_right, rs_right);
		}else{
			rasterizeTriangles(triangles, framebuffer, rs_main);
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
				rs_right.world = rot * uniforms.vr_right_controller_pose.transpose() 
					* mat4::scale(sController, sController, sController);
				if(uniforms.vr_left_controller_active){
					rasterizeTriangles(triangles, fb_vr_left, rs_left);
					rasterizeTriangles(triangles, fb_vr_right, rs_right);
				}

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
		// drawSkybox(
		// 	uniforms.vr_left_proj, uniforms.vr_left_view, 
		// 	uniforms.vr_left_proj_inv, uniforms.vr_left_view_inv, 
		// 	framebuffer, 
		// 	uniforms.vr_left_width, uniforms.vr_left_height, 
		// 	skybox
		// );

		// drawSkybox(
		// 	uniforms.vr_right_proj, uniforms.vr_right_view, 
		// 	uniforms.vr_right_proj_inv, uniforms.vr_right_view_inv, 
		// 	framebuffer, 
		// 	uniforms.vr_right_width, uniforms.vr_right_height, 
		// 	skybox
		// );
		
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