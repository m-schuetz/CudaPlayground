#pragma once

#include "HostDeviceInterface.h"

void drawSkybox(
	mat4 proj, mat4 view, 
	mat4 proj_inv, mat4 view_inv, 
	uint64_t* framebuffer,
	float width, float height,
	const Skybox& skybox
){ 
	auto projToWorld = [&](float4 pos) -> float4{
		float4 viewspace = proj_inv * pos;

		if(!uniforms.vrEnabled){
			viewspace = viewspace / viewspace.w;
		}

		return view_inv * viewspace;
	};

	float4 origin_projspace = proj * float4{0.0f, 0.0f, 0.0f, 1.0f};
	float4 dir_00_projspace = float4{-1.0f, -1.0f, 0.0f, 1.0f};
	float4 dir_01_projspace = float4{-1.0f,  1.0f, 0.0f, 1.0f};
	float4 dir_10_projspace = float4{ 1.0f, -1.0f, 0.0f, 1.0f};
	float4 dir_11_projspace = float4{ 1.0f,  1.0f, 0.0f, 1.0f};

	float4 origin_worldspace = projToWorld(origin_projspace);
	float4 dir_00_worldspace = projToWorld(dir_00_projspace);
	float4 dir_01_worldspace = projToWorld(dir_01_projspace);
	float4 dir_10_worldspace = projToWorld(dir_10_projspace);
	float4 dir_11_worldspace = projToWorld(dir_11_projspace);

	processRange(width * height, [&](int pixelID){

		// { // early depth test (fast if skybox barely visible, slower if mostly visible)
		// 	uint64_t currentPixelValue = framebuffer[pixelID];
		// 	uint32_t depth = (currentPixelValue >> 32);

		// 	if(depth != Infinity) return;
		// }

		
		int x = pixelID % int(width);
		int y = pixelID / int(width);

		float u = float(x) / width;
		float v = float(y) / height;

		float A_00 = (1.0f - u) * (1.0f - v);
		float A_01 = (1.0f - u) *         v;
		float A_10 =         u  * (1.0f - v);
		float A_11 =         u  *         v;

		float3 dir = make_float3(
			A_00 * dir_00_worldspace + 
			A_01 * dir_01_worldspace + 
			A_10 * dir_10_worldspace + 
			A_11 * dir_11_worldspace - origin_worldspace);
		dir = normalize(dir);
		// float3 origin = make_float3(origin_worldspace);
		float3 origin = {0.0f, 0.0f, 0.0f};

		float3 planes[6] = {
			float3{ 1.0f,  0.0f,  0.0f},
			float3{ 0.0f,  0.0f,  1.0f}, 
			float3{ 0.0f,  1.0f,  0.0f},
			float3{-1.0f,  0.0f,  0.0f},
			float3{ 0.0f,  0.0f, -1.0f},
			float3{ 0.0f, -1.0f,  0.0f},
		};

		// skybox:
		// x: left-right
		// y: bottom-top
		// z: front-back
		int planeIndex = 2 + 3;
		float boxsize = 10.0f;

		float closest_t = Infinity;
		int closest_plane = 0;

		// for(int i = 0; i < 6; i++)
		for(int i : {0, 1, 2, 3, 4, 5})
		{
			float t = rayPlaneIntersection(origin, dir, planes[i], boxsize);

			if(t > 0.0f && t < closest_t){
				closest_t = t;
				closest_plane = i;
			}
		}

		float t = closest_t;
		float3 I = t * dir;
		float2 box_uv;

		if(closest_plane == 0){
			box_uv = {
				0.5f * (I.y / boxsize) + 0.5f, 
				0.5f * (I.z / boxsize) + 0.5f
			};
		}else if(closest_plane == 1){
			box_uv = {
				0.5f * (I.x / boxsize) + 0.5f, 
				0.5f * (I.y / boxsize) + 0.5f
			};
		}else if(closest_plane == 2){
			box_uv = {
				1.0f - (0.5f * (I.x / boxsize) + 0.5f), 
				0.5f * (I.z / boxsize) + 0.5f
			};
		}else if(closest_plane == 3){
			box_uv = {
				1.0f - (0.5f * (I.y / boxsize) + 0.5f), 
				0.5f * (I.z / boxsize) + 0.5f
			};
		}else if(closest_plane == 4){
			box_uv = {
				0.5f * (I.x / boxsize) + 0.5f, 
				1.0f - (0.5f * (I.y / boxsize) + 0.5f)
			};
		}else if(closest_plane == 5){
			box_uv = {
				0.5f * (I.x / boxsize) + 0.5f, 
				0.5f * (I.z / boxsize) + 0.5f
			};
		}

		if(t < 0.0f) return;
		if(box_uv.x > 1.0f) return;
		if(box_uv.x < 0.0f) return;
		if(box_uv.y > 1.0f) return;
		if(box_uv.y < 0.0f) return;

		uint32_t color;
		uint8_t* rgba = (uint8_t*)&color;

		uint8_t* textureData = skybox.textures[closest_plane];
		int tx = clamp(box_uv.x * skybox.width, 0.0f, skybox.width - 1.0f);
		int ty = clamp((1.0f - box_uv.y) * skybox.height, 0.0f, skybox.height - 1.0f);
		int texelIndex = tx + ty * skybox.width;

		rgba[0] = textureData[4 * texelIndex + 0];
		rgba[1] = textureData[4 * texelIndex + 1];
		rgba[2] = textureData[4 * texelIndex + 2];

		// rgba[0] = -dir.x * 200.0f;
		// rgba[1] = -dir.y * 200.0f;
		// rgba[2] = -dir.z * 200.0f;

		// if(x == 1000 && y == 1000){

		// 	float3 abc3 = origin;
		// 	printf("dir: %.1f, %.1f, %.1f \n", abc3.x, abc3.y, abc3.z);

		// 	float4 abc = view_inv * proj_inv * dir_00_projspace;
		// 	printf("dir: %.1f, %.1f, %.1f, %1f \n", abc.x, abc.y, abc.z, abc.w);

		// 	// mat4 mat = proj_inv;
		// 	// mat4 mat = view_inv;
		// 	// printf("==================\n");
		// 	// printf("%6.1f, %6.1f, %6.1f, %6.1f \n", mat[0].x, mat[0].y, mat[0].z, mat[0].w);
		// 	// printf("%6.1f, %6.1f, %6.1f, %6.1f \n", mat[1].x, mat[1].y, mat[1].z, mat[1].w);
		// 	// printf("%6.1f, %6.1f, %6.1f, %6.1f \n", mat[2].x, mat[2].y, mat[2].z, mat[2].w);
		// 	// printf("%6.1f, %6.1f, %6.1f, %6.1f \n", mat[3].x, mat[3].y, mat[3].z, mat[3].w);
		// }

		float depth = 100000000000.0f;
		uint64_t idepth = *((uint32_t*)&depth);
		uint64_t pixel = idepth << 32 | color;
		
		atomicMin(&framebuffer[pixelID], pixel);
	});
}