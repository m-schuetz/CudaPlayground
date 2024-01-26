#pragma once

#include "globals.cuh"
#include "more_math.cuh"

void rasterizePoint(
	uint64_t* framebuffer, float3 position, uint32_t color,
	mat4 transform, float width, float height
){

	float4 ndc = transform * float4{position.x, position.y, position.z, 1.0f};
	ndc.x = ndc.x / ndc.w;
	ndc.y = ndc.y / ndc.w;
	
	float2 screen = {
		(0.5f * ndc.x + 0.5f) * width,
		(0.5f * ndc.y + 0.5f) * height,
	};

	int px = screen.x;
	int py = screen.y;

	if(px < 0 || px >= width) return;
	if(py < 0 || py >= height) return;

	int pixelID = px + width * py;

	float depth = ndc.w;
	uint64_t udepth = *((uint32_t*)&depth);
	uint64_t pixel = (udepth << 32) | color;

	atomicMin(&framebuffer[pixelID], pixel);
	
}

void rasterizeSprite(
	uint64_t* framebuffer, float3 position, uint32_t color, int size,
	mat4 transform, float width, float height
){

	float4 ndc = transform * float4{position.x, position.y, position.z, 1.0f};
	ndc.x = ndc.x / ndc.w;
	ndc.y = ndc.y / ndc.w;
	
	float2 screen = {
		(0.5f * ndc.x + 0.5f) * width,
		(0.5f * ndc.y + 0.5f) * height,
	};

	for(float ix = 0; ix < size; ix += 1.0f)
	for(float iy = 0; iy < size; iy += 1.0f)
	{
		int px = screen.x + ix - size / 2;
		int py = screen.y + iy - size / 2;

		if(px < 0 || px >= width) continue;
		if(py < 0 || py >= height) continue;

		// if(ix == 0 && iy == 0)
		// printf("%6.1f, %6.1f\n", screen.x, screen.y);

		int pixelID = px + width * py;

		float depth = ndc.w;
		uint64_t udepth = *((uint32_t*)&depth);
		uint64_t pixel = (udepth << 32) | color;

		atomicMin(&framebuffer[pixelID], pixel);
	}
}