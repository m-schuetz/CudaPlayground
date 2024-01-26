#pragma once

#include "globals.cuh"
#include "more_math.cuh"

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

	// printf("%6.1f, %6.1f\n", screen.x, screen.y);
	// printf("%6.1f, %6.1f, %6.1f\n", position.x, position.y, position.z);
	// printf("%f, %f \n", uniforms.width, uniforms.height);
	// printf("=====================\n");
	// printf("%6.1f, %6.1f, %6.1f, %6.1f \n", transform[0].x, transform[0].y, transform[0].z, transform[0].w);
	// printf("%6.1f, %6.1f, %6.1f, %6.1f \n", transform[1].x, transform[1].y, transform[1].z, transform[1].w);
	// printf("%6.1f, %6.1f, %6.1f, %6.1f \n", transform[2].x, transform[2].y, transform[2].z, transform[2].w);
	// printf("%6.1f, %6.1f, %6.1f, %6.1f \n", transform[3].x, transform[3].y, transform[3].z, transform[3].w);

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