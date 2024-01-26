#pragma once

#include "globals.cuh"
#include "more_math.cuh"


void drawLine(float3 start, float3 end, uint32_t color){
	int index = atomicAdd(&g_lines.count, 2);
	g_lines.positions[index + 0] = start;
	g_lines.positions[index + 1] = end;
	g_lines.colors[index + 0] = color;
	g_lines.colors[index + 1] = color;
}

void drawBoundingBox(float3 pos, float3 size, uint32_t color){

	float3 min = pos - size / 2.0;
	float3 max = pos + size / 2.0;

	// BOTTOM
	drawLine(float3{min.x, min.y, min.z}, float3{max.x, min.y, min.z}, color);
	drawLine(float3{max.x, min.y, min.z}, float3{max.x, max.y, min.z}, color);
	drawLine(float3{max.x, max.y, min.z}, float3{min.x, max.y, min.z}, color);
	drawLine(float3{min.x, max.y, min.z}, float3{min.x, min.y, min.z}, color);

	// TOP
	drawLine(float3{min.x, min.y, max.z}, float3{max.x, min.y, max.z}, color);
	drawLine(float3{max.x, min.y, max.z}, float3{max.x, max.y, max.z}, color);
	drawLine(float3{max.x, max.y, max.z}, float3{min.x, max.y, max.z}, color);
	drawLine(float3{min.x, max.y, max.z}, float3{min.x, min.y, max.z}, color);

	// BOTTOM TO TOP
	drawLine(float3{max.x, min.y, min.z}, float3{max.x, min.y, max.z}, color);
	drawLine(float3{max.x, max.y, min.z}, float3{max.x, max.y, max.z}, color);
	drawLine(float3{min.x, max.y, min.z}, float3{min.x, max.y, max.z}, color);
	drawLine(float3{min.x, min.y, min.z}, float3{min.x, min.y, max.z}, color);
}

void rasterizeLines(
	uint64_t* framebuffer, mat4 transform, 
	float width, float height
){
	auto grid = cg::this_grid();
	grid.sync();
 
	Frustum frustum = Frustum::fromWorldViewProj(transform);
	
	int numLines = g_lines.count / 2;
	processRange(0, numLines, [&](int lineIndex){

		float3 start = g_lines.positions[2 * lineIndex + 0];
		float3 end = g_lines.positions[2 * lineIndex + 1];

		float3 dir = float3{
			end.x - start.x,
			end.y - start.y,
			end.z - start.z
		};
		dir = normalize(dir);

		if(!frustum.contains({start.x, start.y, start.z})){
			float3 I = frustum.intersectRay({start.x, start.y, start.z}, dir);

			start.x = I.x;
			start.y = I.y;
			start.z = I.z;
		}

		if(!frustum.contains({end.x, end.y, end.z})){
			float3 I = frustum.intersectRay({end.x, end.y, end.z}, dir * -1.0f);

			end.x = I.x;
			end.y = I.y;
			end.z = I.z;
		}

		float4 ndc_start = transform * float4{start.x, start.y, start.z, 1.0f};
		ndc_start.x = ndc_start.x / ndc_start.w;
		ndc_start.y = ndc_start.y / ndc_start.w;
		ndc_start.z = ndc_start.z / ndc_start.w;

		float4 ndc_end = transform * float4{end.x, end.y, end.z, 1.0f};
		ndc_end.x = ndc_end.x / ndc_end.w;
		ndc_end.y = ndc_end.y / ndc_end.w;
		ndc_end.z = ndc_end.z / ndc_end.w;

		float3 screen_start = {
			(ndc_start.x * 0.5f + 0.5f) * width,
			(ndc_start.y * 0.5f + 0.5f) * height,
			1.0f
		};
		float3 screen_end = {
			(ndc_end.x * 0.5f + 0.5f) * width,
			(ndc_end.y * 0.5f + 0.5f) * height,
			1.0f
		};

		float steps = length(screen_end - screen_start);
		// prevent long lines, to be safe
		steps = clamp(steps, 0.0f, 400.0f); 
		float stepSize = 1.0 / steps;

		float start_depth_linear = ndc_start.w;
		float end_depth_linear = ndc_end.w;

		for(float u = 0; u <= 1.0; u += stepSize){
			float ndc_x = (1.0 - u) * ndc_start.x + u * ndc_end.x;
			float ndc_y = (1.0 - u) * ndc_start.y + u * ndc_end.y;
			float depth = (1.0 - u) * start_depth_linear + u * end_depth_linear;

			if(ndc_x < -1.0 || ndc_x > 1.0) continue;
			if(ndc_y < -1.0 || ndc_y > 1.0) continue;

			int x = (ndc_x * 0.5 + 0.5) * width;
			int y = (ndc_y * 0.5 + 0.5) * height;

			x = clamp(x, 0, int(width) - 1);
			y = clamp(y, 0, int(height) - 1);

			// if(RIGHTSIDE_BOXES){
			// 	if(x < width / 2) continue;
			// }

			int pixelID = x + width * y;

			uint64_t idepth = *((uint32_t*)&depth);
			uint32_t color = g_lines.colors[lineIndex];
			uint64_t encoded = (idepth << 32) | color;

			atomicMin(&framebuffer[pixelID], encoded);
		}
	});

	grid.sync();
}