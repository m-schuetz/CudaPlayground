#pragma once

// #include "globals.cuh"
#include "utils.cuh"

Uniforms uniforms;
Allocator* allocator;
uint64_t nanotime_start;

constexpr int LINES_CAPACITY = 1'000'000;
struct {
	uint32_t count;
	float3 positions[LINES_CAPACITY];
	uint32_t colors[LINES_CAPACITY];
} g_lines;