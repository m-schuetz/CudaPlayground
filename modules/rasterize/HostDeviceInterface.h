
#pragma once

#include "builtin_types.h"

struct mat4{
	float4 rows[4];
};

struct Uniforms{
	float width;
	float height;
	float time;
	mat4 world;
	mat4 view;
	mat4 proj;
	mat4 transform;
};