
#pragma once

#include "builtin_types.h"

struct mat4{
	float4 rows[4];
};

int COLORMODE_TEXTURE      = 0;
int COLORMODE_UV           = 1;
int COLORMODE_TRIANGLE_ID  = 2;

int SAMPLEMODE_NEAREST     = 0;
int SAMPLEMODE_LINEAR      = 1;

struct Uniforms{
	float width;
	float height;
	float time;
	mat4 world;
	mat4 view;
	mat4 proj;
	mat4 transform;
	int colorMode;
	int sampleMode;
};
