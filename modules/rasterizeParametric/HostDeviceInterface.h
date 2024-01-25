
#pragma once

#include "builtin_types.h"

struct mat4{
	float4 rows[4];

	static mat4 identity(){
		mat4 id;

		id.rows[0] = {1.0f, 0.0f, 0.0f, 0.0f};
		id.rows[1] = {0.0f, 1.0f, 0.0f, 0.0f};
		id.rows[2] = {0.0f, 0.0f, 1.0f, 0.0f};
		id.rows[3] = {0.0f, 0.0f, 0.0f, 1.0f};

		return id;
	}

	static mat4 rotate(float angle, float3 axis){
		// see https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

		float cosa = cos(-angle);
		float sina = sin(-angle);

		float ux = axis.x;
		float uy = axis.y;
		float uz = axis.z;

		mat4 rot;
		rot.rows[0].x = cosa + ux * ux * (1.0f - cosa);
		rot.rows[0].y = ux * uy * (1.0f - cosa) - uz * sina;
		rot.rows[0].z = ux * uz * (1.0f - cosa) + uy * sina;
		rot.rows[0].w = 0.0f;

		rot.rows[1].x = uy * ux * (1.0f - cosa) + uz * sina;
		rot.rows[1].y = cosa + uy * uy * (1.0f - cosa);
		rot.rows[1].z = uy * uz * (1.0f - cosa) - ux * sina;
		rot.rows[1].w = 0.0f;

		rot.rows[2].x = uz * ux * (1.0f - cosa) - uy * sina;
		rot.rows[2].y = uz * uy * (1.0f - cosa) + ux * sina;
		rot.rows[2].z = cosa + uz * uz * (1.0f - cosa);
		rot.rows[2].w = 0.0f;

		rot.rows[3].x = 0.0f;
		rot.rows[3].y = 0.0f;
		rot.rows[3].z = 0.0f;
		rot.rows[3].w = 1.0f;

		return rot;
	}

	static mat4 translate(float x, float y, float z){

		mat4 trans = mat4::identity();

		trans.rows[0].w = x;
		trans.rows[1].w = y;
		trans.rows[2].w = z;

		return trans;
	}

	static mat4 scale(float sx, float sy, float sz){

		mat4 scaled = mat4::identity();

		scaled.rows[0].x = sx;
		scaled.rows[1].y = sy;
		scaled.rows[2].z = sz;

		return scaled;
	}

	mat4 transpose(){
		mat4 result;

		result.rows[0] = {rows[0].x, rows[1].x, rows[2].x, rows[3].x};
		result.rows[1] = {rows[0].y, rows[1].y, rows[2].y, rows[3].y};
		result.rows[2] = {rows[0].z, rows[1].z, rows[2].z, rows[3].z};
		result.rows[3] = {rows[0].w, rows[1].w, rows[2].w, rows[3].w};

		return result;
	}
};

int COLORMODE_TEXTURE          = 0;
int COLORMODE_UV               = 1;
int COLORMODE_TRIANGLE_ID      = 2;
int COLORMODE_TIME             = 3;
int COLORMODE_TIME_NORMALIZED  = 4;

int SAMPLEMODE_NEAREST     = 0;
int SAMPLEMODE_LINEAR      = 1;

int METHOD_32X32           = 0;
int METHOD_RUNNIN_THRU     = 1;
int METHOD_SAMPLEPERF_TEST = 2;

constexpr int MODEL_PLANE                  = 0;
constexpr int MODEL_FUNKY_PLANE            = 1;
constexpr int MODEL_EXTRA_FUNKY_PLANE      = 2;
constexpr int MODEL_SPHERE                 = 3;
constexpr int MODEL_GLYPH                  = 4;
constexpr int JOHIS_HEART                  = 5;
constexpr int SPHEREHOG                    = 6;

constexpr int PATCHES_CAPACITY = 100'000;
constexpr int PATCH_RESOLUTION = 32; // n x n points per node
constexpr int POINTS_PER_PATCH = PATCH_RESOLUTION * PATCH_RESOLUTION;
constexpr int POINTS_CAPACITY = PATCHES_CAPACITY * POINTS_PER_PATCH;

struct Uniforms{
	float width;
	float height;
	float time;
	
	mat4 world;
	mat4 view;
	mat4 proj;
	mat4 transform;

	mat4 locked_view;
	mat4 locked_transform;

	int method;
	int model;
	bool isPaused;
	bool enableRefinement;
	bool lockFrustum;
	int  cullingMode;
	int  showHeatmap;
	int frameCount;
};

struct Stats{
	uint32_t frameID                    = 0;
	uint32_t dbg                        = 0;
	float time_0                        = 0.0;
	float time_1                        = 0.0;
	float time_2                        = 0.0;
	float time_3                        = 0.0;
	int numPatches[30]                  = {0};
};

struct Point{
	float3 pos;
	float3 normal;
	union{
		uint32_t color;
		uint8_t rgba[4];
	};
};

struct Patch{
	Patch* parent;
	Patch* children[4];
	uint32_t numPoints;
	float2 st_min;
	float2 st_max;
	float3 min;
	float3 max;
	int modelID;
	Point points[POINTS_PER_PATCH];

	bool isLeaf(){
		if(children[0] != nullptr) return false;
		if(children[1] != nullptr) return false;
		if(children[2] != nullptr) return false;
		if(children[3] != nullptr) return false;

		return true;
	}
};

struct PatchPool{
	uint32_t capacity;
	uint32_t offset;
	Patch* pointers[PATCHES_CAPACITY];
};

