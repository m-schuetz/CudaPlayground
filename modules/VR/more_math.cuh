#pragma once

#include "helper_math.h"
#include "HostDeviceInterface.h"

float cross(float2 a, float2 b){
	return a.x * b.y - a.y * b.x;
}

float4 operator*(const mat4& a, const float4& b){
	return make_float4(
		dot(a.rows[0], b),
		dot(a.rows[1], b),
		dot(a.rows[2], b),
		dot(a.rows[3], b)
	);
}

mat4 operator*(const mat4& a, const mat4& b){
	
	mat4 result;

	result.rows[0].x = dot(a.rows[0], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
	result.rows[0].y = dot(a.rows[0], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
	result.rows[0].z = dot(a.rows[0], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
	result.rows[0].w = dot(a.rows[0], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

	result.rows[1].x = dot(a.rows[1], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
	result.rows[1].y = dot(a.rows[1], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
	result.rows[1].z = dot(a.rows[1], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
	result.rows[1].w = dot(a.rows[1], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

	result.rows[2].x = dot(a.rows[2], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
	result.rows[2].y = dot(a.rows[2], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
	result.rows[2].z = dot(a.rows[2], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
	result.rows[2].w = dot(a.rows[2], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

	result.rows[3].x = dot(a.rows[3], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
	result.rows[3].y = dot(a.rows[3], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
	result.rows[3].z = dot(a.rows[3], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
	result.rows[3].w = dot(a.rows[3], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

	return result;
}

struct Intersection{
	float3 position;
	float distance;
	
	bool intersects(){
		return distance > 0.0f && distance != Infinity;
	}
};

Intersection rayPlane(float3 origin, float3 direction, float3 planeNormal, float planeDistance){

	float denominator = dot(planeNormal, direction);

	if(denominator == 0){
		Intersection I;
		I.distance = Infinity;

		return I;
	}else{
		float distance = - (dot(origin, planeNormal) + planeDistance ) / denominator;

		Intersection I;
		I.distance = distance;
		I.position = origin + direction * distance;

		return I;
	}

}

// rayPlaneIntersection from three.js
// https://github.com/mrdoob/three.js/blob/13a5874eabfe45fb8459e268e9786a20054bb6a2/src/math/Ray.js#L256
// LICENSE: https://github.com/mrdoob/three.js/blob/13a5874eabfe45fb8459e268e9786a20054bb6a2/LICENSE#L1-L21
//          (MIT)
float rayPlaneIntersection(float3 origin, float3 dir, float3 normal, float d){
	
	float denominator = dot(normal, dir);

	if(denominator == 0){
		return 0.0f;
	}

	float t = -(dot(origin, normal) + d) / denominator;

	return t;
};