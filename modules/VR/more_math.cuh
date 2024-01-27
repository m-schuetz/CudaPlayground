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

int transposeIndex(int index){
	int a = index % 4;
	int b = index / 4;

	return 4 * a + b;
}

struct Plane{
	float3 normal;
	float constant;
};

float t(int index, mat4& transform){

	float* cols = reinterpret_cast<float*>(&transform);

	return cols[index];
}

float distanceToPoint(float3 point, Plane plane){
	// return plane.normal.dot(point) + plane.constant;
	return dot(plane.normal, point) + plane.constant;
}

float distanceToPlane(float3 origin, float3 direction, Plane plane){

	float denominator = dot(plane.normal, direction);

	if(denominator < 0.0f){
		return Infinity;
	}

	if(denominator == 0.0f){

		// line is coplanar, return origin
		if(distanceToPoint(origin, plane) == 0.0f){
			return 0.0f;
		}

		// Null is preferable to undefined since undefined means.... it is undefined
		return Infinity;
	}

	float t = -(dot(origin, plane.normal) + plane.constant) / denominator;

	if(t >= 0.0){
		return t;
	}else{
		return Infinity;
	}
}

Plane createPlane(float x, float y, float z, float w){

	float nLength = length(float3{x, y, z});

	Plane plane;
	plane.normal = float3{x, y, z} / nLength;
	plane.constant = w / nLength;

	return plane;
}

struct Frustum{
	Plane planes[6];

	static Frustum fromWorldViewProj(mat4 worldViewProj){
		float* values = reinterpret_cast<float*>(&worldViewProj);

		float m_0  = values[transposeIndex( 0)];
		float m_1  = values[transposeIndex( 1)];
		float m_2  = values[transposeIndex( 2)];
		float m_3  = values[transposeIndex( 3)];
		float m_4  = values[transposeIndex( 4)];
		float m_5  = values[transposeIndex( 5)];
		float m_6  = values[transposeIndex( 6)];
		float m_7  = values[transposeIndex( 7)];
		float m_8  = values[transposeIndex( 8)];
		float m_9  = values[transposeIndex( 9)];
		float m_10 = values[transposeIndex(10)];
		float m_11 = values[transposeIndex(11)];
		float m_12 = values[transposeIndex(12)];
		float m_13 = values[transposeIndex(13)];
		float m_14 = values[transposeIndex(14)];
		float m_15 = values[transposeIndex(15)];

		Plane planes[6] = {
			createPlane(m_3 - m_0, m_7 - m_4, m_11 -  m_8, m_15 - m_12),
			createPlane(m_3 + m_0, m_7 + m_4, m_11 +  m_8, m_15 + m_12),
			createPlane(m_3 + m_1, m_7 + m_5, m_11 +  m_9, m_15 + m_13),
			createPlane(m_3 - m_1, m_7 - m_5, m_11 -  m_9, m_15 - m_13),
			createPlane(m_3 - m_2, m_7 - m_6, m_11 - m_10, m_15 - m_14),
			createPlane(m_3 + m_2, m_7 + m_6, m_11 + m_10, m_15 + m_14),
		};

		Frustum frustum;

		frustum.planes[0] = createPlane(m_3 - m_0, m_7 - m_4, m_11 -  m_8, m_15 - m_12);
		frustum.planes[1] = createPlane(m_3 + m_0, m_7 + m_4, m_11 +  m_8, m_15 + m_12);
		frustum.planes[2] = createPlane(m_3 + m_1, m_7 + m_5, m_11 +  m_9, m_15 + m_13);
		frustum.planes[3] = createPlane(m_3 - m_1, m_7 - m_5, m_11 -  m_9, m_15 - m_13);
		frustum.planes[4] = createPlane(m_3 - m_2, m_7 - m_6, m_11 - m_10, m_15 - m_14);
		frustum.planes[5] = createPlane(m_3 + m_2, m_7 + m_6, m_11 + m_10, m_15 + m_14);
		
		return frustum;
	}

	float3 intersectRay(float3 origin, float3 direction){

		float closest = Infinity;
		float farthest = -Infinity;

		for(int i = 0; i < 6; i++){

			Plane plane = planes[i];

			float d = distanceToPlane(origin, direction, plane);

			if(d > 0){
				closest = min(closest, d);
			}
			if(d > 0 && d != Infinity){
				farthest = max(farthest, d);
			}
		}

		float3 intersection = {
			origin.x + direction.x * farthest,
			origin.y + direction.y * farthest,
			origin.z + direction.z * farthest
		};

		return intersection;
	}

	bool contains(float3 point){
		for(int i = 0; i < 6; i++){

			Plane plane = planes[i];

			float d = distanceToPoint(point, plane);

			if(d < 0){
				return false;
			}
		}

		return true;
	}
};

bool intersectsFrustum(mat4 worldViewProj, float3 wgMin, float3 wgMax){

	float* values = reinterpret_cast<float*>(&worldViewProj);

	float m_0  = values[transposeIndex( 0)];
	float m_1  = values[transposeIndex( 1)];
	float m_2  = values[transposeIndex( 2)];
	float m_3  = values[transposeIndex( 3)];
	float m_4  = values[transposeIndex( 4)];
	float m_5  = values[transposeIndex( 5)];
	float m_6  = values[transposeIndex( 6)];
	float m_7  = values[transposeIndex( 7)];
	float m_8  = values[transposeIndex( 8)];
	float m_9  = values[transposeIndex( 9)];
	float m_10 = values[transposeIndex(10)];
	float m_11 = values[transposeIndex(11)];
	float m_12 = values[transposeIndex(12)];
	float m_13 = values[transposeIndex(13)];
	float m_14 = values[transposeIndex(14)];
	float m_15 = values[transposeIndex(15)];

	Plane planes[6] = {
		createPlane(m_3 - m_0, m_7 - m_4, m_11 -  m_8, m_15 - m_12),
		createPlane(m_3 + m_0, m_7 + m_4, m_11 +  m_8, m_15 + m_12),
		createPlane(m_3 + m_1, m_7 + m_5, m_11 +  m_9, m_15 + m_13),
		createPlane(m_3 - m_1, m_7 - m_5, m_11 -  m_9, m_15 - m_13),
		createPlane(m_3 - m_2, m_7 - m_6, m_11 - m_10, m_15 - m_14),
		createPlane(m_3 + m_2, m_7 + m_6, m_11 + m_10, m_15 + m_14),
	};
	
	for(int i = 0; i < 6; i++){

		Plane plane = planes[i];

		float3 vector;
		vector.x = plane.normal.x > 0.0 ? wgMax.x : wgMin.x;
		vector.y = plane.normal.y > 0.0 ? wgMax.y : wgMin.y;
		vector.z = plane.normal.z > 0.0 ? wgMax.z : wgMin.z;

		float d = distanceToPoint(vector, plane);

		if(d < 0){
			return false;
		}
	}

	return true;
}

// from three.js
// https://github.com/mrdoob/three.js/blob/dev/src/math/Ray.js
// License: MIT (https://github.com/mrdoob/three.js/blob/dev/LICENSE)
float raySphereIntersection(float3 origin, float3 dir, float3 spherePos, float radius) {

	float3 origToSphere = spherePos - origin;
	float tca = dot(origToSphere, dir);
	float d2 = dot(origToSphere, origToSphere) - tca * tca;
	float radius2 = radius * radius;

	if(d2 > radius2) return -1.0f;

	float thc = sqrt(radius2 - d2);
	float t0 = tca - thc;
	float t1 = tca + thc;

	if(t1 < 0) return -1.0f;
	if(t0 < 0) return t1;

	return t0;
}