

// This CUDA shader is adapted from "Seascape" by Alexander Alekseev aka TDM - 2014
// https://www.shadertoy.com/view/Ms2SD1
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.


#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"

namespace cg = cooperative_groups;

// #define AA

float time;
float width;
float height;

struct mat2{
	float2 rows[2];
};

struct mat3{
	float3 rows[3];
};

struct mat4{
	float4 rows[4];
};

float3 matMul(const mat3& m, const float3& v){
	return make_float3(
		dot(m.rows[0], v),
		dot(m.rows[1], v),
		dot(m.rows[2], v)
	);
}

float4 matMul(const mat4& m, const float4& v){
	return make_float4(
		dot(m.rows[0], v), 
		dot(m.rows[1], v), 
		dot(m.rows[2], v), 
		dot(m.rows[3], v)
	);
}

float3 operator*(const mat3& a, const float3& b){
	return make_float3(
		dot(a.rows[0], b),
		dot(a.rows[1], b),
		dot(a.rows[2], b)
	);
}

float3 operator*(const float3& b, const mat3& a){
	return make_float3(
		dot(a.rows[0], b),
		dot(a.rows[1], b),
		dot(a.rows[2], b)
	);
}

float2 operator*(const float2& b, const mat2& a){
	return make_float2(
		dot(a.rows[0], b),
		dot(a.rows[1], b)
	);
}


constexpr int NUM_STEPS = 8;
constexpr float PI      = 3.141592f;
constexpr float EPSILON = 1e-3f;
#define EPSILON_NRM (0.1 / width)
//#define AA

// sea
const float speed            = 0.3f;
const int ITER_GEOMETRY      = 3;
const int ITER_FRAGMENT      = 5;
const float SEA_HEIGHT       = 0.6;
const float SEA_CHOPPY       = 4.0;
const float SEA_SPEED        = 0.8;
const float SEA_FREQ         = 0.16;
const float3 SEA_BASE        = float3{0.0, 0.09, 0.18};
const float3 SEA_WATER_COLOR = float3{0.8 * 0.6, 0.9 * 0.6, 0.6 * 0.6};
#define SEA_TIME (1.0 + time * speed * SEA_SPEED)
const mat2 octave_m = mat2{1.6, 1.2, -1.2, 1.6};

float fract(float value){
	return value - floor(value);
}

float2 fract(float2 value){
	return {
		value.x - floor(value.x),
		value.y - floor(value.y),
	};
}

float mix(float x, float y, float a){
	return x * (1 - a) + y * a;
}

float2 mix(float2 x, float2 y, float2 a){
	return {
		x.x * (1 - a.x) + y.x * a.x,
		x.y * (1 - a.y) + y.y * a.y
	};
}

float3 mix(float3 x, float3 y, float a){
	return {
		x.x * (1 - a) + y.x * a,
		x.y * (1 - a) + y.y * a,
		x.z * (1 - a) + y.z * a
	};
}

float2 floor(float2 value){
	return {floor(value.x), floor(value.y)};
}

float2 sin(float2 value){
	return {sin(value.x), sin(value.y)};
}

float2 cos(float2 value){
	return {cos(value.x), cos(value.y)};
}

float2 abs(float2 value){
	return {abs(value.x), abs(value.y)};
}


// math
mat3 fromEuler(float3 ang) {
	float2 a1 = {sin(ang.x), cos(ang.x)};
	float2 a2 = {sin(ang.y), cos(ang.y)};
	float2 a3 = {sin(ang.z), cos(ang.z)};

	mat3 m;
	m.rows[0] = float3{a1.y*a3.y+a1.x*a2.x*a3.x,a1.y*a2.x*a3.x+a3.y*a1.x,-a2.y*a3.x};
	m.rows[1] = float3{-a2.y*a1.x,a1.y*a2.y,a2.x};
	m.rows[2] = float3{a3.y*a1.x*a2.x+a1.y*a3.x,a1.x*a3.x-a1.y*a3.y*a2.x,a2.y*a3.y};

	return m;
}

float hash(float2 p ) {
	float h = dot(p, float2{127.1f, 311.7f});

	return fract(sin(h) * 43758.5453123f);
}
float noise(float2& p) {
	float2 i = floor(p);
	float2 f = fract(p);
	float2 u = f * f * (3.0f - 2.0f * f);

	return -1.0f + 2.0f * mix( mix( hash( i + float2{0.0f, 0.0f} ), 
						hash( i + float2{1.0f, 0.0f} ), u.x),
				mix( hash( i + float2{0.0f, 1.0f} ), 
						hash( i + float2{1.0f, 1.0f} ), u.x), u.y);
}

// lighting
float diffuse(float3 n, float3 l, float p) {
	return pow(dot(n, l) * 0.4f + 0.6f, p);
}
float specular(float3 n, float3 l, float3 e, float s) {
	float nrm = (s + 8.0f) / (PI * 8.0f);
	return pow(max(dot(reflect(e,n), l), 0.0f), s) * nrm;
}

// sky
float3 getSkyColor(float3 e) {
	e.y = (max(e.y, 0.0f) * 0.8f + 0.2f) * 0.8f;
	return float3{pow(1.0f - e.y, 2.0f), 1.0f - e.y, 0.6f + (1.0f - e.y) * 0.4f} * 1.1f;
}

// sea
float sea_octave(float2 uv, float choppy) {
	uv += noise(uv);
	float2 wv = 1.0f - abs(sin(uv));
	float2 swv = abs(cos(uv));
	wv = mix(wv, swv, wv);

	return pow(1.0f - pow(wv.x * wv.y, 0.65f), choppy);
}

float map(float3 p) {
	float freq = SEA_FREQ;
	float amp = SEA_HEIGHT;
	float choppy = SEA_CHOPPY;
	float2 uv = {p.x, p.z};
	uv.x *= 0.75f;

	float d, h = 0.0f;
	for(int i = 0; i < ITER_GEOMETRY; i++) {
		d = sea_octave((uv + SEA_TIME) * freq, choppy);
		d += sea_octave((uv - SEA_TIME) * freq, choppy);
		h += d * amp;
		uv = uv * octave_m;
		freq *= 1.9f;
		amp *= 0.22f;
		choppy = mix(choppy, 1.0f, 0.2f);
	}

	return p.y - h;
}

float map_detailed(float3 p) {
	float freq = SEA_FREQ;
	float amp = SEA_HEIGHT;
	float choppy = SEA_CHOPPY;
	float2 uv = {p.x, p.z};
	uv.x *= 0.75f;

	float d, h = 0.0f;
	for(int i = 0; i < ITER_FRAGMENT; i++) {
		d = sea_octave((uv+SEA_TIME)*freq,choppy);
		d += sea_octave((uv-SEA_TIME)*freq,choppy);
		h += d * amp;
		uv = uv * octave_m;
		freq *= 1.9f;
		amp *= 0.22f;
		choppy = mix(choppy, 1.0f, 0.2f);
	}
	return p.y - h;
}

float3 getSeaColor(float3 p, float3 n, float3 l, float3 eye, float3 dist) {
	float fresnel = clamp(1.0f - dot(n,-eye), 0.0f, 1.0f);
	fresnel = pow(fresnel, 3.0f) * 0.5f;
		
	float3 reflected = getSkyColor(reflect(eye,n));
	float3 refracted = SEA_BASE + diffuse(n, l, 80.0f) * SEA_WATER_COLOR * 0.12f;

	float3 color = mix(refracted, reflected, fresnel);

	float atten = max(1.0f - dot(dist,dist) * 0.001f, 0.0f);
	float3 watercolor = SEA_WATER_COLOR * (p.y - SEA_HEIGHT) * 0.18f * atten;
	watercolor.x = clamp(watercolor.x, 0.0f, 1.0f);
	watercolor.y = clamp(watercolor.y, 0.0f, 1.0f);
	watercolor.z = clamp(watercolor.z, 0.0f, 1.0f);
	color += watercolor;

	float spec = specular(n, l, eye, 60.0f);
	color += float3{spec, spec, spec};

	return color;
}

// tracing
float3 getNormal(float3 p, float eps) {
	float3 n;
	n.y = map_detailed(p);
	n.x = map_detailed(float3{p.x + eps, p.y, p.z}) - n.y;
	n.z = map_detailed(float3{p.x, p.y, p.z + eps}) - n.y;
	n.y = eps;

	return normalize(n);
}

float heightMapTracing(float3 ori, float3 dir, float3 &p) {
	float tm = 0.0f;
	float tx = 1000.0f;
	float hx = map(ori + dir * tx);

	if(hx > 0.0f) {
		p = ori + dir * tx;
		return tx;
	}
	float hm = map(ori + dir * tm);
	float tmid = 0.0f;
	for(int i = 0; i < NUM_STEPS; i++) {
		tmid = mix(tm,tx, hm/(hm-hx));
		p = ori + dir * tmid;
		float hmid = map(p);
		if(hmid < 0.0f) {
			tx = tmid;
			hx = hmid;
		} else {
			tm = tmid;
			hm = hmid;
		}
	}
	return tmid;
}

uint4 sample(float u, float v, float time){

	u = 2.0f * u - 1.0f;
	v = 2.0f * v - 1.0f;

	time = time * speed;

	// ray
	float3 ang = {
		sin(time * 3.0f) * 0.1f, 
		sin(time) * 0.2f + 0.3f, 
		time};
	float3 ori = {0.0f, 3.5f, time * 5.0f};
	float3 dir = normalize(float3{u, v, -2.0f});
	dir.z += length(float2{u, v}) * 0.14f;
	dir = normalize(dir) * fromEuler(ang);

	// tracing
	float3 p;
	heightMapTracing(ori, dir, p);
	float3 dist = p - ori;
	float3 n = getNormal(p, dot(dist, dist) * EPSILON_NRM);
	float3 light = normalize(float3{0.0f, 1.0f, 0.8f}); 

	float3 skyColor = getSkyColor(dir);
	float3 seaColor = getSeaColor(p, n, light, dir, dist);
	float weight = pow(smoothstep(0.0f, -0.02f, dir.y), 0.2f);

	float3 rgb = mix(skyColor, seaColor, weight);

	float r = rgb.x;
	float g = rgb.y;
	float b = rgb.z;

	r = pow(r, 0.65f);
	g = pow(g, 0.65f);
	b = pow(b, 0.65f);

	uint32_t R = clamp(255.0f * r, 0.0, 255.0);
	uint32_t G = clamp(255.0f * g, 0.0, 255.0);
	uint32_t B = clamp(255.0f * b, 0.0, 255.0);

	uint4 color = {R, G, B, 255};

	return color;
}


extern "C" __global__
void kernel(
	unsigned int* buffer,
	cudaSurfaceObject_t gl_colorbuffer,
	int _width, int _height, float _time
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	time = _time;
	width = _width;
	height = _height;

	Allocator allocator(buffer, 0);

	processRange(0, width * height, [&](int pixelIndex){
		int x = pixelIndex % _width;
		int y = pixelIndex / _width;
		float2 fragCoord = {x, y};

		float u = float(x) / float(width - 1.0f);
		float v = float(y) / float(height - 1.0f);

		#if defined(AA)
			uint4 color = {0, 0, 0, 0};
			for(int i = -1; i <= 1; i++)
			for(int j = -1; j <= 1; j++){
				
				float u_aa = u + (float(i) / 3.0f) / width;
				float v_aa = v + (float(j) / 3.0f) / height;
				color += sample(u_aa, v_aa, time);
			}
			
			color.x = color.x / 9;
			color.y = color.y / 9;
			color.z = color.z / 9;
		#else
			uint4 color = sample(u, v, time);
		#endif
		
		uint32_t color_u32 = color.x | (color.y << 8) | (color.z << 16);

		surf2Dwrite(color_u32, gl_colorbuffer, x * 4, y);
	});

}
