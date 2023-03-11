#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

struct{
	uint32_t* img_ascii_16;
	uint64_t* framebuffer;
} globals;

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

namespace cg = cooperative_groups;

Uniforms uniforms;
Allocator* allocator;
uint64_t nanotime_start;

constexpr float PI = 3.1415;
constexpr uint32_t BACKGROUND_COLOR = 0x00332211ull;

struct Triangles{
	int numTriangles;
	float3* positions;
	float2* uvs;
	uint32_t* colors;
};

struct Texture{
	int width;
	int height;
	uint32_t* data;
};

struct RasterizationSettings{
	Texture* texture = nullptr;
	int colorMode = COLORMODE_TRIANGLE_ID;
	mat4 world;
};

struct Tileset{

	Texture texture;
	int width;
	int height;
	int tilesize;

};

struct Layer{
	Tileset tileset;
	uint32_t* data;
};

struct Map{

	int width;
	int height;

	Layer layer_0;

};

void drawMap(Map* map, uint64_t* framebuffer){

	auto grid = cg::this_grid();

	uint32_t numTiles = map->width * map->height;
	uint32_t pixelScale = 4;
	uint32_t tilesize = 16;

	uint32_t px_width = map->width * pixelScale * tilesize;
	uint32_t px_height = map->height * pixelScale * tilesize;
	
	// if(grid.thread_rank() == 0)
	// 	printf("%u \n", px_width);

	processRange(px_width * px_height, [&](int localPixelIndex){

		Layer& layer = map->layer_0;
		Tileset& tileset = layer.tileset;

		uint32_t lpx = localPixelIndex % px_width;
		uint32_t lpy = localPixelIndex / px_width;

		uint32_t maptile_x = (lpx / pixelScale) / tilesize;
		uint32_t maptile_y = (lpy / pixelScale) / tilesize;
		uint32_t maptileID = maptile_x + maptile_y * map->width;

		uint32_t tileID = layer.data[maptileID];
		uint32_t tileX = tileID % tileset.width;
		uint32_t tileY = tileID / tileset.width;

		uint32_t tox = tileset.tilesize * tileX;
		uint32_t toy = tileset.tilesize * tileY;
		uint32_t maptile_local_x = (lpx / pixelScale) % tileset.tilesize;
		uint32_t maptile_local_y = (lpy / pixelScale) % tileset.tilesize;

		uint32_t target_x = lpx + 1200;
		uint32_t target_y = lpy +  50;

		uint32_t color = 0;
		uint8_t* rgba = (uint8_t*)&color;

		uint32_t tx = tox + maptile_local_x;
		uint32_t ty = (tileset.texture.height - 1) - (toy + maptile_local_y);
		uint32_t texelID = tx + ty * tileset.texture.width;
		color = tileset.texture.data[texelID];

		// if(lpx == 136)
		// if(lpy == 10)
		// {
		// 	printf("%u \n", ty);
		// }

		uint64_t encoded = color;
		uint32_t targetPixelIndex = target_x + uniforms.width * target_y;

		framebuffer[targetPixelIndex] = encoded;
	});

}

void drawText(const char* text, float x, float y, float fontsize){

	auto grid = cg::this_grid();

	uint32_t* image = globals.img_ascii_16;

	float tilesize = 16;
	int NUM_CHARS = 95;

	int numchars = strlen(text);

	// one char after the other, utilizing 10k threads for each char haha
	for(int i = 0; i < numchars; i++){

		int charcode = text[i];
		int tilepx = (charcode - 32) * tilesize;

		processRange(ceil(fontsize) * ceil(fontsize), [&](int index){
			int l_x = index % int(ceil(fontsize));
			int l_y = index / int(ceil(fontsize));

			float u = float(l_x) / fontsize;
			float v = 1.0f - float(l_y) / fontsize;

			int sx = tilepx + tilesize * u;
			int sy = tilesize * v;
			int sourceTexel = sx + sy * NUM_CHARS * tilesize;

			uint32_t color = color = image[sourceTexel];
			uint8_t* rgba = (uint8_t*)&color;

			int t_x = l_x + x + i * fontsize;
			int t_y = l_y + y;
			int targetPixelIndex = t_x + t_y * uniforms.width;

			// blend with current framebuffer value
			uint64_t current = globals.framebuffer[targetPixelIndex];
			uint32_t currentColor = current & 0xffffffff;
			uint8_t* currentRGBA = (uint8_t*)&currentColor;

			float w = float(rgba[3]) / 255.0f;
			rgba[0] = (1.0f - w) * float(currentRGBA[0]) + w * rgba[0];
			rgba[1] = (1.0f - w) * float(currentRGBA[1]) + w * rgba[1];
			rgba[2] = (1.0f - w) * float(currentRGBA[2]) + w * rgba[2];
			
			globals.framebuffer[targetPixelIndex] = color;
		});

		grid.sync();
	}
}

extern "C" __global__
void kernel(
	const Uniforms _uniforms,
	unsigned int* buffer,
	cudaSurfaceObject_t gl_colorbuffer,
	uint32_t numTriangles,
	float3* positions,
	float2* uvs,
	uint32_t* colors,
	uint32_t* textureData,
	uint32_t* img_ascii_16
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime_start));

	uniforms = _uniforms;

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	// allocate framebuffer memory
	int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
	uint64_t* framebuffer = allocator->alloc<uint64_t*>(framebufferSize);

	globals.img_ascii_16 = img_ascii_16;
	globals.framebuffer = framebuffer;

	Texture texture;
	texture.width  = 256;
	texture.height = 256;
	texture.data   = textureData;

	int tilesize = 16;
	
	Tileset tileset;
	tileset.width = texture.width / tilesize;
	tileset.height = texture.width / tilesize;
	tileset.tilesize = tilesize;
	tileset.texture = texture;

	Map map;
	map.width = 13;
	map.height = 10;
	map.layer_0.tileset = tileset;
	map.layer_0.data = allocator->alloc<uint32_t*>(map.width * map.height * sizeof(uint32_t));

	processRange(map.width * map.height, [&](int index){
		map.layer_0.data[index] = index;
	});

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
	});

	grid.sync();

	drawMap(&map, framebuffer);

	processRange(0, 1024 * 1024, [&](int localPixelIndex){

		int x = localPixelIndex % 1024;
		int y = localPixelIndex / 1024;

		float2 uv = {
			float(x) / 1024.0f,
			float(y) / 1024.0f
		};

		int tx = x / 4;
		int ty = (texture.height - 1) - y / 4;
		int texelIndex = tx + texture.width * ty;
		uint32_t texel = texture.data[texelIndex];

		uint32_t color = texel;

		x = x + 50;
		y = y + 50;

		uint64_t encoded = color;

		uint32_t targetPixelIndex = x + uniforms.width * y;

		framebuffer[targetPixelIndex] = encoded;
	});

	grid.sync();

	drawText("Draw Text with CUDA!", 50, 1200, 64.0f);
	drawText("And experimenting with tilesets", 50, 1150, 32.0f);
	
	// DRAW TILE COORDINATES
	for(int i = 0; i < 16; i++){
		uint8_t text[3];
		if(i < 10){
			text[0] = i + '0';
			text[1] = 0;
		}else{
			text[0] = (i / 10) + '0';
			text[1] = (i % 10) + '0';
			text[2] = 0;
		}

		float fontsize = 32.0f;
		drawText((const char*)&text, 65 + i * fontsize * 2.0f, 1075, fontsize);
		drawText((const char*)&text, 15, 1025 - i * fontsize * 2.0f, fontsize);
	}

	grid.sync();

	// transfer framebuffer to opengl texture
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){

		int x = pixelIndex % int(uniforms.width);
		int y = pixelIndex / int(uniforms.width);

		uint64_t encoded = framebuffer[pixelIndex];
		uint32_t color = encoded & 0xffffffffull;

		surf2Dwrite(color, gl_colorbuffer, x * 4, y);
	});


}