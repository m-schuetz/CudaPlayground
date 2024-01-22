#pragma once


struct Texture{
	int width;
	int height;
	uint32_t* data;
};

Texture createGridTexture(){


	Texture texture;
	texture.width = 512;
	texture.height = 512;
	texture.data = allocator->alloc<uint32_t*>(4 * texture.width * texture.height);

	// grid.sync();

	processRange(0, texture.width * texture.height, [&](int index){
		
		int x = index % texture.width;
		int y = index / texture.width;

		uint32_t color;
		uint8_t* rgba = (uint8_t*)&color;

		if((x % 16) == 0 || (y % 16) == 0){
			color = 0x00000000;
		}else{
			color = 0x00aaaaaa;
		}

		texture.data[index] = color;

	});

	return texture;
}