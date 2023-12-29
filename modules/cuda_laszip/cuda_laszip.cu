
#include <cooperative_groups.h>

#include "utils.cuh"
#include "utils_hd.cuh"
#include "HostDeviceInterface.h"

#include "ArithmeticDecoder.cuh"
#include "IntegerCompressor.cuh"
#include "lasreaditemcompressed_v2.cuh"

namespace cg = cooperative_groups;

// struct float3{
// 	float x, y, z;
// };

// struct double3{
// 	double x, y, z;
// };

struct LasHeader{
	uint8_t versionMajor;
	uint8_t versionMinor;
	uint32_t offsetToPointData;
	uint32_t numVLRs;
	uint8_t point_data_format;
	uint16_t recordLength;
	uint32_t numPoints;
	uint16_t headerSize;

	double3 scale;
	double3 offset;
	double3 min;
	double3 max;
};



struct LaszipVlrItem{
	uint16_t type;
	uint16_t size;
	uint16_t version;
	uint16_t _padding;
};

constexpr int MAX_LASZIP_VLR_ITEMS = 10;

struct LaszipVlr{
	uint16_t compressor;
	uint16_t coder;
	uint8_t versionMajor;
	uint8_t versionMinor;
	uint16_t versionRevision;
	uint32_t options;
	uint32_t chunkSize;
	uint64_t numberOfSpecialEvlrs;
	uint64_t offsetToSpecialEvlrs;
	uint16_t numItems;
	LaszipVlrItem items[MAX_LASZIP_VLR_ITEMS];
};

LasHeader readLasHeader(uint8_t* buffer){
	LasHeader header;

	header.versionMajor        = readAs<uint8_t>(buffer, 24);
	header.versionMinor        = readAs<uint8_t>(buffer, 25);
	header.offsetToPointData   = readAs<uint32_t>(buffer, 96);
	header.numVLRs             = readAs<uint32_t>(buffer, 100);
	header.point_data_format   = readAs<uint8_t>(buffer, 104);
	header.recordLength        = readAs<uint16_t>(buffer, 105);
	header.numPoints           = readAs<uint32_t>(buffer, 107);
	header.headerSize          = readAs<uint16_t>(buffer, 94);

	header.scale.x = readAs<double>(buffer, 131);
	header.scale.y = readAs<double>(buffer, 139);
	header.scale.z = readAs<double>(buffer, 147);
	
	header.offset.x = readAs<double>(buffer, 155);
	header.offset.y = readAs<double>(buffer, 163);
	header.offset.z = readAs<double>(buffer, 171);
	
	header.max.x = readAs<double>(buffer, 179);
	header.max.y = readAs<double>(buffer, 195);
	header.max.z = readAs<double>(buffer, 211);
	
	header.min.x = readAs<double>(buffer, 187);
	header.min.y = readAs<double>(buffer, 203);
	header.min.z = readAs<double>(buffer, 219);
	
	if(header.versionMajor == 1 && header.versionMinor >= 4){
		header.numPoints = readAs<uint64_t>(buffer, 247);
	}

	return header;
}

LaszipVlr parseLaszipVlr(uint8_t* buffer){
	LaszipVlr vlr;

	vlr.compressor            = readAs<uint16_t>(buffer, 0);
	vlr.coder                 = readAs<uint16_t>(buffer, 2);
	vlr.versionMajor          = readAs<uint8_t>(buffer, 4);
	vlr.versionMinor          = readAs<uint8_t>(buffer, 5);
	vlr.versionRevision       = readAs<uint16_t>(buffer, 6);
	vlr.options               = readAs<uint32_t>(buffer, 8);
	vlr.chunkSize             = readAs<uint32_t>(buffer, 12);
	vlr.numberOfSpecialEvlrs  = readAs<uint64_t>(buffer, 16);
	vlr.offsetToSpecialEvlrs  = readAs<uint64_t>(buffer, 24);
	vlr.numItems              = readAs<uint16_t>(buffer, 32);

	if(vlr.numItems > MAX_LASZIP_VLR_ITEMS){
		printf("ERROR: vlr.numItems larger than MAX_LASZIP_VLR_ITEMS not implemented!!");
		
		return;
	}

	for(int i = 0; i < vlr.numItems; i++){
		LaszipVlrItem item; 
		item.type    = readAs<uint16_t>(buffer, 34 + i * 6 + 0);
		item.size    = readAs<uint16_t>(buffer, 34 + i * 6 + 2);
		item.version = readAs<uint16_t>(buffer, 34 + i * 6 + 4);
		
		vlr.items[i] = item;
	}

	return vlr;
}

extern "C" __global__
void kernel_read_chunk_infos(
	Uniforms uniforms,
	uint8_t* buffer,
	uint8_t* input,
	LasHeader* lasheader,
	Chunk* chunks,
	uint32_t* numChunks
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	AllocatorGlobal* g_allocator = (AllocatorGlobal*)buffer;

	if(grid.thread_rank() == 0){
		g_allocator->buffer = buffer;
		g_allocator->offset = 16;
	}

	// __shared__ LasHeader sh_header;

	if(grid.thread_rank() == 0){
		// sh_header = readLasHeader(input);
		*lasheader = readLasHeader(input);
		LaszipVlr laszipvlr;

		// READ LASZIP VLR
		uint64_t byteOffset = lasheader->headerSize;
		for(int i = 0; i < lasheader->numVLRs; i++){
			uint32_t recordId = readAs<uint16_t>(input, byteOffset + 18);
			uint32_t recordLength = readAs<uint16_t>(input, byteOffset + 20);

			if(recordId == 22204){
				laszipvlr = parseLaszipVlr(input + byteOffset + 54);
			}

			byteOffset = byteOffset + 54 + recordLength;
		}

		// READ CHUNK TABLE
		// see lasreadpoint.cpp; read_chunk_table()
		int64_t chunkTableStart = readAs<uint64_t>(input, byteOffset);
		int64_t chunkTableSize = uniforms.lazByteSize - chunkTableStart;
		uint8_t* chunkTableBuffer = input + chunkTableStart;
		uint32_t version = readAs<uint32_t>(chunkTableBuffer, 0);
		uint32_t numChunks = readAs<uint32_t>(chunkTableBuffer, 4);

		ArithmeticDecoder* dec = g_allocator->alloc<ArithmeticDecoder>(1);
		*dec = ArithmeticDecoder(chunkTableBuffer, 8);

		IntegerCompressor* ic = g_allocator->alloc<IntegerCompressor>(1);
		ic->init(dec, 32, 2);
		ic->initDecompressor(g_allocator);


		// read chunk byte sizes
		for (int i = 0; i < numChunks; i++) {
			int pred = (i == 0) ? 0 : chunks[i - 1].byteSize;
			int chunk_size = ic->decompress(pred, 1);
			chunks[i].byteSize = chunk_size;
		}

		int64_t firstChunkOffset = byteOffset + 8;

		// int64_t* chunk_starts = (int64_t*)malloc(8 * numChunks);
		chunks[0].byteOffset = firstChunkOffset;
		// chunk_starts[0] = firstChunkOffset;
		for (int i = 1; i <= numChunks; i++) {
			int64_t chunkStart = chunks[i - 1].byteOffset + int64_t(chunks[i - 1].byteSize);
			chunks[i].byteOffset = chunkStart;
		}

		// printf("chunks[0].byteOffset: %llu \n", chunks[0].byteOffset);

	}
}

extern "C" __global__
void kernel(
	Uniforms uniforms,
	uint8_t* buffer,
	uint8_t* input,
	LasHeader* lasheader,
	Chunk* chunks,
	uint32_t* numChunks
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	AllocatorGlobal* g_allocator = (AllocatorGlobal*)buffer;

	if(grid.thread_rank() == 0){
		g_allocator->buffer = buffer;
		g_allocator->offset = 16;
	}

	int chunkIndex = 0;
	Chunk chunk = chunks[chunkIndex];

	if(grid.thread_rank() == 0){

		int64_t offset_rgb = 0;
		if(lasheader->point_data_format % 128 == 2) offset_rgb = 20;
		if(lasheader->point_data_format % 128 == 3) offset_rgb = 28;

		int64_t offset_gps = 20;

		printf("lasheader->point_data_format: %i \n", lasheader->point_data_format);
		printf("offset_rgb: %i \n", offset_rgb);

		// int chunkIndex = 0;
		// Chunk chunk = chunks[chunkIndex];
		
		uint64_t start_offset = g_allocator->offset;

		// First point is uncompressed 
		uint8_t* chunkBuffer = input + chunk.byteOffset;
		int X = readAs<int32_t>(input, chunk.byteOffset + 0);
		int Y = readAs<int32_t>(input, chunk.byteOffset + 4);
		int Z = readAs<int32_t>(input, chunk.byteOffset + 8);

		uint16_t R = readAs<uint16_t>(input, chunk.byteOffset + offset_rgb + 0);
		uint16_t G = readAs<uint16_t>(input, chunk.byteOffset + offset_rgb + 2);
		uint16_t B = readAs<uint16_t>(input, chunk.byteOffset + offset_rgb + 4);

		uint16_t intensity = readAs<uint16_t>(input, chunk.byteOffset + 12);
		uint8_t returnNumber = readAs<uint8_t>(input, chunk.byteOffset + 14);
		uint8_t classification = readAs<uint8_t>(input, chunk.byteOffset + 15);
		double gpsTime = readAs<double>(input, chunk.byteOffset + offset_gps);

		printf("[%i] XYZ: %i, %i, %i   RGB:  %i, %i, %i  intensity: %i, Class.: %i, GPS: %f \n", 
			0, 
			X, Y, Z, 
			R, G, B,
			intensity, 
			classification, 
			float(gpsTime)
		);

		// Now start decompressing further points
		auto dec = new ArithmeticDecoder(chunkBuffer, lasheader->recordLength);
		// auto readerPoint10 = g_allocator->alloc<LASreadItemCompressed_POINT10_v2>(1);
		// auto readerRgb12   = g_allocator->alloc<LASreadItemCompressed_RGB12_v2>(1);

		__shared__ LASreadItemCompressed_POINT10_v2 readerPoint10;
		__shared__ LASreadItemCompressed_RGB12_v2 readerRgb12;
		__shared__ LASreadItemCompressed_GPSTIME11_v2 readerGps11;

		// printf("sizeof(LASreadItemCompressed_POINT10_v2): %llu \n", sizeof(LASreadItemCompressed_POINT10_v2)); 
		// printf("sizeof(LASreadItemCompressed_RGB12_v2): %llu \n", sizeof(LASreadItemCompressed_RGB12_v2)); 

		// printf("%i, %i, %i    -    %i, %i, %i \n", X, Y, Z, R, G, B);

		laszip_point lazpoint;
		uint8_t* ptr_XYZ = (uint8_t*)&lazpoint.X;
		uint8_t* ptr_GPS = (uint8_t*)&lazpoint.gps_time;
		uint8_t* ptr_RGB = (uint8_t*)&lazpoint.rgb[0];

		memset(&lazpoint, 0, sizeof(laszip_point));

		lazpoint.X = X;
		lazpoint.Y = Y;
		lazpoint.Z = Z;
		lazpoint.intensity = intensity;
		lazpoint.gps_time = gpsTime;

		// return number is a bit tricky since it's a bit field
		// access return number via intensity + 2, 
		// and copy our return number which also contains the other bitfield components
		uint8_t* ptrIntensity = (uint8_t*)(&lazpoint.intensity);
		memcpy(ptrIntensity + 2, &returnNumber, 1); 
		lazpoint.classification = classification;

		lazpoint.rgb[0] = R;
		lazpoint.rgb[1] = G;
		lazpoint.rgb[2] = B;

		// enableTrace = true;
		
		uint32_t context = 0;
		readerPoint10.init(dec, ptr_XYZ, context, g_allocator);
		readerGps11.init(dec, ptr_GPS, context, g_allocator);
		readerRgb12.init(dec, ptr_RGB, context, g_allocator);

		enableTrace = false;

		// enableTrace = true;

		// for (int i = 1; i < 2; i++) 
		uint64_t t_start = nanotime();
		// for (int i = 1; i < 5; i++) 
		for (int i = 1; i < 50'000; i++) 
		{
			dbg_pointIndex = i;
			// enableTrace = (i >= 20'000 && i < 20'001);
			// enableTrace = (i == 1000);

			if(enableTrace) printf("========================== \n");
			if(enableTrace) printf("== DECODING POINT %i \n", dbg_pointIndex);
			if(enableTrace) printf("========================= \n");
			
			readerPoint10.read(ptr_XYZ, context, g_allocator);
			readerGps11.read(ptr_GPS, context);
			readerRgb12.read(ptr_RGB, context);

			if(i < 5 || i == 49'999){
				int32_t X = readAs<int32_t>(ptr_XYZ, 0);
				int32_t Y = readAs<int32_t>(ptr_XYZ, 4);
				int32_t Z = readAs<int32_t>(ptr_XYZ, 8);

				uint16_t R = readAs<uint16_t>(ptr_RGB, 0);
				uint16_t G = readAs<uint16_t>(ptr_RGB, 2);
				uint16_t B = readAs<uint16_t>(ptr_RGB, 4);

				int intensity = readAs<uint16_t>(ptr_XYZ, 12);
				int classification = readAs<uint8_t>(ptr_XYZ, 15);
				double gpsTime = readAs<double>(ptr_GPS, 0);

				printf("[%i] XYZ: %i, %i, %i   RGB:  %i, %i, %i  intensity: %i, Class.: %i, GPS: %f \n", 
					i, 
					X, Y, Z, 
					R, G, B,
					intensity, 
					classification, 
					float(gpsTime)
				);
			}

		}

		uint64_t t_end = nanotime();
		uint64_t nanos = t_end - t_start;
		// float millies = double(nanos) / 1'000'000.0;
		printf("======================================================= \n");
		printf("decoding time:   %5.1f ms \n", float(double(nanos) / 1'000'000.0));
		printf("======================================================= \n");

		printf("t_readPoint10:   %5.1f ms \n", float(double(t_readPoint10) / 1'000'000.0));
		printf("t_readGps11:     %5.1f ms \n", float(double(t_readGps11) / 1'000'000.0));
		printf("t_readRgb12:     %5.1f ms \n", float(double(t_readRgb12) / 1'000'000.0));
		printf("sum:             %5.1f ms \n", float(double(t_readPoint10 + t_readGps11 + t_readRgb12) / 1'000'000.0));
		printf("update:          %5.1f ms \n", float(double(t_update) / 1'000'000.0));
		printf("decodeSymbol:    %5.1f ms \n", float(double(t_decodeSymbol) / 1'000'000.0));
		printf("decodeSymbols_0: %5.1f ms \n", float(double(t_decodeSymbols_0) / 1'000'000.0));
		printf("decodeSymbols_1: %5.1f ms \n", float(double(t_decodeSymbols_1) / 1'000'000.0));
		printf("renorm:          %5.1f ms \n", float(double(t_renorm) / 1'000'000.0));
		printf("readCorrector:   %5.1f ms \n", float(double(t_readCorrector) / 1'000'000.0));
		printf("streamingMedian: %5.1f ms \n", float(double(t_streamingMedian) / 1'000'000.0));
		printf("======================================================= \n");

		// printf("duration: %.3f ms \n", millies);

		uint64_t end_offset = g_allocator->offset;
		uint64_t allocatedBytes = end_offset - start_offset;
		printf("allocatedBytes: %llu kb \n", allocatedBytes / 1000);

	

	}

}
