
#include <cooperative_groups.h>

#include "utils.cuh"

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
void kernel(
	unsigned int* buffer,
	uint8_t* input
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// allows allocating bytes from buffer
	Allocator allocator(buffer, 0);


	if(grid.thread_rank() == 0){
		printf("cuda test \n");
		printf("input: %llu \n", input);

		LasHeader header = readLasHeader(input);

		printf("numVLRs: %i \n", header.numVLRs);
		printf("recordLength: %i \n", header.recordLength);
		printf("min: %.2f, %.2f, %.2f \n", float(header.min.x), float(header.min.y), float(header.min.z));
		printf("max: %.2f, %.2f, %.2f \n", float(header.max.x), float(header.max.y), float(header.max.z));

		uint64_t byteOffset = header.headerSize;
		for(int i = 0; i < header.numVLRs; i++){
			// let vlrHeaderBuffer = allocator.alloc
			// await handle.read(vlrHeaderBuffer, 0, vlrHeaderBuffer.byteLength, byteOffset);

			// console.log("vlr");
			
			// let userId = parseCString(vlrHeaderBuffer.slice(2, 18));
			// let recordId = vlrHeaderBuffer.readUInt16LE(18);
			// let recordLength = vlrHeaderBuffer.readUInt16LE(20);

			uint32_t recordId = readAs<uint16_t>(input, byteOffset + 18);
			uint32_t recordLength = readAs<uint16_t>(input, byteOffset + 20);

			if(recordId == 22204){
				auto vlr = parseLaszipVlr(input + byteOffset + 54);

				printf("compressor            %i \n", vlr.compressor);
				printf("coder                 %i \n", vlr.coder);
				printf("versionMajor          %i \n", vlr.versionMajor);
				printf("versionMinor          %i \n", vlr.versionMinor);
				printf("versionRevision       %i \n", vlr.versionRevision);
				printf("options               %i \n", vlr.options);
				printf("chunkSize             %i \n", vlr.chunkSize);
				printf("numberOfSpecialEvlrs  %i \n", vlr.numberOfSpecialEvlrs);
				printf("offsetToSpecialEvlrs  %i \n", vlr.offsetToSpecialEvlrs);
				printf("numItems              %i \n", vlr.numItems);
			}

			printf("vlr recordId: %i \n", recordId);
			printf("vlr recordLength: %i \n", recordLength);

			byteOffset = byteOffset + 54 + recordLength;
		}
	}

}
