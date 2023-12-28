//
//#include "ArithmeticDecoder.cuh"
//#include "IntegerCompressor.cuh"
//#include "lasreaditemcompressed_v2.cuh"
//
//#include "unsuck.hpp"
//
//
//
//namespace testLaszipClasses{
//
//	template<typename T>
//	T readAs(uint8_t* buffer, uint64_t offset) {
//		T value;
//		memcpy(&value, buffer + offset, sizeof(T));
//
//		return value;
//	}
//
//	struct float3 {
//		float x, y, z;
//	};
//
//	struct double3 {
//		double x, y, z;
//	};
//
//	struct LasHeader {
//		uint8_t versionMajor;
//		uint8_t versionMinor;
//		uint32_t offsetToPointData;
//		uint32_t numVLRs;
//		uint8_t point_data_format;
//		uint16_t recordLength;
//		uint32_t numPoints;
//		uint16_t headerSize;
//
//		double3 scale;
//		double3 offset;
//		double3 min;
//		double3 max;
//	};
//
//	struct LaszipVlrItem {
//		uint16_t type;
//		uint16_t size;
//		uint16_t version;
//		uint16_t _padding;
//	};
//
//	constexpr int MAX_LASZIP_VLR_ITEMS = 10;
//
//	struct LaszipVlr {
//		uint16_t compressor;
//		uint16_t coder;
//		uint8_t versionMajor;
//		uint8_t versionMinor;
//		uint16_t versionRevision;
//		uint32_t options;
//		uint32_t chunkSize;
//		uint64_t numberOfSpecialEvlrs;
//		uint64_t offsetToSpecialEvlrs;
//		uint16_t numItems;
//		LaszipVlrItem items[MAX_LASZIP_VLR_ITEMS];
//	};
//
//	LasHeader readLasHeader(uint8_t* buffer) {
//		LasHeader header;
//
//		header.versionMajor = readAs<uint8_t>(buffer, 24);
//		header.versionMinor = readAs<uint8_t>(buffer, 25);
//		header.offsetToPointData = readAs<uint32_t>(buffer, 96);
//		header.numVLRs = readAs<uint32_t>(buffer, 100);
//		header.point_data_format = readAs<uint8_t>(buffer, 104);
//		header.recordLength = readAs<uint16_t>(buffer, 105);
//		header.numPoints = readAs<uint32_t>(buffer, 107);
//		header.headerSize = readAs<uint16_t>(buffer, 94);
//
//		header.scale.x = readAs<double>(buffer, 131);
//		header.scale.y = readAs<double>(buffer, 139);
//		header.scale.z = readAs<double>(buffer, 147);
//
//		header.offset.x = readAs<double>(buffer, 155);
//		header.offset.y = readAs<double>(buffer, 163);
//		header.offset.z = readAs<double>(buffer, 171);
//
//		header.max.x = readAs<double>(buffer, 179);
//		header.max.y = readAs<double>(buffer, 195);
//		header.max.z = readAs<double>(buffer, 211);
//
//		header.min.x = readAs<double>(buffer, 187);
//		header.min.y = readAs<double>(buffer, 203);
//		header.min.z = readAs<double>(buffer, 219);
//
//		if (header.versionMajor == 1 && header.versionMinor >= 4) {
//			header.numPoints = readAs<uint64_t>(buffer, 247);
//		}
//
//		return header;
//	}
//
//	LaszipVlr parseLaszipVlr(uint8_t* buffer) {
//		LaszipVlr vlr;
//
//		vlr.compressor = readAs<uint16_t>(buffer, 0);
//		vlr.coder = readAs<uint16_t>(buffer, 2);
//		vlr.versionMajor = readAs<uint8_t>(buffer, 4);
//		vlr.versionMinor = readAs<uint8_t>(buffer, 5);
//		vlr.versionRevision = readAs<uint16_t>(buffer, 6);
//		vlr.options = readAs<uint32_t>(buffer, 8);
//		vlr.chunkSize = readAs<uint32_t>(buffer, 12);
//		vlr.numberOfSpecialEvlrs = readAs<uint64_t>(buffer, 16);
//		vlr.offsetToSpecialEvlrs = readAs<uint64_t>(buffer, 24);
//		vlr.numItems = readAs<uint16_t>(buffer, 32);
//
//		if (vlr.numItems > MAX_LASZIP_VLR_ITEMS) {
//			printf("ERROR: vlr.numItems larger than MAX_LASZIP_VLR_ITEMS not implemented!!");
//
//			return vlr;
//		}
//
//		for (int i = 0; i < vlr.numItems; i++) {
//			LaszipVlrItem item;
//			item.type = readAs<uint16_t>(buffer, 34 + i * 6 + 0);
//			item.size = readAs<uint16_t>(buffer, 34 + i * 6 + 2);
//			item.version = readAs<uint16_t>(buffer, 34 + i * 6 + 4);
//
//			vlr.items[i] = item;
//		}
//
//		return vlr;
//	}
//
//	void run(string path){
//
//		auto lazbuffer = readBinaryFile(path);
//
//		uint8_t* input = lazbuffer->data_u8;
//
//		LasHeader header = readLasHeader(input);
//		LaszipVlr laszipvlr;
//
//		printf("numVLRs: %i \n", header.numVLRs);
//		printf("recordLength: %i \n", header.recordLength);
//		printf("min: %.2f, %.2f, %.2f \n", float(header.min.x), float(header.min.y), float(header.min.z));
//		printf("max: %.2f, %.2f, %.2f \n", float(header.max.x), float(header.max.y), float(header.max.z));
//
//		// READ LASZIP VLR
//		uint64_t byteOffset = header.headerSize;
//		for (int i = 0; i < header.numVLRs; i++) {
//			uint32_t recordId = readAs<uint16_t>(input, byteOffset + 18);
//			uint32_t recordLength = readAs<uint16_t>(input, byteOffset + 20);
//
//			if (recordId == 22204) {
//				laszipvlr = parseLaszipVlr(input + byteOffset + 54);
//			}
//
//			// printf("vlr recordId: %i \n", recordId);
//			// printf("vlr recordLength: %i \n", recordLength);
//
//			byteOffset = byteOffset + 54 + recordLength;
//		}
//
//		printf("compressor            %i \n", laszipvlr.compressor);
//		printf("coder                 %i \n", laszipvlr.coder);
//		printf("versionMajor          %i \n", laszipvlr.versionMajor);
//		printf("versionMinor          %i \n", laszipvlr.versionMinor);
//		printf("versionRevision       %i \n", laszipvlr.versionRevision);
//		printf("options               %i \n", laszipvlr.options);
//		printf("chunkSize             %i \n", laszipvlr.chunkSize);
//		printf("numberOfSpecialEvlrs  %i \n", laszipvlr.numberOfSpecialEvlrs);
//		printf("offsetToSpecialEvlrs  %i \n", laszipvlr.offsetToSpecialEvlrs);
//		printf("numItems              %i \n", laszipvlr.numItems);
//
//		// READ CHUNK TABLE
//		// see lasreadpoint.cpp; read_chunk_table()
//		int64_t chunkTableStart = readAs<uint64_t>(input, byteOffset);
//		int64_t chunkTableSize = lazbuffer->size - chunkTableStart;
//		uint8_t* chunkTableBuffer = input + chunkTableStart;
//
//		uint32_t version = readAs<uint32_t>(chunkTableBuffer, 0);
//		uint32_t numChunks = readAs<uint32_t>(chunkTableBuffer, 4);
//
//		printf("chunkTableStart: %lli \n", chunkTableStart);
//		printf("chunkTableSize:  %lli \n", chunkTableSize);
//		printf("version:  %lli \n", version);
//		printf("numChunks:  %lli \n", numChunks);
//
//		auto dec = new ArithmeticDecoder(chunkTableBuffer, 8);
//		auto ic = new IntegerCompressor(dec, 32, 2);
//		ic->initDecompressor();
//
//		int32_t* chunk_sizes = (int32_t * )malloc(4 * numChunks);
//		for (int i = 0; i < numChunks; i++) chunk_sizes[i] = 0;
//
//		// read chunk byte sizes
//		for (int i = 0; i < numChunks; i++) {
//			int pred = (i == 0) ? 0 : chunk_sizes[i - 1];
//			int chunk_size = ic->decompress(pred, 1);
//			chunk_sizes[i] = chunk_size;
//		}
//
//		// header + vlrs + 8 bytes describing chunk table location
//		int64_t firstChunkOffset = byteOffset + 8;
//
//		int64_t* chunk_starts = (int64_t*)malloc(8 * numChunks);
//		chunk_starts[0] = firstChunkOffset;
//		for (int i = 1; i <= numChunks; i++) {
//			int64_t chunkStart = chunk_starts[i - 1] + int64_t(chunk_sizes[i - 1]);
//			chunk_starts[i] = chunkStart;
//		}
//
//		// try reading a chunk
//		// first point is uncompressed
//
//		{
//
//			int chunkIndex = 0;
//			int64_t chunkStart = chunk_starts[chunkIndex];
//			uint8_t* chunkBuffer = input + chunkStart;
//			int X = readAs<int32_t>(input, chunkStart + 0);
//			int Y = readAs<int32_t>(input, chunkStart + 4);
//			int Z = readAs<int32_t>(input, chunkStart + 8);
//
//			uint16_t R = readAs<uint16_t>(input, chunkStart + 20);
//			uint16_t G = readAs<uint16_t>(input, chunkStart + 22);
//			uint16_t B = readAs<uint16_t>(input, chunkStart + 24);
//
//			// Now start decompressing further points
//			auto dec = new ArithmeticDecoder(chunkBuffer, header.recordLength);
//			auto readerPoint10 = new LASreadItemCompressed_POINT10_v2(dec);
//			auto readerRgb12 = new LASreadItemCompressed_RGB12_v2(dec);
//
//			uint32_t context = 0;
//			uint8_t itemPoint10[20] = { 0 };
//			memcpy(itemPoint10 + 0, &X, 4);
//			memcpy(itemPoint10 + 4, &Y, 4);
//			memcpy(itemPoint10 + 8, &Z, 4);
//			readerPoint10->init(itemPoint10, context);
//
//
//			uint8_t itemRgb12[6] = { 0 };
//			memcpy(itemRgb12 + 0, &R, 2);
//			memcpy(itemRgb12 + 2, &G, 2);
//			memcpy(itemRgb12 + 4, &B, 2);
//			readerRgb12->init(itemRgb12, context);
//
//			for (int i = 0; i < 10; i++) {
//				
//				readerPoint10->read(itemPoint10, context);
//				readerRgb12->read(itemRgb12, context);
//
//				int32_t X = readAs<int32_t>(itemPoint10, 0);
//				int32_t Y = readAs<int32_t>(itemPoint10, 4);
//				int32_t Z = readAs<int32_t>(itemPoint10, 8);
//
//				uint16_t R = readAs<uint16_t>(itemRgb12, 0);
//				uint16_t G = readAs<uint16_t>(itemRgb12, 2);
//				uint16_t B = readAs<uint16_t>(itemRgb12, 4);
//
//				printfmt("{}, {}, {}   -   {}, {}, {} \n", X, Y, Z, R, G, B);
//
//				int a = 10;
//			}
//
//
//		}
//
//
//
//
//
//
//		// ArithmeticDecoder;
//
//	}
//}
