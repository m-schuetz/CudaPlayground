#pragma once

bool enableTrace = false;
int dbg_pointIndex = 0;
uint64_t dbg_0 = 0;
uint64_t dbg_1 = 0;

uint64_t t_readPoint10 = 0;
uint64_t t_readGps11 = 0;
uint64_t t_readRgb12 = 0;
uint64_t t_update = 0;
uint64_t t_decodeSymbol = 0;
uint64_t t_renorm = 0;
uint64_t t_readCorrector = 0;
uint64_t t_streamingMedian = 0;
uint64_t t_decodeSymbols_0 = 0;
uint64_t t_decodeSymbols_1 = 0;
uint64_t t_decodeSymbols_2 = 0;


struct AllocatorGlobal{

	uint8_t* buffer = nullptr;
	uint64_t offset = 0;

	uint8_t* alloc(uint64_t size){

		// make allocated buffer location 16-byte aligned to avoid 
		// potential problems with bad alignments
		// round up to nearest 16
		uint64_t size_16 = 16ll * ((size + 16ll) / 16ll);

		uint64_t oldOffset = atomicAdd(&offset, size_16);

		uint8_t* ptr = buffer + oldOffset;

		return ptr;
	}

	template<typename T>
	T* alloc(uint64_t numItems){

		// make allocated buffer location sizeof(T) aligned to avoid 
		// potential problems with bad alignments

		uint64_t itemSize = sizeof(T);
		uint64_t byteSize = itemSize * (1 + numItems);
		uint64_t size_aligned = itemSize * ((byteSize + itemSize) / itemSize);

		uint64_t oldOffset = atomicAdd(&offset, size_aligned);
		uint64_t offset_aligned = oldOffset + itemSize - (oldOffset % itemSize);

		uint8_t* ptr = buffer + offset_aligned;

		return (T*)ptr;
	}
};