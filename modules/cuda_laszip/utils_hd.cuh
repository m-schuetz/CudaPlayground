#pragma once

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