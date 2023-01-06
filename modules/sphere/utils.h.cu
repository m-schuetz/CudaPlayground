
#pragma once

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define FALSE 0
#define TRUE 1

typedef unsigned int uint32_t;
typedef int int32_t;
typedef char int8_t;
typedef unsigned char uint8_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

#define Infinity 0x7f800000

// calls function <f> <size> times
// calls are distributed over all available threads
template<typename Function>
void processRange(int first, int size, Function&& f){

	uint32_t totalThreadCount = blockDim.x * gridDim.x;
	
	int itemsPerThread = size / totalThreadCount + 1;

	for(int i = 0; i < itemsPerThread; i++){
		int block_offset  = itemsPerThread * blockIdx.x * blockDim.x;
		int thread_offset = itemsPerThread * threadIdx.x;
		int index = first + block_offset + thread_offset + i;

		if(index >= first + size){
			break;
		}

		f(index);
	}
}

void printNumber(int64_t number, int leftPad = 0);

struct Allocator{

	uint8_t* buffer = nullptr;
	int64_t offset = 0;

	template<class T>
	Allocator(T buffer){
		this->buffer = reinterpret_cast<uint8_t*>(buffer);
		this->offset = 0;
	}

	Allocator(unsigned int* buffer, int64_t offset){
		this->buffer = reinterpret_cast<uint8_t*>(buffer);
		this->offset = offset;
	}

	template<class T>
	T alloc(int64_t size){

		auto ptr = reinterpret_cast<T>(buffer + offset);

		int64_t newOffset = offset + size;
		
		// make allocated buffer location 16-byte aligned to avoid 
		// potential problems with bad alignments
		int64_t remainder = (newOffset % 16ll);

		if(remainder != 0ll){
			newOffset = (newOffset - remainder) + 16ll;
		}
		
		this->offset = newOffset;

		return ptr;
	}

	template<class T>
	T alloc(int64_t size, const char* label){

		// if(isFirstThread()){
		// 	printf("offset: ");
		// 	printNumber(offset, 13);
		// 	printf(", allocating: ");
		// 	printNumber(size, 13);
		// 	printf(", label: %s \n", label);
		// }

		auto ptr = reinterpret_cast<T>(buffer + offset);

		int64_t newOffset = offset + size;
		
		// make allocated buffer location 16-byte aligned to avoid 
		// potential problems with bad alignments
		int64_t remainder = (newOffset % 16ll);

		if(remainder != 0ll){
			newOffset = (newOffset - remainder) + 16ll;
		}
		
		this->offset = newOffset;

		return ptr;
	}

};