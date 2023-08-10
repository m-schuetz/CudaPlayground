#pragma once

#include "CudaModularProgram.h"
#include "cuda.h"
#include "cuda_runtime.h"

struct CudaPrintBuffer{
	uint64_t CAPACITY = 1'000'000;
	uint64_t offset = 0;
	uint8_t* data = nullptr;
};

struct CudaPrint {
	uint32_t CAPACITY = 1'000'000;
	CUdeviceptr cptr;
	cudaStream_t cstream;

	CudaPrint(){

	}

	void init(){
		cuStreamCreate(&cstream, CU_STREAM_NON_BLOCKING);
		cuMemAlloc(&cptr, 16 + CAPACITY);
		// cuMemsetD32(cptr, CAPACITY, sizeof(CAPACITY));
		// cuMemsetD32(cptr, CAPACITY, sizeof(CAPACITY));

		struct {
			uint64_t CAPACITY = 1'000'000;
			uint64_t offset = 0;
		}initdata;

		cuMemcpyHtoD (cptr, &initdata, 16);
	}

};