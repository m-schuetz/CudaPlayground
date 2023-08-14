#pragma once

#define FMT_HEADER_ONLY

#include "CudaModularProgram.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include "fmt/format.h"
//#include "fmt/core.h"
#include "fmt/args.h"

#include "HostDeviceInterface.h"

// #include "glm/common.hpp"
// #include "glm/matrix.hpp"

using glm::mat4;
using glm::mat3;

constexpr uint32_t MAX_CUDAPRINT_ENTRIES = 1000;

constexpr uint32_t TYPE_UINT32_T = 0;
constexpr uint32_t TYPE_UINT64_T = 1;
constexpr uint32_t TYPE_INT32_T  = 2;
constexpr uint32_t TYPE_INT64_T  = 3;
constexpr uint32_t TYPE_FLOAT    = 4;
constexpr uint32_t TYPE_DOUBLE   = 5;
constexpr uint32_t TYPE_FLOAT2   = 100;
constexpr uint32_t TYPE_FLOAT3   = 101;
constexpr uint32_t TYPE_FLOAT4   = 102;
constexpr uint32_t TYPE_MAT3     = 121;
constexpr uint32_t TYPE_MAT4     = 122;

constexpr uint32_t TYPE_SIZE_UINT32_T = sizeof(uint32_t);
constexpr uint32_t TYPE_SIZE_UINT64_T = sizeof(uint64_t);
constexpr uint32_t TYPE_SIZE_INT32_T  = sizeof(int32_t);
constexpr uint32_t TYPE_SIZE_INT64_T  = sizeof(int64_t);
constexpr uint32_t TYPE_SIZE_FLOAT    = sizeof(float);
constexpr uint32_t TYPE_SIZE_DOUBLE   = sizeof(double);
constexpr uint32_t TYPE_SIZE_FLOAT2   = sizeof(float2);
constexpr uint32_t TYPE_SIZE_FLOAT3   = sizeof(float3);
constexpr uint32_t TYPE_SIZE_FLOAT4   = sizeof(float4);
constexpr uint32_t TYPE_SIZE_MAT3     = sizeof(mat3);
constexpr uint32_t TYPE_SIZE_MAT4     = sizeof(mat4);

static int getByteSize(uint32_t type){

	if(type == TYPE_UINT32_T) return sizeof(uint32_t);
	if(type == TYPE_UINT64_T) return sizeof(uint64_t);
	if(type == TYPE_INT32_T)  return sizeof(int32_t);
	if(type == TYPE_INT64_T)  return sizeof(int64_t);
	if(type == TYPE_FLOAT)    return sizeof(float);
	if(type == TYPE_DOUBLE)   return sizeof(double);
	if(type == TYPE_FLOAT2)   return sizeof(float2);
	if(type == TYPE_FLOAT3)   return sizeof(float3);
	if(type == TYPE_FLOAT4)   return sizeof(float4);
	if(type == TYPE_MAT3)     return sizeof(mat3);
	if(type == TYPE_MAT4)     return sizeof(mat4);

}

struct CudaPrintArgument{
	uint32_t type;
	uint8_t data[1024];

	template<class T>
	T get() {

		T value;

		memcpy(&value, data, sizeof(T));

		return value;
	}
};

struct CudaPrintEntry{
	uint32_t keylen;

	uint8_t numArgs;
	uint8_t method;
	uint8_t padding_0;
	uint8_t padding_1;

	// uint32_t arglen;
	// uint32_t argtype;
	uint8_t data[1024 - 16];

	CudaPrintArgument getArgument(int index){

		uint8_t* pos = data + keylen;

		// skip bytes
		for(int i = 0; i < index; i++){
			uint32_t argtype = *((uint32_t*)pos);
			uint32_t argsize = getByteSize(argtype);

			pos += 4 + argsize;
		}

		uint32_t argtype = *((uint32_t*)pos);
		uint32_t argsize = getByteSize(argtype);

		CudaPrintArgument arg;
		arg.type = argtype;
		memcpy(arg.data, pos + 4, argsize);

		return arg;
	}
};

struct CudaPrintBuffer{
	uint64_t entryCounter = 0;
	uint64_t padding;
	CudaPrintEntry entries[MAX_CUDAPRINT_ENTRIES];
};

enum class CudaPrintStage{
	NONE = 0,
	LOADING_COUNTER = 1,
	LOADING_ENTRIES = 2,
};

struct CudaPrint {
	CUdeviceptr cptr;
	cudaStream_t cstream;
	uint64_t entryCounter = 0;
	uint8_t* data_pinned = nullptr;
	CudaPrintBuffer* printBuffer = nullptr;
	CUevent cevent_loadCounterFinished;
	CUevent cevent_loadEntriesFinished;

	unordered_map<string, CudaPrintEntry> table;
	vector<string> lines;

	CudaPrintStage stage = CudaPrintStage::NONE;

	CudaPrint(){

	}

	void init(){
		cuStreamCreate(&cstream, CU_STREAM_NON_BLOCKING);
		cuMemAlloc(&cptr, sizeof(CudaPrintBuffer));
		cuMemsetD32(cptr, 0, 4);
		cuEventCreate(&cevent_loadCounterFinished, 0);
		cuEventCreate(&cevent_loadEntriesFinished, 0);

		cuMemAllocHost((void**)&data_pinned , sizeof(CudaPrintBuffer));
		printBuffer = (CudaPrintBuffer*)data_pinned;


		// struct {
		// 	uint64_t CAPACITY = 1'000'000;
		// 	uint64_t offset = 0;
		// }initdata;

		// cuMemcpyHtoD (cptr, &initdata, 16);
	}

	void processEntry(CudaPrintEntry entry){
		const char* strStart = (const char*)entry.data;
		string strKey(strStart, entry.keylen);

		if(entry.method == 0){
			// METHOD PRINT LINE

			auto store = fmt::dynamic_format_arg_store<fmt::format_context>();
			
			for(int i = 0; i < entry.numArgs; i++){
				auto arg = entry.getArgument(i);

				if(arg.type == TYPE_INT32_T)        store.push_back(arg.get<int32_t>());
				else if(arg.type == TYPE_UINT32_T)  store.push_back(arg.get<uint32_t>());
				else if(arg.type == TYPE_FLOAT)     store.push_back(arg.get<float>());
				else                                store.push_back("missing arg handler");
			}

			fmt::vprint(strKey, store);

		}else if(entry.method == 1){
			// Method SET key/value
			table[strKey] = entry;
		}

		// printfmt("loaded entry! keylen: {:3}, key: {} \n", entry.keylen, strKey);
	}

	void update(){

		// always load metadata/counters each frame
		cuMemcpyDtoHAsync(data_pinned, cptr, 16, cstream);

		static struct{
			uint32_t start_first;
			uint32_t end_first;
			uint32_t start_second;
			uint32_t end_second;
		} loaded;

		if(stage == CudaPrintStage::NONE){
			// asynchronously load counter
			cuMemcpyDtoHAsync(data_pinned, cptr, 16, cstream);
			cuEventRecord(cevent_loadCounterFinished, cstream);

			stage = CudaPrintStage::LOADING_COUNTER;
			
		}else if(stage == CudaPrintStage::LOADING_COUNTER){

			// check if async load of counters is finished
			auto loadCountersFinished = cuEventQuery(cevent_loadCounterFinished) == CUDA_SUCCESS;
			if(loadCountersFinished){

				uint32_t firstEntry = this->entryCounter;
				uint32_t lastEntry = printBuffer->entryCounter;
				uint32_t numEntries = lastEntry - firstEntry;
				// uint32_t firstByte = 16 + firstEntry * sizeof(CudaPrintEntry);

				uint32_t start_first = this->entryCounter % MAX_CUDAPRINT_ENTRIES;
				uint32_t start_second = 0;
				uint32_t end_first, end_second;
				
				if((firstEntry % MAX_CUDAPRINT_ENTRIES ) <= (lastEntry % MAX_CUDAPRINT_ENTRIES)){
					end_first = start_first + numEntries;
					end_second = 0;
				}else{
					end_first = MAX_CUDAPRINT_ENTRIES;
					end_second = (start_first + numEntries) % MAX_CUDAPRINT_ENTRIES;
				}

				// memcpy entries (first part if ring-overflow)
				cuMemcpyDtoHAsync(
					data_pinned + 16 + start_first * sizeof(CudaPrintEntry), 
					cptr + 16 + start_first * sizeof(CudaPrintEntry), 
					(end_first - start_first) * sizeof(CudaPrintEntry), 
					cstream);

				// memcpy second part if ring-overflow, or nothing
				if(start_second != end_second)
				cuMemcpyDtoHAsync(
					data_pinned + 16 + start_second * sizeof(CudaPrintEntry), 
					cptr + 16 + start_second * sizeof(CudaPrintEntry), 
					(end_second - start_second) * sizeof(CudaPrintEntry), 
					cstream);

				// printfmt("counters(host/gpu): {:4} / {:4}, first: {:4} - {:4}, second: {:4} - {:4} \n", firstEntry, lastEntry, start_first, end_first, start_second, end_second);

				cuEventRecord(cevent_loadEntriesFinished, cstream);
				
				loaded.end_first = end_first;
				loaded.end_second = end_second;
				loaded.start_first = start_first;
				loaded.start_second = start_second;

				this->entryCounter = lastEntry;
				stage = CudaPrintStage::LOADING_ENTRIES;
			}else{
				// do nothing and try again next frame
			}


		}else if(stage == CudaPrintStage::LOADING_ENTRIES){
			// check if async load of entries is finished
			auto loadEntriesFinished = cuEventQuery(cevent_loadEntriesFinished) == CUDA_SUCCESS;
			if(loadEntriesFinished){

				for(int i = loaded.start_first; i < loaded.end_first; i++){
					auto entry = printBuffer->entries[i];
					
					processEntry(entry);
				}

				stage = CudaPrintStage::NONE;
			}else{
				// do nothing and try again next frame
			}
		}

		

		for(string line : lines){
			printfmt(line);
		}

		lines.clear();
		
		
		

	}

};