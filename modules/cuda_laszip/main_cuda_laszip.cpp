

#include <iostream>
#include <filesystem>
#include<locale.h>
#include <string>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>
#include <format>

#include "CudaModularProgram.h"

#include "HostDeviceInterface.h"
#include "unsuck.hpp"

#include "ArithmeticDecoder.cuh"
#include "testLaszipClasses.h"

using namespace std;

CUdeviceptr cptr_buffer, cptr_input;
CudaModularProgram* cuda_program = nullptr;

// string lazfile = "E:/dev/pointclouds/archpro/heidentor.laz";
string lazfile = "E:/resources/pointclouds/archpro/heidentor.laz";
int64_t lazByteSize = 0;

void initCuda(){
	cuInit(0);
	CUdevice cuDevice;
	CUcontext context;
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&context, 0, cuDevice);
}

void loadData(){
	auto lazbuffer = readBinaryFile(lazfile);
	lazByteSize = lazbuffer->size;

	auto result = cuMemAlloc(&cptr_input, lazbuffer->size);

	if (result != CUDA_SUCCESS) {
		exit(123);
	}

	cuMemcpyHtoD(cptr_input, lazbuffer->data, lazbuffer->size);
}

// void testLaszipClasses(string path){
// 	auto lazbuffer = readBinaryFile(lazfile);




// 	ArithmeticDecoder ac(buffer, offset);
// }

void runCudaProgram(){

	cout << "================================================================================" << endl;
	cout << "=== RUNNING" << endl;
	cout << "================================================================================" << endl;

	CUresult resultcode = CUDA_SUCCESS;
	CUevent cevent_start, cevent_end;
	cuEventCreate(&cevent_start, 0);
	cuEventCreate(&cevent_end, 0);

	CUdevice device;
	int numSMs;
	cuCtxGetDevice(&device);
	cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

	int workgroupSize = 256;

	int numGroups;
	cuOccupancyMaxActiveBlocksPerMultiprocessor(&numGroups, cuda_program->kernels["kernel"], workgroupSize, 0);
	numGroups *= numSMs;
	
	// make sure at least 10 workgroups are spawned)
	numGroups = std::clamp(numGroups, 10, numGroups);

	cuEventRecord(cevent_start, 0);

	Uniforms uniforms;
	uniforms.lazByteSize = lazByteSize;

	void* args[] = {&uniforms, &cptr_buffer, &cptr_input};

	auto res_launch = cuLaunchCooperativeKernel(cuda_program->kernels["kernel"],
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);

	if(res_launch != CUDA_SUCCESS){
		const char* str; 
		cuGetErrorString(res_launch, &str);
		printf("error: %s \n", str);
	}

	cuEventRecord(cevent_end, 0);
	cuEventSynchronize(cevent_end);

	{
		float total_ms;
		cuEventElapsedTime(&total_ms, cevent_start, cevent_end);

		cout << "CUDA durations: " << endl;
		cout << std::format("total:     {:6.1f} ms", total_ms) << endl;
	}

	cuCtxSynchronize();

	// Buffer buffer(100);
	// cuMemcpyDtoH(buffer.data, cptr_buffer, buffer.size);

	// cout << "print generated random values from host: ";
	// for(int i = 0; i < 10; i++){
	// 	cout << buffer.get<uint32_t>(4 * i) << ", ";
	// }
	// cout << " ..." << endl;

}

void initCudaProgram(){

	cuMemAlloc(&cptr_buffer, 100'000'000);
	// cuMemAlloc(&cptr_input, 100'000'000);

	// cuMemcpyHtoD(cptr_input, input.data(), input.size() * sizeof(int));

	cuda_program = new CudaModularProgram({
		.modules = {
			"./modules/cuda_laszip/cuda_laszip.cu",
			"./modules/cuda_laszip/utils.cu",
		},
		.kernels = {"kernel"}
	});

	cuda_program->onCompile([&](){
		runCudaProgram();
	});
}



int main(){

	cout << std::setprecision(2) << std::fixed;
	setlocale( LC_ALL, "en_AT.UTF-8" );

	initCuda();
	initCudaProgram();

	loadData();

	testLaszipClasses::run(lazfile);

	runCudaProgram();

	while(true){
		EventQueue::instance->process();

		std::this_thread::sleep_for(1ms);
	}

	return 0;
}
