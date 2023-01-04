
# CUDA Playground

This project provides a basic template of a cuda program that can be <b>modified and executed</b> at runtime by saving the cuda kernel file (with ctrl+s). It's meant to help with developing algorithms by providing a near-instant feedback loop. 

* [randomNumbers.cu](./modules/randomNumbers/randomNumbers.cu) is the example cuda kernel that simply generates random numbers and then computes the average.
* It exploits some interesting and useful CUDA functionality such as:
	* cooperative groups, which enables you to globally sync all GPU threads via ```grid.sync()```. This, in turn, allows you to write megakernels where you can fuse all your kernels into one single large megakernel. If one pass depends on the results of another, you simply add grid.sync() to make sure all threads finished working on the first pass. 
	* CUDA runtime API which allows compiling CUDA code at runtime. 
	* It does not spawn threads/blocks based on the number of items that need to be processed, it spawns as many blocks as CUDA recommends for a given kernel code and workgroup size. It is up to you to make sure that a workload of X elements is processed with only Y blocks. Although a little more complex, it's essential if you have multiple passes with different workload sizes. 
	* However, the ```processRange(start, size, lambda)``` utility function helps you by calling function ```lambda``` exactly ```size``` times with respective indices. 
	```
	// distributes <numElements> calls to the given lambda function to all GPU threads. 
	processRange(0, numElements, [&](int index){
		atomicAdd(&sum, elements[index]);
	});
	```
	* To avoid the need to adjust bindings between host and device (and recompilation of C++ host code), all buffers and counters are allocated directly in the CUDA program. The host only passes a simple byte buffer, and the device code allocates within that buffer. For example, to allocate an array of ```numElements``` integers, call:
	```
	uint32_t* randomValues = allocator.alloc<uint32_t*>(numElements * sizeof(uint32_t));
	```
	Note that _all_ threads need to do the same allocations in the same order, as each thread keeps track of its own offsets into the buffer. All threads need to agree on the same offsets. 
	You can also use this to allocate new counter variables at runtime, e.g.:
	```
	uint64_t& counter = *allocator.alloc<uint64_t*>(8);
	if(grid.thread_rank() == 0){
		counter = 0;
	}
	grid.sync();
	atomicAdd(&counter, 1);
	```
* ```"--gpu-architecture=compute_75"``` is hardcoded in CudaModularProgram.h. You may want to change this to access newer features. 
