
#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"

namespace cg = cooperative_groups;

constexpr int numElements = 1'000'000;

extern "C" __global__
void kernel(
	unsigned int* buffer,
	unsigned int* input
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// allows allocating bytes from buffer
	Allocator allocator(buffer, 0);

	// allocate array of random values and initialize random number generators
	uint32_t* randomValues = allocator.alloc<uint32_t*>(numElements * sizeof(uint32_t));
	curandStateXORWOW_t thread_random_state;
	curand_init(grid.thread_rank(), 0, 0, &thread_random_state);

	// store random values in array
	processRange(0, numElements, [&](int index){
		uint32_t randomValue = curand(&thread_random_state);
		uint32_t random_0_100 = randomValue % 101;
		randomValues[index] = random_0_100;
	});

	// globally sync all threads (wait until all numbers are generated)
	grid.sync();

	float average;
	uint64_t& sum = *allocator.alloc<uint64_t*>(8);
	{ // compute average
		if(grid.thread_rank() == 0){
			sum = 0;
		}
		grid.sync();

		// sum up all values
		processRange(0, numElements, [&](int index){
			atomicAdd(&sum, randomValues[index]);
		});

		grid.sync();
		
		average = double(sum) / double(numElements);
	}
	
	// print stats and some of the random numbers
	// disable printing to see real kernel performance
	// if(false)
	if(grid.thread_rank() == 0){

		printf("created ");
		printNumber(numElements);
		printf(" random numbers between [0, 100] \n");

		printf("sum:      ", sum);
		printNumber(sum, 10);

		printf("\n");
		printf("average:  %.2f \n", average);

		printf("values:   ");
		for(int i = 0; i < 10; i++){
			printf("%i, ", randomValues[i]);
		}
		printf("... \n");

		printf("===========\n");
		printf("#blocks:     %i \n", grid.num_blocks());
		printf("#blocksize:  %i \n", block.num_threads());
	}

	// if(grid.thread_rank() == 0){
	// 	printf("======================================================================================\n");
	// 	printf(" _    _   ______   _        _         ____       _____   _    _   _____            \n");
	// 	printf("| |  | | |  ____| | |      | |       / __ \\     / ____| | |  | | |  __ \\      /\\    \n");
	// 	printf("| |__| | | |__    | |      | |      | |  | |   | |      | |  | | | |  | |    /  \\   \n");
	// 	printf("|  __  | |  __|   | |      | |      | |  | |   | |      | |  | | | |  | |   / /\\ \\  \n");
	// 	printf("| |  | | | |____  | |____  | |____  | |__| |   | |____  | |__| | | |__| |  / ____ \\ \n");
	// 	printf("|_|  |_| |______| |______| |______|  \\____/     \\_____|  \\____/  |_____/  /_/    \\_\n");
	// 	printf("======================================================================================\n");
	// }

}
