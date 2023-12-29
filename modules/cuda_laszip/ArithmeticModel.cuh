// Adapted from LASzip
// https://github.com/LASzip/LASzip/blob/master/src/arithmeticmodel.hpp
// LICENSE: https://github.com/LASzip/LASzip/blob/80f929870f4ae2c3dc4e4f9e00e0e68f17ba1f1e/COPYING (Apache 2.0)
//
// Which in turn is an adaption from Amir Said's FastAC Code.
// see: http://www.cipr.rpi.edu/~said/FastAC.html

#pragma once


inline constexpr uint32_t AC__MinLength = 0x01000000;
inline constexpr uint32_t AC__MaxLength = 0xFFFFFFFF;

inline constexpr uint32_t BM__LengthShift = 13;
inline constexpr uint32_t BM__MaxCount    = 1 << BM__LengthShift; 

inline constexpr uint32_t DM__LengthShift = 15;
inline constexpr uint32_t DM__MaxCount    = 1 << DM__LengthShift;

struct ArithmeticModel{

	uint16_t* distribution;
	uint16_t* symbol_count;
	uint32_t total_count;
	uint32_t update_cycle;
	uint32_t symbols_until_update;
	uint32_t symbols;
	uint32_t last_symbol;

	ArithmeticModel(){
		// this->symbols = symbols;
		// distribution = nullptr;
	}

	~ArithmeticModel(){
		if (distribution) delete [] distribution;
	}

	int32_t init(uint32_t symbols, AllocatorGlobal* allocator){

		 // former constructor stuff
		this->symbols = symbols;
		distribution = nullptr;
		

		
		if ( (symbols < 2) || (symbols > (1 << 11))){
			printf("ERROR: invalid number of symbols \n");
			return -1; // invalid number of symbols
		}

		last_symbol = symbols - 1;

		// NOTE: removed decoder table stuff
		// distribution = allocator->alloc<uint32_t>(2 * symbols);
		distribution = allocator->alloc<uint16_t>(symbols);
		symbol_count = allocator->alloc<uint16_t>(symbols);

		if(enableTrace) printf("    allocating distribution uint32_t[%u] \n", 2 * symbols);
		
		// symbol_count = distribution + symbols;
		total_count = 0;
		update_cycle = symbols;

		for (uint32_t k = 0; k < symbols; k++){
			symbol_count[k] = 1;
		}

		update();

		symbols_until_update = update_cycle = (symbols + 6) >> 1;

		return 0;
	}

	void update(){

		uint64_t t_start = nanotime();

		if(enableTrace) printf("##################################  update model. point %i \n", dbg_pointIndex);

		// halve counts when a threshold is reached
		if ((total_count += update_cycle) > DM__MaxCount){
			total_count = 0;

			if(enableTrace) printf("    for %i \n", symbols);
			for (uint32_t n = 0; n < symbols; n++){
				symbol_count[n] = (symbol_count[n] + 1) >> 1;
				total_count += symbol_count[n];
			}
		}
		
		// compute cumulative distribution
		uint32_t k, sum = 0;
		uint32_t scale = 0x80000000U / total_count;

		if(enableTrace) printf("    for %i \n", symbols);
		for (k = 0; k < symbols; k++){
			distribution[k] = (scale * sum) >> (31 - DM__LengthShift);

			// if(distribution[k] > 34'000) printf("distribution[k] = %i \n", distribution[k]);

			sum += symbol_count[k];
		}

		// set frequency of model updates
		update_cycle = (5 * update_cycle) >> 2;
		uint32_t max_cycle = (symbols + 6) << 3;
		if (update_cycle > max_cycle){ 
			update_cycle = max_cycle;
		}
		symbols_until_update = update_cycle;

		uint64_t t_end = nanotime();
		if(t_end > t_start){
			t_update += t_end - t_start;
		}
	}

};

struct ArithmeticBitModel{
	uint32_t update_cycle;
	uint32_t bits_until_update;
	uint32_t bit_0_prob;
	uint32_t bit_0_count;
	uint32_t bit_count;

	ArithmeticBitModel(){
		init();
	}

	void init(){
		// initialization to equiprobable model
		bit_0_count = 1;
		bit_count   = 2;
		bit_0_prob  = 1U << (BM__LengthShift - 1);

		// start with frequent updates
		update_cycle = bits_until_update = 4;
	}

	void update(){
		// halve counts when a threshold is reached
		if ((bit_count += update_cycle) > BM__MaxCount){
			bit_count = (bit_count + 1) >> 1;
			bit_0_count = (bit_0_count + 1) >> 1;
			if (bit_0_count == bit_count) ++bit_count;
		}
		
		// compute scaled bit 0 probability
		uint32_t scale = 0x80000000U / bit_count;
		bit_0_prob = (bit_0_count * scale) >> (31 - BM__LengthShift);

		// set frequency of model updates
		update_cycle = (5 * update_cycle) >> 2;

		if (update_cycle > 64) update_cycle = 64;

		bits_until_update = update_cycle;
	}
};
