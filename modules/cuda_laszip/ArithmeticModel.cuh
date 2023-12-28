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

	uint32_t* distribution;
	uint32_t* symbol_count;
	uint32_t* decoder_table;
	uint32_t total_count;
	uint32_t update_cycle;
	uint32_t symbols_until_update;
	uint32_t symbols;
	uint32_t last_symbol;
	uint32_t table_size;
	uint32_t table_shift;
	bool compress;

	ArithmeticModel(uint32_t symbols, bool compress){
		this->symbols = symbols;
		this->compress = compress;
		distribution = nullptr;
	}

	~ArithmeticModel(){
		if (distribution) delete [] distribution;
	}

	int32_t init(uint32_t* table, AllocatorGlobal* allocator){

		if (distribution == 0){

			if ( (symbols < 2) || (symbols > (1 << 11))){
				return -1; // invalid number of symbols
			}

			last_symbol = symbols - 1;

			if ((!compress) && (symbols > 16)){
				uint32_t table_bits = 3;

				while (symbols > (1U << (table_bits + 2))){
					 ++table_bits;
				}

				table_size  = 1 << table_bits;
				table_shift = DM__LengthShift - table_bits;

				distribution = allocator->alloc<uint32_t>(2 * symbols + table_size + 2);
				decoder_table = distribution + 2 * symbols;
			}else {
				// small alphabet: no table needed
				decoder_table = 0;
				table_size = table_shift = 0;
				distribution = allocator->alloc<uint32_t>(2 * symbols);
			}
			
			if (distribution == 0){
				return -1; // "cannot allocate model memory");
			}
			symbol_count = distribution + symbols;
		}

		total_count = 0;
		update_cycle = symbols;

		if (table){
			for (uint32_t k = 0; k < symbols; k++) symbol_count[k] = table[k];
		}else{
			for (uint32_t k = 0; k < symbols; k++) symbol_count[k] = 1;
		}

		update();

		symbols_until_update = update_cycle = (symbols + 6) >> 1;

		return 0;
	}

	void update(){
		// halve counts when a threshold is reached
		if ((total_count += update_cycle) > DM__MaxCount){
			total_count = 0;
			for (uint32_t n = 0; n < symbols; n++){
				total_count += (symbol_count[n] = (symbol_count[n] + 1) >> 1);
			}
		}
		
		// compute cumulative distribution, decoder table
		uint32_t k, sum = 0, s = 0;
		uint32_t scale = 0x80000000U / total_count;

		if (compress || (table_size == 0)){
			for (k = 0; k < symbols; k++){
				distribution[k] = (scale * sum) >> (31 - DM__LengthShift);
				sum += symbol_count[k];
			}
		}else{
			for (k = 0; k < symbols; k++){
				distribution[k] = (scale * sum) >> (31 - DM__LengthShift);
				sum += symbol_count[k];
				uint32_t w = distribution[k] >> table_shift;
				while (s < w) decoder_table[++s] = k - 1;
			}
			decoder_table[0] = 0;
			while (s <= table_size) decoder_table[++s] = symbols - 1;
		}
		
		// set frequency of model updates
		update_cycle = (5 * update_cycle) >> 2;
		uint32_t max_cycle = (symbols + 6) << 3;
		if (update_cycle > max_cycle) update_cycle = max_cycle;
		symbols_until_update = update_cycle;
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
