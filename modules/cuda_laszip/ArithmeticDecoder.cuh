// Adapted from LASzip
// https://github.com/LASzip/LASzip/blob/master/src/arithmeticdecoder.hpp
// LICENSE: https://github.com/LASzip/LASzip/blob/80f929870f4ae2c3dc4e4f9e00e0e68f17ba1f1e/COPYING (Apache 2.0)

#pragma once

#include "ArithmeticModel.cuh"
#include "utils_hd.cuh"

struct ArithmeticDecoder{

	uint8_t* buffer;
	int64_t bufferOffset;
	uint32_t length;
	uint32_t value;


	ArithmeticDecoder(uint8_t* buffer, int64_t offset){
		this->buffer = buffer;
		this->bufferOffset = offset;
		this->length = AC__MaxLength;

		this->value = 0;

		init(true);
	}

	bool init(bool really_init){
		length = AC__MaxLength;

		value  = (getByte() << 24);
		value |= (getByte() << 16);
		value |= (getByte() << 8);
		value |= (getByte());

		return true;
	}
	uint32_t decodeBit(ArithmeticBitModel* m){

		if(enableTrace) printf("decodeBit() \n");

		// product l x p0
		uint32_t x = m->bit_0_prob * (length >> BM__LengthShift);

		// decision
		// update & shift interval
		uint32_t sym = (value >= x);
		
		if (sym == 0) {
			length  = x;
			++m->bit_0_count;
		} else {
			// shifted interval base = 0
			value  -= x;
			length -= x;
		}

		// renormalization
		if (length < AC__MinLength) renorm_dec_interval();

		// periodic model update
		m->bits_until_update--;
		if (m->bits_until_update == 0){
			m->update();
		}

		// return data bit value
		return sym;
	}

	// LASZIP source: https://github.com/LASzip/LASzip/blob/80f929870f4ae2c3dc4e4f9e00e0e68f17ba1f1e/src/arithmeticdecoder.cpp#L182
	// FastAC source: https://github.com/richgel999/FastAC/blob/f39296ce6abbdaca8f86fef9248285c88e262797/arithmetic_codec.cpp#L305
	uint32_t decodeSymbol(ArithmeticModel* m){

		uint64_t t_start = nanotime();

		if(enableTrace) printf("decodeSymbol \n");

		uint32_t x   = 0;
		uint32_t sym = 0;
		uint32_t n   = m->symbols;
		uint32_t y   = length;
		

		// NOTE: removed decoder table path
		{// decode using only multiplications
			uint64_t t_start_decodeSymbols_1 = nanotime();

			// if(enableTrace) printf("    decode using only multiplications \n");

			length     = length >> DM__LengthShift;
			uint32_t k = n >> 1;
			
			// decode via bisection search
			int counter = 0;

			if(enableTrace) printf("    do... while (k != sym); k = %i, sym = %i \n", k, sym);
			do {
				if(enableTrace) printf("\n        # iteration", k);
				uint32_t z = length * m->distribution[k];
				if(enableTrace) printf("        x     = %u \n", x);
				if(enableTrace) printf("        sym   = %u \n", sym);
				if(enableTrace) printf("        k     = %u \n", k);
				if(enableTrace) printf("        z     = %u \n", z);
				if(enableTrace) printf("        n     = %u \n", n);
				if(enableTrace) printf("        d[k]  = %u \n", m->distribution[k]);
				if(enableTrace) printf("        value = %u \n", value);

				if (z > value) {
					// value is smaller
					n = k;
					y = z;
					if(enableTrace) printf("        value is smaller. n = %u, y = %u \n", k, z);
				} else {
					// value is larger or equal
					sym = k;
					x = z;
					if(enableTrace) printf("        value is larger. sym = %u, x = %u \n", k, z);
				}
				counter++;

				k = (sym + n) >> 1;
			} while (k != sym);
			if(enableTrace) printf("        loop done, k == sym == %u \n", k);

			// if(counter > 10) printf("            counter_2 is large: %i \n", counter);
			// if(enableTrace) printf("    counter_2: %i \n", counter);

			uint64_t t_end_decodeSymbols_1 = nanotime();
			if(t_end_decodeSymbols_1 > t_start_decodeSymbols_1){
				t_decodeSymbols_1 += (t_end_decodeSymbols_1 - t_start_decodeSymbols_1);
			}
		}

		// update interval
		value -= x;
		length = y - x;

		// renormalization
		if (length < AC__MinLength){ 
			renorm_dec_interval();
		}

		m->symbol_count[sym]++;
		m->symbols_until_update--;

		// periodic model update
		if (m->symbols_until_update == 0){
			m->update();
		}

		uint64_t t_end = nanotime();
		if(t_end > t_start){
			t_decodeSymbol += (t_end - t_start);
		}

		return sym;
	}

	uint32_t readBit(){
		uint32_t sym = value / (length >>= 1);            // decode symbol, change length
		value -= length * sym;                                    // update interval

		if (length < AC__MinLength) renorm_dec_interval();        // renormalization

		if (sym >= 2){
			// TODO: handle potential error
			// throw 4711;
		}

		return sym;
	}

	uint32_t readBits(uint32_t bits){

		if(enableTrace) printf("readBits(%i) \n", bits);

		//assert(bits && (bits <= 32));
		if (!(bits && (bits <= 32))) {
			// TODO: handle potential error
			// exit(123);
		}

		if (bits > 19){
			uint32_t tmp = readShort();
			bits = bits - 16;
			uint32_t tmp1 = readBits(bits) << 16;
			return (tmp1|tmp);
		}

		uint32_t sym = value / (length >>= bits);// decode symbol, change length
		value -= length * sym;                                    // update interval

		if (length < AC__MinLength){
			// renormalization
			renorm_dec_interval();
		}

		if (sym >= (1u<<bits)){
			// TODO: handle potential error
			// throw 4711;
		}

		return sym;
	}

	uint8_t readByte()
	{
		uint32_t sym = value / (length >>= 8);            // decode symbol, change length
		value -= length * sym;                                    // update interval

		if (length < AC__MinLength) renorm_dec_interval();        // renormalization

		if (sym >= (1u<<8)){
			// TODO: handle potential errors
			// throw 4711;
		}

		return (uint8_t)sym;
	}

	uint16_t readShort()
	{
		uint32_t sym = value / (length >>= 16);           // decode symbol, change length
		value -= length * sym;                                    // update interval

		if (length < AC__MinLength) renorm_dec_interval();        // renormalization

		if (sym >= (1u<<16)){
			// TODO: handle potential errors
			// throw 4711;
		}

		return (uint16_t)sym;
	}

	uint32_t readInt(){
		uint32_t lowerInt = readShort();
		uint32_t upperInt = readShort();
		
		return (upperInt<<16) | lowerInt;
	}

	inline void renorm_dec_interval(){

		uint64_t t_start = nanotime();

		// if(enableTrace) printf("renorm_dec_interval \n");



		// read least-significant byte
		int counter = 0;
		do {
			value = (value << 8) | getByte();
			counter++;
			
			// length multiplied by 256
			length = length << 8;
		} while (length < AC__MinLength);

		// alternative without loop. Not faster, though
		// value = (value << 8) | getByte();
		// length = length << 8;
		// if(length < AC__MinLength){
		// 	value = (value << 8) | getByte();
		// 	length = length << 8;
		// }
		// if(length < AC__MinLength){
		// 	value = (value << 8) | getByte();
		// 	length = length << 8;
		// }


		
		// if(enableTrace) printf("    counter: %i \n", counter);
		// if(counter > 1) printf("    !!!!!!! large counter: %i \n", counter);
		
		// if(enableTrace) printf("    counter: %i \n", counter);

		uint64_t t_end = nanotime();
		if(t_end > t_start){
			t_renorm += t_end - t_start;
		}
	}

	uint32_t getByte(){
		uint32_t value = this->buffer[this->bufferOffset];
		this->bufferOffset++;

		return value;
	}

};