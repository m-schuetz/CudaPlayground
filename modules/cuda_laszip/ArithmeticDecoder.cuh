// Adapted from LASzip
// https://github.com/LASzip/LASzip/blob/master/src/arithmeticdecoder.hpp
// LICENSE: https://github.com/LASzip/LASzip/blob/80f929870f4ae2c3dc4e4f9e00e0e68f17ba1f1e/COPYING (Apache 2.0)

#pragma once

#include "ArithmeticModel.cuh"

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
		//this->value  = (getByte() << 24);
		//this->value |= (getByte() << 16);
		//this->value |= (getByte() << 8);
		//this->value |= (getByte());

		init(true);
	}

	void initSymbolModel(ArithmeticModel* model, uint32_t* table=0){
		model->init(table);
	}

	bool init(bool really_init){

		length = AC__MaxLength;

		if (really_init){
			// TODO: is this a simple memcpy or does it reverse byte order?
			value  = (getByte() << 24);
			value |= (getByte() << 16);
			value |= (getByte() << 8);
			value |= (getByte());
		}

		return true;
	}

	ArithmeticBitModel* createBitModel(){
		ArithmeticBitModel* m = new ArithmeticBitModel();
		return m;
	}

	void initBitModel(ArithmeticBitModel* m){
		m->init();
	}

	void destroyBitModel(ArithmeticBitModel* m){
		delete m;
	}

	ArithmeticModel* createSymbolModel(uint32_t n){
		ArithmeticModel* m = new ArithmeticModel(n, false);
		return m;
	}

	void destroySymbolModel(ArithmeticModel* m){
		delete m;
	}

	uint32_t decodeBit(ArithmeticBitModel* m){

		// product l x p0
		uint32_t x = m->bit_0_prob * (length >> BM__LengthShift);       

		// decision
		// update & shift interval
		uint32_t sym = (value >= x);
		
		
		if (sym == 0) {
			length  = x;
			++m->bit_0_count;
		}
		else {
			// shifted interval base = 0
			value  -= x;
			length -= x;
		}

		// renormalization
		if (length < AC__MinLength) renorm_dec_interval();

		// periodic model update
		if (--m->bits_until_update == 0) m->update();

		// return data bit value
		return sym;
	}

	uint32_t decodeSymbol(ArithmeticModel* m)
	{
		uint32_t n, sym, x, y = length;

		if (m->decoder_table) {             // use table look-up for faster decoding

			unsigned dv = value / (length >>= DM__LengthShift);
			unsigned t = dv >> m->table_shift;

			sym = m->decoder_table[t];      // initial decision based on table look-up
			n = m->decoder_table[t+1] + 1;

			while (n > sym + 1) {                      // finish with bisection search
				uint32_t k = (sym + n) >> 1;
				if (m->distribution[k] > dv) n = k; else sym = k;
			}
																// compute products
			x = m->distribution[sym] * length;
			if (sym != m->last_symbol) y = m->distribution[sym+1] * length;
		}

		else {                                  // decode using only multiplications

			x = sym = 0;
			length >>= DM__LengthShift;
			uint32_t k = (n = m->symbols) >> 1;
														// decode via bisection search
			do {
				uint32_t z = length * m->distribution[k];
				if (z > value) {
					n = k;
					y = z;                                             // value is smaller
				}
				else {
					sym = k;
					x = z;                                     // value is larger or equal
				}
			} while ((k = (sym + n) >> 1) != sym);
		}

		value -= x;                                               // update interval
		length = y - x;

		if (length < AC__MinLength) renorm_dec_interval();        // renormalization

		++m->symbol_count[sym];
		if (--m->symbols_until_update == 0) m->update();    // periodic model update

		//assert(sym < m->symbols);
		if (!(sym < m->symbols)) {
			// TODO handle potential errors
			// exit(123);
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
		//assert(bits && (bits <= 32));
		if (!(bits && (bits <= 32))) {
			// TODO: handle potential error
			// exit(123);
		}

		if (bits > 19)
		{
			uint32_t tmp = readShort();
			bits = bits - 16;
			uint32_t tmp1 = readBits(bits) << 16;
			return (tmp1|tmp);
		}

		uint32_t sym = value / (length >>= bits);// decode symbol, change length
		value -= length * sym;                                    // update interval

		if (length < AC__MinLength) renorm_dec_interval();        // renormalization

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

		if (sym >= (1u<<8))
		{
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

		if (sym >= (1u<<16))
		{
			// TODO: handle potential errors
			// throw 4711;
		}

		return (uint16_t)sym;
	}

	inline void renorm_dec_interval()
	{
		do {                                          // read least-significant byte
			value = (value << 8) | getByte();
		} while ((length <<= 8) < AC__MinLength);        // length multiplied by 256
	}

	uint32_t getByte(){
		uint32_t value = this->buffer[this->bufferOffset];
		this->bufferOffset++;

		return value;
	}

};