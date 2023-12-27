// Adapted from LASzip
// https://github.com/LASzip/LASzip/blob/master/src/arithmeticdecoder.hpp
// LICENSE: https://github.com/LASzip/LASzip/blob/80f929870f4ae2c3dc4e4f9e00e0e68f17ba1f1e/COPYING (Apache 2.0)

#pragma once

#include "ArithmeticDecoder.cuh"
#include "ArithmeticModel.cuh"

constexpr int I32_MIN = -2147483648;
constexpr int I32_MAX =  2147483647;

struct IntegerCompressor{

	uint32_t k;

	uint32_t contexts;
	uint32_t bits_high;

	uint32_t bits;
	uint32_t range;

	uint32_t corr_bits;
	uint32_t corr_range;
	int32_t corr_min;
	int32_t corr_max;

	ArithmeticDecoder* dec;
	ArithmeticModel** mBits;
	ArithmeticModel** mCorrector;

	IntegerCompressor(ArithmeticDecoder* dec, uint32_t bits = 16, uint32_t contexts = 1, uint32_t bits_high = 8, uint32_t range = 0){
		this->dec = dec;
		this->bits = bits;
		this->contexts = contexts;
		this->bits_high = bits_high;
		this->range = range;

		if (range) // the corrector's significant bits and range
		{
			corr_bits = 0;
			corr_range = range;
			while (range)
			{
			range = range >> 1;
			corr_bits++;
			}
			if (corr_range == (1u << (corr_bits-1)))
			{
			corr_bits--;
			}
				// the corrector must fall into this interval
			corr_min = -((int32_t)(corr_range/2));
			corr_max = corr_min + corr_range - 1;
		}
		else if (bits && bits < 32)
		{
			corr_bits = bits;
			corr_range = 1u << bits;
				// the corrector must fall into this interval
			corr_min = -((int32_t)(corr_range/2));
			corr_max = corr_min + corr_range - 1;
		}
			else
			{
			corr_bits = 32;
				corr_range = 0;
				// the corrector must fall into this interval
			corr_min = I32_MIN;
			corr_max = I32_MAX;
			}

		k = 0;

		mBits = 0;
		mCorrector = 0;
	}

	// void IntegerCompressor::initCompressor()
	// {
	// 	uint32_t i;

	// 	// maybe create the models
	// 	if (mBits == 0)
	// 	{
	// 		mBits = new ArithmeticModel*[contexts];
	// 		for (i = 0; i < contexts; i++)
	// 		{
	// 		mBits[i] = enc->createSymbolModel(corr_bits+1);
	// 		}
	// 	#ifndef COMPRESS_ONLY_K
	// 		mCorrector = new ArithmeticModel*[corr_bits+1];
	// 		mCorrector[0] = (ArithmeticModel*)enc->createBitModel();
	// 		for (i = 1; i <= corr_bits; i++)
	// 		{
	// 		if (i <= bits_high)
	// 		{
	// 			mCorrector[i] = enc->createSymbolModel(1<<i);
	// 		}
	// 		else
	// 		{
	// 			mCorrector[i] = enc->createSymbolModel(1<<bits_high);
	// 		}
	// 		}
	// 	#endif
	// 	}

	// 	// certainly init the models
	// 	for (i = 0; i < contexts; i++)
	// 	{
	// 		enc->initSymbolModel(mBits[i]);
	// 	}
	// 	#ifndef COMPRESS_ONLY_K
	// 	enc->initBitModel((ArithmeticBitModel*)mCorrector[0]);
	// 	for (i = 1; i <= corr_bits; i++)
	// 	{
	// 		enc->initSymbolModel(mCorrector[i]);
	// 	}
	// 	#endif
	// }

	void initDecompressor(){
		uint32_t i;

		// maybe create the models
		if (mBits == 0){
			mBits = new ArithmeticModel*[contexts];
			for (i = 0; i < contexts; i++){
				mBits[i] = dec->createSymbolModel(corr_bits+1);
			}

			mCorrector = new ArithmeticModel*[corr_bits+1];
			mCorrector[0] = (ArithmeticModel*)dec->createBitModel();
			for (i = 1; i <= corr_bits; i++)
			{
			if (i <= bits_high)
			{
				mCorrector[i] = dec->createSymbolModel(1<<i);
			}
			else
			{
				mCorrector[i] = dec->createSymbolModel(1<<bits_high);
			}
			}
		}

		// certainly init the models
		for (i = 0; i < contexts; i++){
			dec->initSymbolModel(mBits[i]);
		}

		dec->initBitModel((ArithmeticBitModel*)mCorrector[0]);
		for (i = 1; i <= corr_bits; i++)
		{
			dec->initSymbolModel(mCorrector[i]);
		}
	}


	int32_t decompress(int32_t pred, uint32_t context = 0){
		int32_t real = pred + readCorrector(mBits[context]);
		if (real < 0) real += corr_range;
		else if ((uint32_t)(real) >= corr_range) real -= corr_range;
		return real;
	}

	int32_t readCorrector(ArithmeticModel* mBits){
		int32_t c;

		// decode within which interval the corrector is falling

		k = dec->decodeSymbol(mBits);

		// decode the exact location of the corrector within the interval

		if (k) // then c is either smaller than 0 or bigger than 1
		{
			if (k < 32)
			{
			if (k <= bits_high) // for small k we can do this in one step
			{
				// decompress c with the range coder
				c = dec->decodeSymbol(mCorrector[k]);
			}
			else
			{
				// for larger k we need to do this in two steps
				int k1 = k-bits_high;
				// decompress higher bits with table
				c = dec->decodeSymbol(mCorrector[k]);
				// read lower bits raw
				int c1 = dec->readBits(k1);
				// put the corrector back together
				c = (c << k1) | c1;
			}
			// translate c back into its correct interval
			if (c >= (1<<(k-1))) // if c is in the interval [ 2^(k-1)  ...  + 2^k - 1 ]
			{
				// so we translate c back into the interval [ 2^(k-1) + 1  ...  2^k ] by adding 1 
				c += 1;
			}
			else // otherwise c is in the interval [ 0 ...  + 2^(k-1) - 1 ]
			{
				// so we translate c back into the interval [ - (2^k - 1)  ...  - (2^(k-1)) ] by subtracting (2^k - 1)
				c -= ((1<<k) - 1);
			}
			}
			else
			{
			c = corr_min;
			}
		}
		else // then c is either 0 or 1
		{
			c = dec->decodeBit((ArithmeticBitModel*)mCorrector[0]);
		}

		return c;
	}

	// Get the k corrector bits from the last compress/decompress call
	uint32_t getK() const {
		return k;
	}

};

