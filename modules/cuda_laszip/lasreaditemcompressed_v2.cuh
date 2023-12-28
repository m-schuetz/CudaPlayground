
// Adapted from LASzip
// https://github.com/LASzip/LASzip/blob/master/src/lasreaditemcompressed_v2.hpp
// LICENSE: https://github.com/LASzip/LASzip/blob/80f929870f4ae2c3dc4e4f9e00e0e68f17ba1f1e/COPYING (Apache 2.0)

#pragma once

#include "ArithmeticDecoder.cuh"
#include "IntegerCompressor.cuh"
#include "StreamingMedian5.cuh"
#include "laszip_mydefs.cuh"

struct LASpoint10{
	int32_t x;
	int32_t y;
	int32_t z;
	uint16_t intensity;
	uint8_t return_number : 3;
	uint8_t number_of_returns_of_given_pulse : 3;
	uint8_t scan_direction_flag : 1;
	uint8_t edge_of_flight_line : 1;
	uint8_t classification;
	int8_t scan_angle_rank;
	uint8_t user_data;
	uint16_t point_source_ID;
};

struct LASreadItemCompressed_POINT10_v2{

	LASreadItemCompressed_POINT10_v2(ArithmeticDecoder* dec){
		// uint32_t i;

		// /* set decoder */
		// this->dec = dec;

		// /* create models and integer compressors */
		// m_changed_values = new ArithmeticModel(64, false);
		// ic_intensity = new IntegerCompressor(dec, 16, 4);
		// m_scan_angle_rank[0] = new ArithmeticModel(256, false);
		// m_scan_angle_rank[1] = new ArithmeticModel(256, false);
		// ic_point_source_ID = new IntegerCompressor(dec, 16);
		// for (i = 0; i < 256; i++){
		// 	m_bit_byte[i] = 0;
		// 	m_classification[i] = 0;
		// 	m_user_data[i] = 0;
		// }
		// ic_dx = new IntegerCompressor(dec, 32, 2);  // 32 bits, 2 context
		// ic_dy = new IntegerCompressor(dec, 32, 22); // 32 bits, 22 contexts
		// ic_z = new IntegerCompressor(dec, 32, 20);  // 32 bits, 20 contexts
	}

	~LASreadItemCompressed_POINT10_v2(){
		uint32_t i;

		delete m_changed_values;
		delete ic_intensity;
		delete m_scan_angle_rank[0];
		delete m_scan_angle_rank[1];
		delete ic_point_source_ID;
		for (i = 0; i < 256; i++)
		{
			if (m_bit_byte[i]) delete m_bit_byte[i];
			if (m_classification[i]) delete m_classification[i];
			if (m_user_data[i]) delete m_user_data[i];
		}
		delete ic_dx;
		delete ic_dy;
		delete ic_z;
	}

	bool init(ArithmeticDecoder* dec, 
		const uint8_t* item, 
		uint32_t& context, 
		AllocatorGlobal* allocator
	){

		{ // former constructor stuff
			uint32_t i;

			/* set decoder */
			this->dec = dec;

			/* create models and integer compressors */
			m_changed_values = new ArithmeticModel(64, false);
			ic_intensity = new IntegerCompressor(dec, 16, 4);
			m_scan_angle_rank[0] = new ArithmeticModel(256, false);
			m_scan_angle_rank[1] = new ArithmeticModel(256, false);
			ic_point_source_ID = new IntegerCompressor(dec, 16);
			for (i = 0; i < 256; i++){
				m_bit_byte[i] = 0;
				m_classification[i] = 0;
				m_user_data[i] = 0;
			}
			ic_dx = new IntegerCompressor(dec, 32, 2);  // 32 bits, 2 context
			ic_dy = new IntegerCompressor(dec, 32, 22); // 32 bits, 22 contexts
			ic_z = new IntegerCompressor(dec, 32, 20);  // 32 bits, 20 contexts
		}



		uint32_t i;

		/* init state */
		for (i=0; i < 16; i++)
		{
			last_x_diff_median5[i].init();
			last_y_diff_median5[i].init();
			last_intensity[i] = 0;
			last_height[i/2] = 0;
		}

		/* init models and integer compressors */
		m_changed_values->init(nullptr, allocator);

		ic_intensity->initDecompressor(allocator);

		m_scan_angle_rank[0]->init(nullptr, allocator);
		m_scan_angle_rank[1]->init(nullptr, allocator);
		// dec->initSymbolModel(m_scan_angle_rank[0]);
		// dec->initSymbolModel(m_scan_angle_rank[1]);

		ic_point_source_ID->initDecompressor(allocator);
		for (i = 0; i < 256; i++) {
			if (m_bit_byte[i])        m_bit_byte[i]->init(nullptr, allocator);
			if (m_classification[i])  m_classification[i]->init(nullptr, allocator);
			if (m_user_data[i])       m_user_data[i]->init(nullptr, allocator);
			// if (m_bit_byte[i]) dec->initSymbolModel(m_bit_byte[i]);
			// if (m_classification[i]) dec->initSymbolModel(m_classification[i]);
			// if (m_user_data[i]) dec->initSymbolModel(m_user_data[i]);
		}
		ic_dx->initDecompressor(allocator);
		ic_dy->initDecompressor(allocator);
		ic_z->initDecompressor(allocator);

		/* init last item */
		memcpy(last_item, item, 20);

		/* but set intensity to zero */ 
		last_item[12] = 0;
		last_item[13] = 0;

		return true;
	}

	void read(uint8_t* item, uint32_t& context, AllocatorGlobal* allocator){
		uint32_t r, n, m, l;
		uint32_t k_bits;
		int32_t median, diff;

		// decompress which other values have changed
		int32_t changed_values = dec->decodeSymbol(m_changed_values);

		if (changed_values) {

			// decompress the edge_of_flight_line, scan_direction_flag, ... if it has changed
			if (changed_values & 32) {
				if (m_bit_byte[last_item[14]] == 0){
					m_bit_byte[last_item[14]] = new ArithmeticModel(256, false);
					m_bit_byte[last_item[14]]->init(nullptr, allocator);
				}

				last_item[14] = (uint8_t)dec->decodeSymbol(m_bit_byte[last_item[14]]);
			}

			r = ((LASpoint10*)last_item)->return_number;
			n = ((LASpoint10*)last_item)->number_of_returns_of_given_pulse;
			m = number_return_map[n][r];
			l = number_return_level[n][r];

			// decompress the intensity if it has changed
			if (changed_values & 16){
				((LASpoint10*)last_item)->intensity = (uint16_t)ic_intensity->decompress(last_intensity[m], (m < 3 ? m : 3));
				last_intensity[m] = ((LASpoint10*)last_item)->intensity;
			}else{
				((LASpoint10*)last_item)->intensity = last_intensity[m];
			}

			// decompress the classification ... if it has changed
			if (changed_values & 8){
				if (m_classification[last_item[15]] == 0){
					m_classification[last_item[15]] = new ArithmeticModel(256, false);
					m_classification[last_item[15]]->init(nullptr, allocator);
				}

				last_item[15] = (uint8_t)dec->decodeSymbol(m_classification[last_item[15]]);
			}
			
			// decompress the scan_angle_rank ... if it has changed
			if (changed_values & 4){
				int32_t val = dec->decodeSymbol(m_scan_angle_rank[((LASpoint10*)last_item)->scan_direction_flag]);
				last_item[16] = U8_FOLD(val + last_item[16]);
			}

			// decompress the user_data ... if it has changed
			if (changed_values & 2)
			{
				if (m_user_data[last_item[17]] == 0){
					m_user_data[last_item[17]] = new ArithmeticModel(256, false);
					m_user_data[last_item[17]]->init(nullptr, allocator);
				}
				last_item[17] = (uint8_t)dec->decodeSymbol(m_user_data[last_item[17]]);
			}

			// decompress the point_source_ID ... if it has changed
			if (changed_values & 1){
				((LASpoint10*)last_item)->point_source_ID = (uint16_t)ic_point_source_ID->decompress(((LASpoint10*)last_item)->point_source_ID);
			}
		}else{
			r = ((LASpoint10*)last_item)->return_number;
			n = ((LASpoint10*)last_item)->number_of_returns_of_given_pulse;
			m = number_return_map[n][r];
			l = number_return_level[n][r];
		}

		// decompress x coordinate
		median = last_x_diff_median5[m].get();
		diff = ic_dx->decompress(median, n==1);
		((LASpoint10*)last_item)->x += diff;
		last_x_diff_median5[m].add(diff);

		// decompress y coordinate
		median = last_y_diff_median5[m].get();
		k_bits = ic_dx->getK();
		diff = ic_dy->decompress(median, (n==1) + ( k_bits < 20 ? U32_ZERO_BIT_0(k_bits) : 20 ));
		((LASpoint10*)last_item)->y += diff;
		last_y_diff_median5[m].add(diff);

		// decompress z coordinate
		k_bits = (ic_dx->getK() + ic_dy->getK()) / 2;
		((LASpoint10*)last_item)->z = ic_z->decompress(last_height[l], (n==1) + (k_bits < 18 ? U32_ZERO_BIT_0(k_bits) : 18));
		last_height[l] = ((LASpoint10*)last_item)->z;

		// copy the last point
		memcpy(item, last_item, 20);
	}

	//~LASreadItemCompressed_POINT10_v2();

	private:
	ArithmeticDecoder* dec;
	uint8_t last_item[20];
	uint16_t last_intensity[16];
	StreamingMedian5 last_x_diff_median5[16];
	StreamingMedian5 last_y_diff_median5[16];
	int32_t last_height[8];

	ArithmeticModel* m_changed_values;
	IntegerCompressor* ic_intensity;
	ArithmeticModel* m_scan_angle_rank[2];
	IntegerCompressor* ic_point_source_ID;
	ArithmeticModel* m_bit_byte[256];
	ArithmeticModel* m_classification[256];
	ArithmeticModel* m_user_data[256];
	IntegerCompressor* ic_dx;
	IntegerCompressor* ic_dy;
	IntegerCompressor* ic_z;
};

struct LASreadItemCompressed_RGB12_v2{
	ArithmeticDecoder* dec;
	uint16_t last_item[3];

	ArithmeticModel* m_byte_used;
	ArithmeticModel* m_rgb_diff_0;
	ArithmeticModel* m_rgb_diff_1;
	ArithmeticModel* m_rgb_diff_2;
	ArithmeticModel* m_rgb_diff_3;
	ArithmeticModel* m_rgb_diff_4;
	ArithmeticModel* m_rgb_diff_5;

	LASreadItemCompressed_RGB12_v2(ArithmeticDecoder* dec){
		// this->dec = dec;

		// /* create models and integer compressors */
		// m_byte_used  = new ArithmeticModel(128, false);
		// m_rgb_diff_0 = new ArithmeticModel(256, false);
		// m_rgb_diff_1 = new ArithmeticModel(256, false);
		// m_rgb_diff_2 = new ArithmeticModel(256, false);
		// m_rgb_diff_3 = new ArithmeticModel(256, false);
		// m_rgb_diff_4 = new ArithmeticModel(256, false);
		// m_rgb_diff_5 = new ArithmeticModel(256, false);
	}

	bool init(ArithmeticDecoder* dec, const uint8_t* item, uint32_t& context, AllocatorGlobal* allocator){

		{ // former constructor stuff
			this->dec = dec;

			/* create models and integer compressors */
			m_byte_used  = new ArithmeticModel(128, false);
			m_rgb_diff_0 = new ArithmeticModel(256, false);
			m_rgb_diff_1 = new ArithmeticModel(256, false);
			m_rgb_diff_2 = new ArithmeticModel(256, false);
			m_rgb_diff_3 = new ArithmeticModel(256, false);
			m_rgb_diff_4 = new ArithmeticModel(256, false);
			m_rgb_diff_5 = new ArithmeticModel(256, false);

			// m_byte_used  = allocator->alloc<ArithmeticModel>(1);
			// m_rgb_diff_0 = allocator->alloc<ArithmeticModel>(1);
			// m_rgb_diff_1 = allocator->alloc<ArithmeticModel>(1);
			// m_rgb_diff_2 = allocator->alloc<ArithmeticModel>(1);
			// m_rgb_diff_3 = allocator->alloc<ArithmeticModel>(1);
			// m_rgb_diff_4 = allocator->alloc<ArithmeticModel>(1);
			// m_rgb_diff_5 = allocator->alloc<ArithmeticModel>(1);
		}

		/* init state */

		/* init models and integer compressors */
		m_byte_used->init(nullptr, allocator);
		m_rgb_diff_0->init(nullptr, allocator);
		m_rgb_diff_1->init(nullptr, allocator);
		m_rgb_diff_2->init(nullptr, allocator);
		m_rgb_diff_3->init(nullptr, allocator);
		m_rgb_diff_4->init(nullptr, allocator);
		m_rgb_diff_5->init(nullptr, allocator);

		/* init last item */
		memcpy(last_item, item, 6);

		return true;
	}

	inline void read(uint8_t* item, uint32_t& context)
	{
		uint8_t corr;
		int32_t diff = 0;
		uint32_t sym = dec->decodeSymbol(m_byte_used);
		if (sym & (1 << 0))
		{
			corr = dec->decodeSymbol(m_rgb_diff_0);
			((uint16_t*)item)[0] = (uint16_t)U8_FOLD(corr + (last_item[0]&255));
		}
		else 
		{
			((uint16_t*)item)[0] = last_item[0]&0xFF;
		}
		if (sym & (1 << 1))
		{
			corr = dec->decodeSymbol(m_rgb_diff_1);
			((uint16_t*)item)[0] |= (((uint16_t)U8_FOLD(corr + (last_item[0]>>8))) << 8);
		}
		else
		{
			((uint16_t*)item)[0] |= (last_item[0]&0xFF00);
		}
		if (sym & (1 << 6))
		{
			diff = (((uint16_t*)item)[0]&0x00FF) - (last_item[0]&0x00FF);
			if (sym & (1 << 2))
			{
			corr = dec->decodeSymbol(m_rgb_diff_2);
			((uint16_t*)item)[1] = (uint16_t)U8_FOLD(corr + U8_CLAMP(diff+(last_item[1]&255)));
			}
			else
			{
			((uint16_t*)item)[1] = last_item[1]&0xFF;
			}
			if (sym & (1 << 4))
			{
			corr = dec->decodeSymbol(m_rgb_diff_4);
			diff = (diff + ((((uint16_t*)item)[1]&0x00FF) - (last_item[1]&0x00FF))) / 2;
			((uint16_t*)item)[2] = (uint16_t)U8_FOLD(corr + U8_CLAMP(diff+(last_item[2]&255)));
			}
			else
			{
			((uint16_t*)item)[2] = last_item[2]&0xFF;
			}
			diff = (((uint16_t*)item)[0]>>8) - (last_item[0]>>8);
			if (sym & (1 << 3))
			{
			corr = dec->decodeSymbol(m_rgb_diff_3);
			((uint16_t*)item)[1] |= (((uint16_t)U8_FOLD(corr + U8_CLAMP(diff+(last_item[1]>>8))))<<8);
			}
			else
			{
			((uint16_t*)item)[1] |= (last_item[1]&0xFF00);
			}
			if (sym & (1 << 5))
			{
			corr = dec->decodeSymbol(m_rgb_diff_5);
			diff = (diff + ((((uint16_t*)item)[1]>>8) - (last_item[1]>>8))) / 2;
			((uint16_t*)item)[2] |= (((uint16_t)U8_FOLD(corr + U8_CLAMP(diff+(last_item[2]>>8))))<<8);
			}
			else
			{
			((uint16_t*)item)[2] |= (last_item[2]&0xFF00);
			}
		}
		else
		{
			((uint16_t*)item)[1] = ((uint16_t*)item)[0];
			((uint16_t*)item)[2] = ((uint16_t*)item)[0];
		}

		memcpy(last_item, item, 6);
	}

};