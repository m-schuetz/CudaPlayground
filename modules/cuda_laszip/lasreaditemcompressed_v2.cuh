
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
			ic_intensity         = allocator->alloc<IntegerCompressor>(1);
			ic_point_source_ID   = allocator->alloc<IntegerCompressor>(1);

			m_changed_values     = allocator->alloc<ArithmeticModel>(1);
			m_scan_angle_rank[0] = allocator->alloc<ArithmeticModel>(1);
			m_scan_angle_rank[1] = allocator->alloc<ArithmeticModel>(1);

			for (i = 0; i < 256; i++){
				m_bit_byte[i]       = nullptr;
				m_classification[i] = nullptr;
				m_user_data[i]      = nullptr;
			}

			ic_dx = allocator->alloc<IntegerCompressor>(1); // 32 bits, 2 context
			ic_dy = allocator->alloc<IntegerCompressor>(1); // 32 bits, 22 contexts
			ic_z  = allocator->alloc<IntegerCompressor>(1); // 32 bits, 20 contexts
		}



		uint32_t i;

		/* init state */
		for (i=0; i < 16; i++){
			last_x_diff_median5[i].init();
			last_y_diff_median5[i].init();
			last_intensity[i] = 0;
			last_height[i/2] = 0;
		}

		/* init models and integer compressors */
		m_changed_values->init(64, allocator);

		ic_intensity->init(dec, 16, 4);
		ic_point_source_ID->init(dec, 16);
		ic_intensity->initDecompressor(allocator);
		ic_point_source_ID->initDecompressor(allocator);

		m_scan_angle_rank[0]->init(256, allocator);
		m_scan_angle_rank[1]->init(256, allocator);
		// dec->initSymbolModel(m_scan_angle_rank[0]);
		// dec->initSymbolModel(m_scan_angle_rank[1]);


			// probably never actually happens because these attributes are 
			// created on-demand during reads?
		// for (i = 0; i < 256; i++) {
		// 	if (m_bit_byte[i])        m_bit_byte[i]->init(nullptr, allocator);
		// 	if (m_classification[i])  m_classification[i]->init(nullptr, allocator);
		// 	if (m_user_data[i])       m_user_data[i]->init(nullptr, allocator);
		// }

		ic_dx->init(dec, 32, 2);
		ic_dy->init(dec, 32, 22);
		ic_z->init(dec, 32, 20);
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

		auto grid = cg::this_grid();
		// auto block = cg::this_thread_block();
		cg::coalesced_group active = cg::coalesced_threads();


		uint64_t t_start = nanotime();

		uint32_t r, n, m, l;
		uint32_t k_bits;
		int32_t median, diff;

		// decompress which other values have changed
		// int32_t changed_values = dec->decodeSymbol(m_changed_values);
		uint32_t changed_values;
		dec->decodeSymbol_warp(m_changed_values, &changed_values);

		// if(active.thread_rank() != 0) return;

		if(enableTrace) printf("changed_values: %i \n", changed_values);


		if (changed_values && active.thread_rank() == 0) {

			// decompress the edge_of_flight_line, scan_direction_flag, ... if it has changed
			if (changed_values & 32) {
				if (m_bit_byte[last_item[14]] == 0){
					m_bit_byte[last_item[14]] = allocator->alloc<ArithmeticModel>(1);
					m_bit_byte[last_item[14]]->init(256, allocator);
				}

				last_item[14] = (uint8_t)dec->decodeSymbol(m_bit_byte[last_item[14]]);
			}

			r = ((LASpoint10*)last_item)->return_number;
			n = ((LASpoint10*)last_item)->number_of_returns_of_given_pulse;
			m = number_return_map[n][r];
			l = number_return_level[n][r];

			if(enableTrace) printf("rnml: %u, %u, %u, %u \n", r, n, m, l);

			// decompress the intensity if it has changed
			if (changed_values & 16){
				((LASpoint10*)last_item)->intensity = (uint16_t)ic_intensity->decompress(last_intensity[m], (m < 3 ? m : 3));
				last_intensity[m] = ((LASpoint10*)last_item)->intensity;
			}else{
				((LASpoint10*)last_item)->intensity = last_intensity[m];
			}

			if(enableTrace) printf("intensity: %i \n", ((LASpoint10*)last_item)->intensity);

			// decompress the classification ... if it has changed
			if (changed_values & 8){
				if (m_classification[last_item[15]] == 0){
					m_classification[last_item[15]] = allocator->alloc<ArithmeticModel>(1);
					m_classification[last_item[15]]->init(256, allocator);
				}

				last_item[15] = (uint8_t)dec->decodeSymbol(m_classification[last_item[15]]);
			}

			if(enableTrace) printf("classification: %i \n", last_item[15]);
			
			// decompress the scan_angle_rank ... if it has changed
			if (changed_values & 4){
				int32_t val = dec->decodeSymbol(m_scan_angle_rank[((LASpoint10*)last_item)->scan_direction_flag]);
				last_item[16] = U8_FOLD(val + last_item[16]);
			}

			// decompress the user_data ... if it has changed
			if (changed_values & 2)
			{
				if (m_user_data[last_item[17]] == 0){
					m_user_data[last_item[17]] = allocator->alloc<ArithmeticModel>(1);
					m_user_data[last_item[17]]->init(256, allocator);
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

		// if(block.thread_rank() != 0) return;

		// decompress x coordinate
		if(enableTrace) printf("decompress x coordinate \n");
		median = last_x_diff_median5[m].get();
		diff = ic_dx->decompress(median, n==1);

		if(active.thread_rank() == 0){
			((LASpoint10*)last_item)->x += diff;
			last_x_diff_median5[m].add(diff);
		

			// decompress y coordinate
			if(enableTrace) printf("decompress y coordinate \n");
			median = last_y_diff_median5[m].get();
			k_bits = ic_dx->getK();
			diff = ic_dy->decompress(median, (n==1) + ( k_bits < 20 ? U32_ZERO_BIT_0(k_bits) : 20 ));
			((LASpoint10*)last_item)->y += diff;
			last_y_diff_median5[m].add(diff);

			// decompress z coordinate
			if(enableTrace) printf("decompress z coordinate \n");
			k_bits = (ic_dx->getK() + ic_dy->getK()) / 2;
			((LASpoint10*)last_item)->z = ic_z->decompress(last_height[l], (n==1) + (k_bits < 18 ? U32_ZERO_BIT_0(k_bits) : 18));
			last_height[l] = ((LASpoint10*)last_item)->z;

			// copy the last point
			memcpy(item, last_item, 20);

			uint64_t t_end = nanotime();
			if(t_end > t_start){
				t_readPoint10 += t_end - t_start;
			}
		}
	}

	//~LASreadItemCompressed_POINT10_v2();
	
};

#define LASZIP_GPSTIME_MULTI 500
#define LASZIP_GPSTIME_MULTI_MINUS -10
#define LASZIP_GPSTIME_MULTI_UNCHANGED (LASZIP_GPSTIME_MULTI - LASZIP_GPSTIME_MULTI_MINUS + 1)
#define LASZIP_GPSTIME_MULTI_CODE_FULL (LASZIP_GPSTIME_MULTI - LASZIP_GPSTIME_MULTI_MINUS + 2)
#define LASZIP_GPSTIME_MULTI_TOTAL (LASZIP_GPSTIME_MULTI - LASZIP_GPSTIME_MULTI_MINUS + 6) 

struct LASreadItemCompressed_GPSTIME11_v2{

	ArithmeticDecoder* dec;
	uint32_t last, next;
	U64I64F64 last_gpstime[4];
	int32_t last_gpstime_diff[4];
	int32_t multi_extreme_counter[4];

	ArithmeticModel* m_gpstime_multi;
	ArithmeticModel* m_gpstime_0diff;
	IntegerCompressor* ic_gpstime;

	bool init(ArithmeticDecoder* dec, const uint8_t* item, uint32_t& context, AllocatorGlobal* allocator){
		this->dec = dec;

		/* create entropy models and integer compressors */
		m_gpstime_multi = allocator->alloc<ArithmeticModel>(1);
		m_gpstime_0diff = allocator->alloc<ArithmeticModel>(1);
		m_gpstime_multi->init(LASZIP_GPSTIME_MULTI_TOTAL, allocator);
		m_gpstime_0diff->init(6, allocator);

		ic_gpstime = allocator->alloc<IntegerCompressor>(1);
		ic_gpstime->init(dec, 32, 9);
		ic_gpstime->initDecompressor(allocator);

		last = 0, next = 0;
		last_gpstime_diff[0] = 0;
		last_gpstime_diff[1] = 0;
		last_gpstime_diff[2] = 0;
		last_gpstime_diff[3] = 0;
		multi_extreme_counter[0] = 0;
		multi_extreme_counter[1] = 0;
		multi_extreme_counter[2] = 0;
		multi_extreme_counter[3] = 0;

		/* init last item */
		last_gpstime[0].u64 = *((uint64_t*)item);
		last_gpstime[1].u64 = 0;
		last_gpstime[2].u64 = 0;
		last_gpstime[3].u64 = 0;

		return true;
	}

	inline void read(uint8_t* item, uint32_t& context){

		auto grid = cg::this_grid();

		if(grid.thread_rank() != 0) return;

		uint64_t t_start = nanotime();

		int32_t multi;
		if (last_gpstime_diff[last] == 0) {
			// if the last integer difference was zero
			multi = dec->decodeSymbol(m_gpstime_0diff);

			if (multi == 1) {
				// the difference can be represented with 32 bits
				last_gpstime_diff[last] = ic_gpstime->decompress(0, 0);
				last_gpstime[last].i64 += last_gpstime_diff[last];
				multi_extreme_counter[last] = 0; 
			}else if (multi == 2) {
				// the difference is huge
				next = (next+1)&3;
				last_gpstime[next].u64 = ic_gpstime->decompress((int32_t)(last_gpstime[last].u64 >> 32), 8);
				last_gpstime[next].u64 = last_gpstime[next].u64 << 32;
				last_gpstime[next].u64 |= dec->readInt();
				last = next;
				last_gpstime_diff[last] = 0;
				multi_extreme_counter[last] = 0; 
			}else if (multi > 2) {
				// we switch to another sequence
				last = (last+multi-2)&3;
				read(item, context);
			}
		}else{
			multi = dec->decodeSymbol(m_gpstime_multi);
			
			if (multi == 1){
				last_gpstime[last].i64 += ic_gpstime->decompress(last_gpstime_diff[last], 1);;
				multi_extreme_counter[last] = 0;
			}else if (multi < LASZIP_GPSTIME_MULTI_UNCHANGED){
				int32_t gpstime_diff;
				
				if (multi == 0){
					gpstime_diff = ic_gpstime->decompress(0, 7);
					multi_extreme_counter[last]++;

					if (multi_extreme_counter[last] > 3){
						last_gpstime_diff[last] = gpstime_diff;
						multi_extreme_counter[last] = 0;
					}
				}else if (multi < LASZIP_GPSTIME_MULTI){
					if (multi < 10)
						gpstime_diff = ic_gpstime->decompress(multi*last_gpstime_diff[last], 2);
					else
						gpstime_diff = ic_gpstime->decompress(multi*last_gpstime_diff[last], 3);
				}else if (multi == LASZIP_GPSTIME_MULTI){
					gpstime_diff = ic_gpstime->decompress(LASZIP_GPSTIME_MULTI*last_gpstime_diff[last], 4);
					multi_extreme_counter[last]++;
					
					if (multi_extreme_counter[last] > 3){
						last_gpstime_diff[last] = gpstime_diff;
						multi_extreme_counter[last] = 0;
					}
				}else{
					multi = LASZIP_GPSTIME_MULTI - multi;

					if (multi > LASZIP_GPSTIME_MULTI_MINUS){
						gpstime_diff = ic_gpstime->decompress(multi*last_gpstime_diff[last], 5);
					}else{
						gpstime_diff = ic_gpstime->decompress(LASZIP_GPSTIME_MULTI_MINUS*last_gpstime_diff[last], 6);
						multi_extreme_counter[last]++;
						
						if (multi_extreme_counter[last] > 3){
							last_gpstime_diff[last] = gpstime_diff;
							multi_extreme_counter[last] = 0;
						}
					}
				}
				last_gpstime[last].i64 += gpstime_diff;
			}else if (multi ==  LASZIP_GPSTIME_MULTI_CODE_FULL){
				next = (next+1)&3;
				last_gpstime[next].u64 = ic_gpstime->decompress((int32_t)(last_gpstime[last].u64 >> 32), 8);
				last_gpstime[next].u64 = last_gpstime[next].u64 << 32;
				last_gpstime[next].u64 |= dec->readInt();
				last = next;
				last_gpstime_diff[last] = 0;
				multi_extreme_counter[last] = 0; 
			}else if (multi >=  LASZIP_GPSTIME_MULTI_CODE_FULL){
				last = (last+multi-LASZIP_GPSTIME_MULTI_CODE_FULL)&3;
				read(item, context);
			}
		}

		*((int64_t*)item) = last_gpstime[last].i64;

		uint64_t t_end = nanotime();
		if(t_end > t_start){
			t_readGps11 += t_end - t_start;
		}
	}

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

	bool init(ArithmeticDecoder* dec, const uint8_t* item, uint32_t& context, AllocatorGlobal* allocator){

		{ // former constructor stuff
			this->dec = dec;

			m_byte_used  = allocator->alloc<ArithmeticModel>(1);
			m_rgb_diff_0 = allocator->alloc<ArithmeticModel>(1);
			m_rgb_diff_1 = allocator->alloc<ArithmeticModel>(1);
			m_rgb_diff_2 = allocator->alloc<ArithmeticModel>(1);
			m_rgb_diff_3 = allocator->alloc<ArithmeticModel>(1);
			m_rgb_diff_4 = allocator->alloc<ArithmeticModel>(1);
			m_rgb_diff_5 = allocator->alloc<ArithmeticModel>(1);
		}

		/* init models and integer compressors */
		m_byte_used->init(128, allocator);
		m_rgb_diff_0->init(256, allocator);
		m_rgb_diff_1->init(256, allocator);
		m_rgb_diff_2->init(256, allocator);
		m_rgb_diff_3->init(256, allocator);
		m_rgb_diff_4->init(256, allocator);
		m_rgb_diff_5->init(256, allocator);

		/* init last item */
		memcpy(last_item, item, 6);

		return true;
	}

	inline void read(uint8_t* item, uint32_t& context){

		auto grid = cg::this_grid();

		if(grid.thread_rank() != 0) return;

		uint64_t t_start = nanotime();

		uint8_t corr;
		int32_t diff = 0;
		uint32_t sym = dec->decodeSymbol(m_byte_used);

		if(enableTrace) printf("sym: %u \n", sym);

		if (sym & (1 << 0)){
			corr = dec->decodeSymbol(m_rgb_diff_0);
			if(enableTrace) printf("corr[10]: %u \n", corr);
			((uint16_t*)item)[0] = (uint16_t)U8_FOLD(corr + (last_item[0]&255));
		}else {
			((uint16_t*)item)[0] = last_item[0]&0xFF;
		}
		
		if (sym & (1 << 1)){
			corr = dec->decodeSymbol(m_rgb_diff_1);
			if(enableTrace) printf("corr[20]: %u \n", corr);
			((uint16_t*)item)[0] |= (((uint16_t)U8_FOLD(corr + (last_item[0]>>8))) << 8);
		}else{
			((uint16_t*)item)[0] |= (last_item[0]&0xFF00);
		}
		
		if (sym & (1 << 6)){
			diff = (((uint16_t*)item)[0]&0x00FF) - (last_item[0]&0x00FF);

			if (sym & (1 << 2)){
				corr = dec->decodeSymbol(m_rgb_diff_2);
				((uint16_t*)item)[1] = (uint16_t)U8_FOLD(corr + U8_CLAMP(diff+(last_item[1]&255)));
			}else{
				((uint16_t*)item)[1] = last_item[1]&0xFF;
			}

			if (sym & (1 << 4)){
				corr = dec->decodeSymbol(m_rgb_diff_4);
				diff = (diff + ((((uint16_t*)item)[1]&0x00FF) - (last_item[1]&0x00FF))) / 2;
				((uint16_t*)item)[2] = (uint16_t)U8_FOLD(corr + U8_CLAMP(diff+(last_item[2]&255)));
			}else{
				((uint16_t*)item)[2] = last_item[2]&0xFF;
			}

			diff = (((uint16_t*)item)[0]>>8) - (last_item[0]>>8);
			
			if (sym & (1 << 3)){
				corr = dec->decodeSymbol(m_rgb_diff_3);
				((uint16_t*)item)[1] |= (((uint16_t)U8_FOLD(corr + U8_CLAMP(diff+(last_item[1]>>8))))<<8);
			}else{
				((uint16_t*)item)[1] |= (last_item[1]&0xFF00);
			}

			if (sym & (1 << 5)){
				corr = dec->decodeSymbol(m_rgb_diff_5);
				diff = (diff + ((((uint16_t*)item)[1]>>8) - (last_item[1]>>8))) / 2;
				((uint16_t*)item)[2] |= (((uint16_t)U8_FOLD(corr + U8_CLAMP(diff+(last_item[2]>>8))))<<8);
			}else{
				((uint16_t*)item)[2] |= (last_item[2]&0xFF00);
			}
		}else{
			((uint16_t*)item)[1] = ((uint16_t*)item)[0];
			((uint16_t*)item)[2] = ((uint16_t*)item)[0];
		}

		memcpy(last_item, item, 6);

		uint64_t t_end = nanotime();
		if(t_end > t_start){
			t_readRgb12 += t_end - t_start;
		}
	}

};