// Adapted from LASzip
// https://github.com/LASzip/LASzip/blob/80f929870f4ae2c3dc4e4f9e00e0e68f17ba1f1e/src/laszip_common_v2.hpp
// LICENSE: https://github.com/LASzip/LASzip/blob/80f929870f4ae2c3dc4e4f9e00e0e68f17ba1f1e/COPYING (Apache 2.0)

#pragma once

struct StreamingMedian5{
	int32_t values[5];
	bool high;

	void init(){
		values[0] = values[1] = values[2] = values[3] = values[4] = 0;
		high = true;
	}

	inline void add(int32_t v){
		uint64_t t_start = nanotime();

		if (high){
			if (v < values[2]){
				values[4] = values[3];
				values[3] = values[2];
				if (v < values[0]){
					values[2] = values[1];
					values[1] = values[0];
					values[0] = v;
				}else if (v < values[1]){
					values[2] = values[1];
					values[1] = v;
				}else{
					values[2] = v;
				}
			}else{
				if (v < values[3]){
					values[4] = values[3];
					values[3] = v;
				}else{
					values[4] = v;
				}
				high = false;
			}
		}else{
			if (values[2] < v){
				values[0] = values[1];
				values[1] = values[2];

				if (values[4] < v){
					values[2] = values[3];
					values[3] = values[4];
					values[4] = v;
				}else if (values[3] < v){
					values[2] = values[3];
					values[3] = v;
				}else{
					values[2] = v;
				}
			}else{
				if (values[1] < v){
					values[0] = values[1];
					values[1] = v;
				}else{
					values[0] = v;
				}
				high = true;
			}
		}

		uint64_t t_end = nanotime();
		if(t_end > t_start){
			t_streamingMedian += t_end - t_start;
		}
	}

	int32_t get() const{
		return values[2];
	}

	StreamingMedian5(){
		init();
	}
};

// for LAS files with the return (r) and the number (n) of
// returns field correctly populated the mapping should really
// be only the following.
//  { 15, 15, 15, 15, 15, 15, 15, 15 },
//  { 15,  0, 15, 15, 15, 15, 15, 15 },
//  { 15,  1,  2, 15, 15, 15, 15, 15 },
//  { 15,  3,  4,  5, 15, 15, 15, 15 },
//  { 15,  6,  7,  8,  9, 15, 15, 15 },
//  { 15, 10, 11, 12, 13, 14, 15, 15 },
//  { 15, 15, 15, 15, 15, 15, 15, 15 },
//  { 15, 15, 15, 15, 15, 15, 15, 15 }
// however, some files start the numbering of r and n with 0,
// only have return counts r, or only have number of return
// counts n, or mix up the position of r and n. we therefore
// "complete" the table to also map those "undesired" r & n
// combinations to different contexts
const uint8_t number_return_map[8][8] = 
{
  { 15, 14, 13, 12, 11, 10,  9,  8 },
  { 14,  0,  1,  3,  6, 10, 10,  9 },
  { 13,  1,  2,  4,  7, 11, 11, 10 },
  { 12,  3,  4,  5,  8, 12, 12, 11 },
  { 11,  6,  7,  8,  9, 13, 13, 12 },
  { 10, 10, 11, 12, 13, 14, 14, 13 },
  {  9, 10, 11, 12, 13, 14, 15, 14 },
  {  8,  9, 10, 11, 12, 13, 14, 15 }
};

// for LAS files with the return (r) and the number (n) of
// returns field correctly populated the mapping should really
// be only the following.
//  {  0,  7,  7,  7,  7,  7,  7,  7 },
//  {  7,  0,  7,  7,  7,  7,  7,  7 },
//  {  7,  1,  0,  7,  7,  7,  7,  7 },
//  {  7,  2,  1,  0,  7,  7,  7,  7 },
//  {  7,  3,  2,  1,  0,  7,  7,  7 },
//  {  7,  4,  3,  2,  1,  0,  7,  7 },
//  {  7,  5,  4,  3,  2,  1,  0,  7 },
//  {  7,  6,  5,  4,  3,  2,  1,  0 }
// however, some files start the numbering of r and n with 0,
// only have return counts r, or only have number of return
// counts n, or mix up the position of r and n. we therefore
// "complete" the table to also map those "undesired" r & n
// combinations to different contexts
const uint8_t number_return_level[8][8] = 
{
  {  0,  1,  2,  3,  4,  5,  6,  7 },
  {  1,  0,  1,  2,  3,  4,  5,  6 },
  {  2,  1,  0,  1,  2,  3,  4,  5 },
  {  3,  2,  1,  0,  1,  2,  3,  4 },
  {  4,  3,  2,  1,  0,  1,  2,  3 },
  {  5,  4,  3,  2,  1,  0,  1,  2 },
  {  6,  5,  4,  3,  2,  1,  0,  1 },
  {  7,  6,  5,  4,  3,  2,  1,  0 }
};