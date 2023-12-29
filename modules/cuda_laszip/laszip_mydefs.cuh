// Adapted from LASzip
// https://github.com/LASzip/LASzip/blob/80f929870f4ae2c3dc4e4f9e00e0e68f17ba1f1e/src/mydefs.hpp#L116
// LICENSE: https://github.com/LASzip/LASzip/blob/80f929870f4ae2c3dc4e4f9e00e0e68f17ba1f1e/COPYING (Apache 2.0)

#pragma once

struct laszip_point{
	int32_t X;
	int32_t Y;
	int32_t Z;
	uint16_t intensity;
	uint8_t return_number : 3;
	uint8_t number_of_returns : 3;
	uint8_t scan_direction_flag : 1;
	uint8_t edge_of_flight_line : 1;
	uint8_t classification : 5;
	uint8_t synthetic_flag : 1;
	uint8_t keypoint_flag  : 1;
	uint8_t withheld_flag  : 1;
	int8_t scan_angle_rank;
	uint8_t user_data;
	uint16_t point_source_ID;

	// LAS 1.4 only
	int16_t extended_scan_angle;
	uint8_t extended_point_type : 2;
	uint8_t extended_scanner_channel : 2;
	uint8_t extended_classification_flags : 4;
	uint8_t extended_classification;
	uint8_t extended_return_number : 4;
	uint8_t extended_number_of_returns : 4;

	// for 8 byte alignment of the GPS time
	uint8_t dummy[7];

	double gps_time;
	uint16_t rgb[4];
	uint8_t wave_packet[29];

	int32_t num_extra_bytes;
	uint8_t* extra_bytes;

};


typedef union U32I32F32 { uint32_t u32; int32_t i32; float f32; } U32I32F32;
typedef union U64I64F64 { uint64_t u64; int64_t i64; double f64; } U64I64F64;
typedef union I64U32I32F32 { int64_t i64; uint32_t u32[2]; int32_t i32[2]; float f32[2]; } I64U32I32F32;

#define F32_MAX            +2.0e+37f
#define F32_MIN            -2.0e+37f

#define F64_MAX            +2.0e+307
#define F64_MIN            -2.0e+307

#define U8_MIN             ((uint8_t)0x0)  // 0
#define U8_MAX             ((uint8_t)0xFF) // 255
#define U8_MAX_MINUS_ONE   ((uint8_t)0xFE) // 254
#define U8_MAX_PLUS_ONE    0x0100     // 256

#define U16_MIN            ((uint16_t)0x0)    // 0
#define U16_MAX            ((uint16_t)0xFFFF) // 65535
#define U16_MAX_MINUS_ONE  ((uint16_t)0xFFFE) // 65534
#define U16_MAX_PLUS_ONE   0x00010000    // 65536

#define U32_MIN            ((uint32_t)0x0)            // 0
#define U32_MAX            ((uint32_t)0xFFFFFFFF)     // 4294967295
#define U32_MAX_MINUS_ONE  ((uint32_t)0xFFFFFFFE)     // 4294967294
#if defined(WIN32)            // 64 byte unsigned int constant under Windows 
#define U32_MAX_PLUS_ONE   0x0000000100000000    // 4294967296
#else                         // 64 byte unsigned int constant elsewhere ... 
#define U32_MAX_PLUS_ONE   0x0000000100000000ull // 4294967296
#endif

#define I8_MIN             ((int8_t)0x80) // -128
#define I8_MAX             ((int8_t)0x7F) // 127

#define I16_MIN            ((int16_t)0x8000) // -32768
#define I16_MAX            ((int16_t)0x7FFF) // 32767

#define I32_MIN            ((int32_t)0x80000000) // -2147483648
#define I32_MAX            ((int32_t)0x7FFFFFFF) //  2147483647

#define I64_MIN            ((int64_t)0x8000000000000000)
#define I64_MAX            ((int64_t)0x7FFFFFFFFFFFFFFF)

#define U8_FOLD(n)      (((n) < U8_MIN) ? (n+U8_MAX_PLUS_ONE) : (((n) > U8_MAX) ? (n-U8_MAX_PLUS_ONE) : (n)))

#define I8_CLAMP(n)     (((n) <= I8_MIN) ? I8_MIN : (((n) >= I8_MAX) ? I8_MAX : ((int8_t)(n))))
#define U8_CLAMP(n)     (((n) <= U8_MIN) ? U8_MIN : (((n) >= U8_MAX) ? U8_MAX : ((uint8_t)(n))))

#define I16_CLAMP(n)    (((n) <= I16_MIN) ? I16_MIN : (((n) >= I16_MAX) ? I16_MAX : ((int16_t)(n))))
#define U16_CLAMP(n)    (((n) <= U16_MIN) ? U16_MIN : (((n) >= U16_MAX) ? U16_MAX : ((uint16_t)(n))))

#define I32_CLAMP(n)    (((n) <= I32_MIN) ? I32_MIN : (((n) >= I32_MAX) ? I32_MAX : ((int32_t)(n))))
#define U32_CLAMP(n)    (((n) <= U32_MIN) ? U32_MIN : (((n) >= U32_MAX) ? U32_MAX : ((uint32_t)(n))))

#define I8_QUANTIZE(n) (((n) >= 0) ? (int8_t)((n)+0.5) : (int8_t)((n)-0.5))
#define U8_QUANTIZE(n) (((n) >= 0) ? (uint8_t)((n)+0.5) : (uint8_t)(0))

#define I16_QUANTIZE(n) (((n) >= 0) ? (int16_t)((n)+0.5) : (int16_t)((n)-0.5))
#define U16_QUANTIZE(n) (((n) >= 0) ? (uint16_t)((n)+0.5) : (uint16_t)(0))

#define I32_QUANTIZE(n) (((n) >= 0) ? (int32_t)((n)+0.5) : (int32_t)((n)-0.5))
#define U32_QUANTIZE(n) (((n) >= 0) ? (uint32_t)((n)+0.5) : (uint32_t)(0))

#define I64_QUANTIZE(n) (((n) >= 0) ? (int64_t)((n)+0.5) : (int64_t)((n)-0.5))
#define U64_QUANTIZE(n) (((n) >= 0) ? (uint64_t)((n)+0.5) : (uint64_t)(0))

#define I8_CLAMP_QUANTIZE(n)     (((n) <= I8_MIN) ? I8_MIN : (((n) >= I8_MAX) ? I8_MAX : (I8_QUANTIZE(n))))
#define U8_CLAMP_QUANTIZE(n)     (((n) <= U8_MIN) ? U8_MIN : (((n) >= U8_MAX) ? U8_MAX : (U8_QUANTIZE(n))))

#define I16_CLAMP_QUANTIZE(n)    (((n) <= I16_MIN) ? I16_MIN : (((n) >= I16_MAX) ? I16_MAX : (I16_QUANTIZE(n))))
#define U16_CLAMP_QUANTIZE(n)    (((n) <= U16_MIN) ? U16_MIN : (((n) >= U16_MAX) ? U16_MAX : (U16_QUANTIZE(n))))

#define I32_CLAMP_QUANTIZE(n)    (((n) <= I32_MIN) ? I32_MIN : (((n) >= I32_MAX) ? I32_MAX : (I32_QUANTIZE(n))))
#define U32_CLAMP_QUANTIZE(n)    (((n) <= U32_MIN) ? U32_MIN : (((n) >= U32_MAX) ? U32_MAX : (U32_QUANTIZE(n))))


#define I16_FLOOR(n) ((((int16_t)(n)) > (n)) ? (((int16_t)(n))-1) : ((int16_t)(n)))
#define I32_FLOOR(n) ((((int32_t)(n)) > (n)) ? (((int32_t)(n))-1) : ((int32_t)(n)))
#define I64_FLOOR(n) ((((int64_t)(n)) > (n)) ? (((int64_t)(n))-1) : ((int64_t)(n)))

#define I16_CEIL(n) ((((int16_t)(n)) < (n)) ? (((int16_t)(n))+1) : ((int16_t)(n)))
#define I32_CEIL(n) ((((int32_t)(n)) < (n)) ? (((int32_t)(n))+1) : ((int32_t)(n)))
#define I64_CEIL(n) ((((int64_t)(n)) < (n)) ? (((int64_t)(n))+1) : ((int64_t)(n)))

#define I8_FITS_IN_RANGE(n) (((n) >= I8_MIN) && ((n) <= I8_MAX) ? TRUE : FALSE)
#define U8_FITS_IN_RANGE(n) (((n) >= U8_MIN) && ((n) <= U8_MAX) ? TRUE : FALSE)
#define I16_FITS_IN_RANGE(n) (((n) >= I16_MIN) && ((n) <= I16_MAX) ? TRUE : FALSE)
#define U16_FITS_IN_RANGE(n) (((n) >= U16_MIN) && ((n) <= U16_MAX) ? TRUE : FALSE)
#define I32_FITS_IN_RANGE(n) (((n) >= I32_MIN) && ((n) <= I32_MAX) ? TRUE : FALSE)
#define U32_FITS_IN_RANGE(n) (((n) >= U32_MIN) && ((n) <= U32_MAX) ? TRUE : FALSE)

#define F32_IS_FINITE(n) ((F32_MIN < (n)) && ((n) < F32_MAX))
#define F64_IS_FINITE(n) ((F64_MIN < (n)) && ((n) < F64_MAX))

#define U32_ZERO_BIT_0(n) (((n)&(uint32_t)0xFFFFFFFE))
#define U32_ZERO_BIT_0_1(n) (((n)&(uint32_t)0xFFFFFFFC))