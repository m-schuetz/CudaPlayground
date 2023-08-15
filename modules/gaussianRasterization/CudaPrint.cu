
#pragma once

constexpr uint32_t TYPE_UINT32_T = 0;
constexpr uint32_t TYPE_UINT64_T = 1;
constexpr uint32_t TYPE_INT32_T  = 2;
constexpr uint32_t TYPE_INT64_T  = 3;
constexpr uint32_t TYPE_FLOAT    = 4;
constexpr uint32_t TYPE_DOUBLE   = 5;

struct CudaPrintEntry{
	uint32_t keylen;

	uint8_t numArgs;
	uint8_t method;
	uint8_t padding_0;
	uint8_t padding_1;

	uint8_t data[1024 - 16];
};

struct CudaPrint{
	uint64_t entryCounter = 0;
	uint64_t padding;
	CudaPrintEntry entries[1000];

	static uint32_t typeof(uint32_t value){ return TYPE_UINT32_T; }
	static uint32_t typeof(uint64_t value){ return TYPE_UINT64_T; }
	static uint32_t typeof(int32_t  value){ return TYPE_INT32_T; }
	static uint32_t typeof(int64_t  value){ return TYPE_INT64_T; }
	static uint32_t typeof(float    value){ return TYPE_FLOAT; }
	static uint32_t typeof(double   value){ return TYPE_DOUBLE; }

	void testType(uint32_t value){
		printf("uint32_t \n");
	}

	void testType(int32_t value){
		printf("int32_t \n");
	}

	void testType(float value){
		printf("float \n");
	}

	template <typename... Args>
	inline void print(const char* key, const Args&... args) {

		uint32_t entryIndex = atomicAdd(&entryCounter, 1) % 1000;
		CudaPrintEntry *entry = &entries[entryIndex];

		constexpr uint32_t numargs{ sizeof...(Args) };

		int argsSize = 0;

		for(const auto p : {args...}) {
			// testType(p);
			argsSize += sizeof(p);
			entry->numArgs++;
		}

		entry->keylen = strlen(key);
		entry->method = 0;

		memcpy(entry->data, key, entry->keylen);

		uint32_t offset = entry->keylen;
		for(const auto p : {args...}) {
			// testType(p);
			uint32_t argSize = sizeof(p);
			uint32_t argtype = CudaPrint::typeof(p);
			
			memcpy(entry->data + offset, &argtype, 4);
			offset += 4;
			memcpy(entry->data + offset, &p, sizeof(p));
			offset += sizeof(p);

		}
	}

	inline void set(const char* key, const char* value) {

		// uint32_t entryIndex = atomicAdd(&entryCounter, 1) % 1000;
		// CudaPrintEntry *entry = &entries[entryIndex];

		// entry->keylen   = strlen(key);
		// entry->numArgs  = 1;
		// entry->method   = 1;

		// uint32_t argtype = CudaPrint::typeof(value);

		// memcpy(entry->data, key, entry->keylen);
		// memcpy(entry->data + entry->keylen, &argtype, 4);
		// memcpy(entry->data + entry->keylen + 4, &value, sizeof(value));
	}

	template <typename T>
	inline void set(const char* key, const T value) {

		uint32_t entryIndex = atomicAdd(&entryCounter, 1) % 1000;
		CudaPrintEntry *entry = &entries[entryIndex];

		entry->keylen   = strlen(key);
		entry->numArgs  = 1;
		entry->method   = 1;

		uint32_t argtype = CudaPrint::typeof(value);

		memcpy(entry->data, key, entry->keylen);
		memcpy(entry->data + entry->keylen, &argtype, 4);
		memcpy(entry->data + entry->keylen + 4, &value, sizeof(value));
	}
};