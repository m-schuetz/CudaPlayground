#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

constexpr bool EDL_ENABLED = false;
constexpr uint32_t gridSize = 128;
constexpr float fGridSize = gridSize;
constexpr uint32_t numCells = gridSize * gridSize * gridSize;
constexpr float3 gridMin = { -1.0f, -1.0f, 0.0f};
constexpr float3 gridMax = { 1.0f, 1.0f, 2.0f};


float cross(float2 a, float2 b){
	return a.x * b.y - a.y * b.x;
}

// table from http://paulbourke.net/geometry/polygonise/marchingsource.cpp
// (publich domain)
constexpr int edgeTable[256] = {
	0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 
	0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 
	0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 
	0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 
	0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60, 
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0, 
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460, 
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0, 
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230, 
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190, 
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

// table from http://paulbourke.net/geometry/polygonise/marchingsource.cpp
// (publich domain)
constexpr int triTable[256][16] =
{
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

struct RenderTarget{
	mat4 view;
	mat4 proj;
	mat4 transform;
	uint64_t* framebuffer;
	float width;
	float height;
};

float4 operator*(const mat4& a, const float4& b){
	return make_float4(
		dot(a.rows[0], b),
		dot(a.rows[1], b),
		dot(a.rows[2], b),
		dot(a.rows[3], b)
	);
}

mat4 operator*(const mat4& a, const mat4& b){
	
	mat4 result;

	result.rows[0].x = dot(a.rows[0], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
	result.rows[0].y = dot(a.rows[0], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
	result.rows[0].z = dot(a.rows[0], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
	result.rows[0].w = dot(a.rows[0], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

	result.rows[1].x = dot(a.rows[1], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
	result.rows[1].y = dot(a.rows[1], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
	result.rows[1].z = dot(a.rows[1], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
	result.rows[1].w = dot(a.rows[1], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

	result.rows[2].x = dot(a.rows[2], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
	result.rows[2].y = dot(a.rows[2], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
	result.rows[2].z = dot(a.rows[2], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
	result.rows[2].w = dot(a.rows[2], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

	result.rows[3].x = dot(a.rows[3], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
	result.rows[3].y = dot(a.rows[3], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
	result.rows[3].z = dot(a.rows[3], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
	result.rows[3].w = dot(a.rows[3], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

	return result;
}

struct Intersection{
	float3 position;
	float distance;
	
	bool intersects(){
		return distance > 0.0f && distance != Infinity;
	}
};

Intersection rayPlane(float3 origin, float3 direction, float3 planeNormal, float planeDistance){

	float denominator = dot(planeNormal, direction);

	if(denominator == 0){
		Intersection I;
		I.distance = Infinity;

		return I;
	}else{
		float distance = - (dot(origin, planeNormal) + planeDistance ) / denominator;

		Intersection I;
		I.distance = distance;
		I.position = origin + direction * distance;

		return I;
	}

}

namespace cg = cooperative_groups;

Uniforms uniforms;
Allocator* allocator;
uint64_t nanotime_start;

constexpr float PI = 3.1415;
constexpr uint32_t BACKGROUND_COLOR = 0x00332211ull;

struct Triangles{
	int numTriangles;
	float3* positions;
	float2* uvs;
	uint32_t* colors;
};

struct Texture{
	int width;
	int height;
	uint32_t* data;
};

struct RasterizationSettings{
	Texture* texture = nullptr;
	int colorMode = COLORMODE_TRIANGLE_ID;
	mat4 world;
	mat4 view;
	mat4 proj;
	mat4 transform;
	float width; 
	float height;
};

uint32_t sample_nearest(float2 uv, Texture* texture){
	int tx = int(uv.x * texture->width) % texture->width;
	int ty = int(uv.y * texture->height) % texture->height;
	ty = texture->height - ty;

	int texelIndex = tx + texture->width * ty;
	uint32_t texel = texture->data[texelIndex];

	return texel;
}

uint32_t sample_linear(float2 uv, Texture* texture){
	float width = texture->width;
	float height = texture->height;

	float tx = uv.x * width;
	float ty = height - uv.y * height;

	int x0 = clamp(floor(tx), 0.0f, width - 1.0f);
	int x1 = clamp(ceil(tx) , 0.0f, width - 1.0f);
	int y0 = clamp(floor(ty), 0.0f, height - 1.0f);
	int y1 = clamp(ceil(ty) , 0.0f, height - 1.0f);
	float wx = tx - floor(tx);
	float wy = ty - floor(ty);

	float w00 = (1.0 - wx) * (1.0 - wy);
	float w10 = wx * (1.0 - wy);
	float w01 = (1.0 - wx) * wy;
	float w11 = wx * wy;

	uint8_t* c00 = (uint8_t*)&texture->data[x0 + y0 * texture->width];
	uint8_t* c10 = (uint8_t*)&texture->data[x1 + y0 * texture->width];
	uint8_t* c01 = (uint8_t*)&texture->data[x0 + y1 * texture->width];
	uint8_t* c11 = (uint8_t*)&texture->data[x1 + y1 * texture->width];

	uint32_t color;
	uint8_t* rgb = (uint8_t*)&color;

	rgb[0] = c00[0] * w00 + c10[0] * w10 + c01[0] * w01 + c11[0] * w11;
	rgb[1] = c00[1] * w00 + c10[1] * w10 + c01[1] * w01 + c11[1] * w11;
	rgb[2] = c00[2] * w00 + c10[2] * w10 + c01[2] * w01 + c11[2] * w11;

	return color;
}

// rasterizes triangles in a block-wise fashion
// - each block grabs a triangle
// - all threads of that block process different fragments of the triangle
// - <framebuffer> stores interleaved 32bit depth and color values
// - The closest fragments are rendered via atomicMin on a combined 64bit depth&color integer
//   atomicMin(&framebuffer[pixelIndex], (depth << 32 | color)); 
// see http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html#algo3
void rasterizeTriangles(Triangles* triangles, uint64_t* framebuffer, RasterizationSettings settings){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	Texture* texture = settings.texture;
	int colorMode = settings.colorMode;
	
	mat4 transform = settings.proj * settings.view * settings.world;

	uint32_t& processedTriangles = *allocator->alloc<uint32_t*>(4);
	if(grid.thread_rank() == 0){
		processedTriangles = 0;
	}
	grid.sync();

	float3 lightPos = {10.0f, 10.0f, 10.0f};
	// float3 L = {10.0f, 10.0f, 10.0f};

	if(uniforms.vrEnabled){
		float4 leftPos = uniforms.vr_left_view_inv * float4{0.0, 0.0f, 0.0f, 1.0f};

		lightPos = {leftPos.x, leftPos.y, leftPos.z};
	}

	{
		__shared__ int sh_triangleIndex;

		block.sync();

		// safety mechanism: each block draws at most <loop_max> triangles
		int loop_max = 10'000;
		for(int loop_i = 0; loop_i < loop_max; loop_i++){
			
			// grab the index of the next unprocessed triangle
			block.sync();
			if(block.thread_rank() == 0){
				sh_triangleIndex = atomicAdd(&processedTriangles, 1);
			}
			block.sync();

			if(sh_triangleIndex >= triangles->numTriangles) break;

			// project x/y to pixel coords
			// z: whatever 
			// w: linear depth
			auto toScreenCoord = [&](float3 p){
				float4 pos = transform * float4{p.x, p.y, p.z, 1.0f};

				pos.x = pos.x / pos.w;
				pos.y = pos.y / pos.w;
				// pos.z = pos.z / pos.w;

				float4 imgPos = {
					(pos.x * 0.5f + 0.5f) * settings.width, 
					(pos.y * 0.5f + 0.5f) * settings.height,
					pos.z, 
					pos.w
				};

				return imgPos;
			};

			int i0 = 3 * sh_triangleIndex + 0;
			int i1 = 3 * sh_triangleIndex + 1;
			int i2 = 3 * sh_triangleIndex + 2;
			
			float3 v0 = triangles->positions[i0];
			float3 v1 = triangles->positions[i1];
			float3 v2 = triangles->positions[i2];

			float4 p0 = toScreenCoord(v0);
			float4 p1 = toScreenCoord(v1);
			float4 p2 = toScreenCoord(v2);

			// cull a triangle if one of its vertices is closer than depth 0
			if(p0.w < 0.0 || p1.w < 0.0 || p2.w < 0.0) continue;


			float3 v01 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
			float3 v02 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
			float3 N = normalize(cross(v02, v01));

			// if(sh_triangleIndex == 0 && block.thread_rank() == 0){
			// 	printf("%f, %f, %f \n", N.x, N.y, N.z);
			// }

			float2 p01 = {p1.x - p0.x, p1.y - p0.y};
			float2 p02 = {p2.x - p0.x, p2.y - p0.y};

			// auto cross = [](float2 a, float2 b){ return a.x * b.y - a.y * b.x; };
			// auto cross = [](float3 a, float3 b){ return a.y * b.z - a.y * b.x; };

			{// backface culling
				float w = cross(p01, p02);
				if(w < 0.0) continue;
			}

			// compute screen-space bounding rectangle
			float min_x = min(min(p0.x, p1.x), p2.x);
			float min_y = min(min(p0.y, p1.y), p2.y);
			float max_x = max(max(p0.x, p1.x), p2.x);
			float max_y = max(max(p0.y, p1.y), p2.y);

			// clamp to screen
			min_x = clamp(min_x, 0.0f, settings.width);
			min_y = clamp(min_y, 0.0f, settings.height);
			max_x = clamp(max_x, 0.0f, settings.width);
			max_y = clamp(max_y, 0.0f, settings.height);

			int size_x = ceil(max_x) - floor(min_x);
			int size_y = ceil(max_y) - floor(min_y);
			int numFragments = size_x * size_y;

			// iterate through fragments in bounding rectangle and draw if within triangle
			int numProcessedSamples = 0;
			for(int fragOffset = 0; fragOffset < numFragments; fragOffset += block.num_threads()){

				// safety mechanism: don't draw more than <x> pixels per thread
				if(numProcessedSamples > 5'000) break;

				int fragID = fragOffset + block.thread_rank();
				int fragX = fragID % size_x;
				int fragY = fragID / size_x;

				float2 pFrag = {
					floor(min_x) + float(fragX), 
					floor(min_y) + float(fragY)
				};
				float2 sample = {pFrag.x - p0.x, pFrag.y - p0.y};

				// v: vertex[0], s: vertex[1], t: vertex[2]
				float s = cross(sample, p02) / cross(p01, p02);
				float t = cross(p01, sample) / cross(p01, p02);
				float v = 1.0 - (s + t);

				int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
				int pixelID = pixelCoords.x + pixelCoords.y * settings.width;
				pixelID = clamp(pixelID, 0, int(settings.width * settings.height) - 1);

				if(s >= 0.0)
				if(t >= 0.0)
				if(s + t <= 1.0)
				{
					uint8_t* v0_rgba = (uint8_t*)&triangles->colors[i0];
					uint8_t* v1_rgba = (uint8_t*)&triangles->colors[i1];
					uint8_t* v2_rgba = (uint8_t*)&triangles->colors[i2];

					float2 v0_uv = triangles->uvs[i0] / p0.z;
					float2 v1_uv = triangles->uvs[i1] / p1.z;
					float2 v2_uv = triangles->uvs[i2] / p2.z;
					float2 uv = {
						v * v0_uv.x + s * v1_uv.x + t * v2_uv.x,
						v * v0_uv.y + s * v1_uv.y + t * v2_uv.y
					};
					float repz = v * (1.0f / p0.z) + s * (1.0f / p1.z) + t * (1.0f / p2.z);
					uv.x = uv.x / repz;
					uv.y = uv.y / repz;

					uint32_t color;
					uint8_t* rgb = (uint8_t*)&color;

					// { // color by vertex color
					// 	rgb[0] = v * v0_rgba[0] + s * v1_rgba[0] + t * v2_rgba[0];
					// 	rgb[1] = v * v0_rgba[1] + s * v1_rgba[1] + t * v2_rgba[1];
					// 	rgb[2] = v * v0_rgba[2] + s * v1_rgba[2] + t * v2_rgba[2];
					// }

					if(colorMode == COLORMODE_TEXTURE && texture != nullptr){
						// TEXTURE
						int tx = int(uv.x * texture->width) % texture->width;
						int ty = int(uv.y * texture->height) % texture->height;
						ty = texture->height - ty;

						int texelIndex = tx + texture->width * ty;
						uint32_t texel = texture->data[texelIndex];
						uint8_t* texel_rgb = (uint8_t*)&texel;

						if(uniforms.sampleMode == SAMPLEMODE_NEAREST){
							color = sample_nearest(uv, texture);
						}else if(uniforms.sampleMode == SAMPLEMODE_LINEAR){
							color = sample_linear(uv, texture);
						}
					}else if(colorMode == COLORMODE_UV && triangles->uvs != nullptr){
						// UV
						rgb[0] = 255.0f * uv.x;
						rgb[1] = 255.0f * uv.y;
						rgb[2] = 0;
					}else if(colorMode == COLORMODE_TRIANGLE_ID){
						// TRIANGLE INDEX
						color = sh_triangleIndex * 123456;
					}else if(colorMode == COLORMODE_TIME || colorMode == COLORMODE_TIME_NORMALIZED){
						// TIME
						uint64_t nanotime;
						asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime));
						color = (nanotime - nanotime_start) % 0x00ffffffull;
					}else if(colorMode == COLORMODE_VERTEXCOLOR){
						color = triangles->colors[i0];
					}else{
						// WHATEVER
						color = sh_triangleIndex * 123456;
					}

					// float3 L = {10.0f, 10.0f, 10.0f};
					float3 L = normalize(v0 - lightPos);

					float lambertian = max(dot(N, L), 0.0);

					// rgb[0] = 255.0f *  N.x;
					// rgb[1] = 255.0f *  N.y;
					// rgb[2] = 255.0f *  N.z;

					float3 diffuse = {0.8f, 0.8f, 0.8f};
					float3 ambient = {0.2f, 0.2f, 0.2f};
					rgb[0] = rgb[0] * (lambertian * diffuse.x + ambient.x);
					rgb[1] = rgb[1] * (lambertian * diffuse.y + ambient.y);
					rgb[2] = rgb[2] * (lambertian * diffuse.z + ambient.z);

					float depth = v * p0.w + s * p1.w + t * p2.w;
					uint64_t udepth = *((uint32_t*)&depth);
					uint64_t pixel = (udepth << 32ull) | color;

					atomicMin(&framebuffer[pixelID], pixel);
				}

				numProcessedSamples++;
			}


		}
	}
}

void rasterizeVoxels(
	int gridSize, 
	int numCells, 
	uint32_t* voxelGrid,
	int numTargets,
	RenderTarget* targets
){

	float3 boxSize = gridMax - gridMin;

	float voxelSize = boxSize.x / float(gridSize);
	
	processRange(0, numCells, [&](int voxelIndex){
		int vx = voxelIndex % gridSize;
		int vy = voxelIndex % (gridSize * gridSize) / gridSize;
		int vz = voxelIndex / (gridSize * gridSize);

		uint32_t value = voxelGrid[voxelIndex];

		if(value != 0){

			float x = (float(vx) / fGridSize) * boxSize.x + gridMin.x;
			float y = (float(vy) / fGridSize) * boxSize.y + gridMin.y;
			float z = (float(vz) / fGridSize) * boxSize.z + gridMin.z;

			for(int targetIndex = 0; targetIndex < numTargets; targetIndex++){
				RenderTarget* target = &targets[targetIndex];

				float4 pos = target->transform * float4{x, y, z, 1.0f};
				
				float4 pos_off = target->view * float4{x, y, z, 1.0f};
				pos_off.x += voxelSize;
				pos_off = target->proj * pos_off;

				float4 ndc = {
					pos.x / pos.w, 
					pos.y / pos.w, 
					pos.z / pos.w,
					pos.w
				};
				float4 ndc_off = {
					pos_off.x / pos.w, 
					pos_off.y / pos.w, 
					pos_off.z / pos.w,
					pos_off.w
				};

				float2 screenPos = {
					(ndc.x * 0.5f + 0.5f) * target->width, 
					(ndc.y * 0.5f + 0.5f) * target->height,
				};
				float2 screenPos_off = {
					(ndc_off.x * 0.5f + 0.5f) * target->width, 
					(ndc_off.y * 0.5f + 0.5f) * target->height,
				};

				float voxelSize_screen = abs(screenPos_off.x - screenPos.x);
				voxelSize_screen = clamp(voxelSize_screen, 1.0f, 50.0f);

				if(0.0f <= screenPos.x && screenPos.x < target->width)
				if(0.0f <= screenPos.y && screenPos.y < target->height)
				if(ndc.w > 0.0f){

					int px = screenPos.x;
					int py = screenPos.y;
					int pixelID = px + py * target->width;

					uint64_t color = 0x000000ff;
					uint64_t idepth = *((uint32_t*)&pos.w);

					// int spriteRadius = voxelSize_screen / 2.0f + 1.0f;
					int spriteRadius = 4;
					int spriteSize = 2 * spriteRadius + 1;
					for(int ox = -spriteRadius; ox <= spriteRadius; ox++)
					for(int oy = -spriteRadius; oy <= spriteRadius; oy++)
					{
						float u = float(ox) / spriteRadius;
						float v = float(oy) / spriteRadius;
						float dd = u * u + v * v;

						if(dd > 1.0f) continue;

						float d = sqrt(d);

						uint32_t color;
						uint8_t* rgba = (uint8_t*)&color;
						rgba[0] = 255 * u;
						rgba[1] = 255 * v;
						rgba[2] = 0;

						float w = dd;
						rgba[0] = 255 - 200 * w;
						rgba[1] = 255 - 200 * w;
						rgba[2] = 255 - 200 * w;
						rgba[3] = 0;

						float fragdepth = pos.w + dd * voxelSize;
						uint64_t idepth = *((uint32_t*)&fragdepth);
     
 
						uint64_t pixel = (idepth << 32) | color;
						
						if(!((0 <= px + ox) && (px + ox < target->width))) continue;
						if(!((0 <= py + oy) && (py + oy < target->height))) continue;

						atomicMin(&target->framebuffer[pixelID + ox + oy * int(target->width)], pixel);
					}
					

				}
			}
			
		}
	});
}


// see 
// * http://paulbourke.net/geometry/polygonise/
// * http://paulbourke.net/geometry/polygonise/marchingsource.cpp
// * https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-1-generating-complex-procedural-terrains-using-gpu
Triangles* marchingCubes(int gridSize, uint32_t* voxelGrid){

	auto grid = cg::this_grid();

	int maxTriangles = 200'000;
	Triangles* triangles = allocator->alloc<Triangles*>(sizeof(Triangles));
	triangles->positions = allocator->alloc<float3*>(3 * maxTriangles * sizeof(float3));
	triangles->uvs = allocator->alloc<float2*>(2 * maxTriangles * sizeof(float2));
	triangles->colors = allocator->alloc<uint32_t*>(maxTriangles * sizeof(float3));
	triangles->numTriangles = 0;
	float3 boxSize = gridMax - gridMin;

	grid.sync();

	processRange(0, gridSize * gridSize * gridSize, [&](int voxelIndex){

		int x = voxelIndex % gridSize;
		int y = (voxelIndex % (gridSize * gridSize)) / gridSize;
		int z = voxelIndex / (gridSize * gridSize);

		if(x >= gridSize - 1) return;
		if(y >= gridSize - 1) return;
		if(z >= gridSize - 1) return;


		auto to1DIndex = [&](int x, int y, int z){
			return x + y * gridSize + z * gridSize * gridSize;
		};

		auto fromMCIndex = [&](int index) -> int3{
			if(index == 0) return int3{0, 0, 0};
			if(index == 1) return int3{1, 0, 0};
			if(index == 2) return int3{1, 1, 0};
			if(index == 3) return int3{0, 1, 0};
			if(index == 4) return int3{0, 0, 1};
			if(index == 5) return int3{1, 0, 1};
			if(index == 6) return int3{1, 1, 1};
			if(index == 7) return int3{0, 1, 1};
		};

		auto getValue = [&](int mcIndex){
			int3 index3D = fromMCIndex(mcIndex);
			int index = to1DIndex(x + index3D.x, y + index3D.y, z + index3D.z);
			float value = voxelGrid[index];

			if(value == 0.0f){

			}else{
				value = 1.0f;
			}

			return value;
		};

		auto getXYZ = [&](int mcIndex){

			int3 xyz_i = fromMCIndex(mcIndex);

			float3 xyz;
			xyz.x = boxSize.x * float(x + xyz_i.x) / fGridSize + gridMin.x;
			xyz.y = boxSize.y * float(y + xyz_i.y) / fGridSize + gridMin.y;
			xyz.z = boxSize.z * float(z + xyz_i.z) / fGridSize + gridMin.z;

			return xyz;
		};

		uint32_t cubeindex = 0;
		
		if(getValue(0) > 0.0f) cubeindex |=   1;
		if(getValue(1) > 0.0f) cubeindex |=   2;
		if(getValue(2) > 0.0f) cubeindex |=   4;
		if(getValue(3) > 0.0f) cubeindex |=   8;
		if(getValue(4) > 0.0f) cubeindex |=  16;
		if(getValue(5) > 0.0f) cubeindex |=  32;
		if(getValue(6) > 0.0f) cubeindex |=  64;
		if(getValue(7) > 0.0f) cubeindex |= 128;

		if (edgeTable[cubeindex] == 0) return;

		// list of vertices from isosurface-cube intersections
		float3 vertices[12];
		const int* et = edgeTable;

		// voxels are either "inside" (=0) or "outside" (>0) the isosurce
		auto getVertex = [](float3 p0, float3 p1, float value_0, float value_1){
			if(value_0 > 0.0f) return p0;
			if(value_1 > 0.0f) return p1;

			return p0;
		};

		if (et[cubeindex] &    1) vertices[ 0] = getVertex(getXYZ(0), getXYZ(1), getValue(0), getValue(1));
		if (et[cubeindex] &    2) vertices[ 1] = getVertex(getXYZ(1), getXYZ(2), getValue(1), getValue(2));
		if (et[cubeindex] &    4) vertices[ 2] = getVertex(getXYZ(2), getXYZ(3), getValue(2), getValue(3));
		if (et[cubeindex] &    8) vertices[ 3] = getVertex(getXYZ(3), getXYZ(0), getValue(3), getValue(0));
		if (et[cubeindex] &   16) vertices[ 4] = getVertex(getXYZ(4), getXYZ(5), getValue(4), getValue(5));
		if (et[cubeindex] &   32) vertices[ 5] = getVertex(getXYZ(5), getXYZ(6), getValue(5), getValue(6));
		if (et[cubeindex] &   64) vertices[ 6] = getVertex(getXYZ(6), getXYZ(7), getValue(6), getValue(7));
		if (et[cubeindex] &  128) vertices[ 7] = getVertex(getXYZ(7), getXYZ(4), getValue(7), getValue(4));
		if (et[cubeindex] &  256) vertices[ 8] = getVertex(getXYZ(0), getXYZ(4), getValue(0), getValue(4));
		if (et[cubeindex] &  512) vertices[ 9] = getVertex(getXYZ(1), getXYZ(5), getValue(1), getValue(5));
		if (et[cubeindex] & 1024) vertices[10] = getVertex(getXYZ(2), getXYZ(6), getValue(2), getValue(6));
		if (et[cubeindex] & 2048) vertices[11] = getVertex(getXYZ(3), getXYZ(7), getValue(3), getValue(7));

		// create up to 5 triangles per cube (http://paulbourke.net/geometry/polygonise/marchingsource.cpp)
		for (int i = 0; i < 5; i++){

			if(triTable[cubeindex][3 * i] < 0) break;

			int index = atomicAdd(&triangles->numTriangles, 1);
			
			if(index >= maxTriangles) break;

			triangles->positions[3 * index + 2] = vertices[triTable[cubeindex][3 * i + 0]];
			triangles->positions[3 * index + 1] = vertices[triTable[cubeindex][3 * i + 1]];
			triangles->positions[3 * index + 0] = vertices[triTable[cubeindex][3 * i + 2]];

			triangles->colors[3 * index + 2] = (2 * x) | (2 * y) << 8 | (2 * z) << 16;
			triangles->colors[3 * index + 1] = (2 * x) | (2 * y) << 8 | (2 * z) << 16;
			triangles->colors[3 * index + 0] = (2 * x) | (2 * y) << 8 | (2 * z) << 16;
		}

	});

	return triangles;
}

// rayPlaneIntersection from three.js
// https://github.com/mrdoob/three.js/blob/13a5874eabfe45fb8459e268e9786a20054bb6a2/src/math/Ray.js#L256
// LICENSE: https://github.com/mrdoob/three.js/blob/13a5874eabfe45fb8459e268e9786a20054bb6a2/LICENSE#L1-L21
//          (MIT)
float rayPlaneIntersection(float3 origin, float3 dir, float3 normal, float d){
	
	float denominator = dot(normal, dir);

	if(denominator == 0){
		return 0.0f;
	}

	float t = -(dot(origin, normal) + d) / denominator;

	return t;
};

void drawSkybox(
	mat4 proj, mat4 view, 
	mat4 proj_inv, mat4 view_inv, 
	uint64_t* framebuffer,
	float width, float height,
	const Skybox& skybox
){ 
	auto projToWorld = [&](float4 pos){
		float4 viewspace = proj_inv * pos;
		viewspace = viewspace / viewspace.w;

		return view_inv * viewspace;
	};

	float4 origin_projspace = proj * float4{0.0f, 0.0f, 0.0f, 1.0f};
	float4 dir_00_projspace = float4{-1.0f, -1.0f, 0.0f, 1.0f};
	float4 dir_01_projspace = float4{-1.0f,  1.0f, 0.0f, 1.0f};
	float4 dir_10_projspace = float4{ 1.0f, -1.0f, 0.0f, 1.0f};
	float4 dir_11_projspace = float4{ 1.0f,  1.0f, 0.0f, 1.0f};

	float4 origin_worldspace = projToWorld(origin_projspace);
	float4 dir_00_worldspace = projToWorld(dir_00_projspace);
	float4 dir_01_worldspace = projToWorld(dir_01_projspace);
	float4 dir_10_worldspace = projToWorld(dir_10_projspace);
	float4 dir_11_worldspace = projToWorld(dir_11_projspace);

	processRange(width * height, [&](int pixelID){
		
		int x = pixelID % int(width);
		int y = pixelID / int(width);

		float u = float(x) / width;
		float v = float(y) / height;

		float A_00 = (1.0f - u) * (1.0f - v);
		float A_01 = (1.0f - u) *         v;
		float A_10 =         u  * (1.0f - v);
		float A_11 =         u  *         v;

		float3 dir = make_float3(
			A_00 * dir_00_worldspace + 
			A_01 * dir_01_worldspace + 
			A_10 * dir_10_worldspace + 
			A_11 * dir_11_worldspace - origin_worldspace);
		dir = normalize(dir);
		// float3 origin = make_float3(origin_worldspace);
		float3 origin = {0.0f, 0.0f, 0.0f};

		float3 planes[6] = {
			float3{ 1.0f,  0.0f,  0.0f},
			float3{ 0.0f,  0.0f,  1.0f}, 
			float3{ 0.0f,  1.0f,  0.0f},
			float3{-1.0f,  0.0f,  0.0f},
			float3{ 0.0f,  0.0f, -1.0f},
			float3{ 0.0f, -1.0f,  0.0f},
		};

		// skybox:
		// x: left-right
		// y: bottom-top
		// z: front-back
		int planeIndex = 2 + 3;
		float boxsize = 10.0f;

		float closest_t = Infinity;
		int closest_plane = 0;

		// for(int i = 0; i < 6; i++)
		for(int i : {0, 1, 2, 3, 4, 5})
		{
			float t = rayPlaneIntersection(origin, dir, planes[i], boxsize);

			if(t > 0.0f && t < closest_t){
				closest_t = t;
				closest_plane = i;
			}
		}

		float t = closest_t;
		float3 I = t * dir;
		float2 box_uv;

		if(closest_plane == 0){
			box_uv = {
				0.5f * (I.y / boxsize) + 0.5f, 
				0.5f * (I.z / boxsize) + 0.5f
			};
		}else if(closest_plane == 1){
			box_uv = {
				0.5f * (I.x / boxsize) + 0.5f, 
				0.5f * (I.y / boxsize) + 0.5f
			};
		}else if(closest_plane == 2){
			box_uv = {
				1.0f - (0.5f * (I.x / boxsize) + 0.5f), 
				0.5f * (I.z / boxsize) + 0.5f
			};
		}else if(closest_plane == 3){
			box_uv = {
				1.0f - (0.5f * (I.y / boxsize) + 0.5f), 
				0.5f * (I.z / boxsize) + 0.5f
			};
		}else if(closest_plane == 4){
			box_uv = {
				0.5f * (I.x / boxsize) + 0.5f, 
				1.0f - (0.5f * (I.y / boxsize) + 0.5f)
			};
		}else if(closest_plane == 5){
			box_uv = {
				0.5f * (I.x / boxsize) + 0.5f, 
				0.5f * (I.z / boxsize) + 0.5f
			};
		}

		if(t < 0.0f) return;
		if(box_uv.x > 1.0f) return;
		if(box_uv.x < 0.0f) return;
		if(box_uv.y > 1.0f) return;
		if(box_uv.y < 0.0f) return;

		uint32_t color;
		uint8_t* rgba = (uint8_t*)&color;

		uint8_t* textureData = skybox.textures[closest_plane];
		int tx = clamp(box_uv.x * skybox.width, 0.0f, skybox.width - 1.0f);
		int ty = clamp((1.0f - box_uv.y) * skybox.height, 0.0f, skybox.height - 1.0f);
		int texelIndex = tx + ty * skybox.width;

		rgba[0] = textureData[4 * texelIndex + 0];
		rgba[1] = textureData[4 * texelIndex + 1];
		rgba[2] = textureData[4 * texelIndex + 2];

		float depth = 100000000000.0f;
		uint64_t idepth = *((uint32_t*)&depth);
		uint64_t pixel = idepth << 32 | color;
		
		atomicMin(&framebuffer[pixelID], pixel);
	});


}

extern "C" __global__
void kernel(
	const Uniforms _uniforms,
	uint32_t* buffer,
	cudaSurfaceObject_t gl_colorbuffer_main,
	cudaSurfaceObject_t gl_colorbuffer_vr_left,
	cudaSurfaceObject_t gl_colorbuffer_vr_right,
	uint32_t numTriangles,
	float3* positions,
	float2* uvs,
	uint32_t* colors,
	uint32_t* textureData,
	const Skybox skybox
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime_start));

	uniforms = _uniforms;

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	uint32_t* voxelGrid = allocator->alloc<uint32_t*>(numCells * sizeof(uint32_t));

	// clear/initialize voxel grid, if necessary. Note that this is done every frame.
	// processRange(0, numCells, [&](int voxelIndex){
	// 	int x = voxelIndex % gridSize;
	// 	int y = voxelIndex % (gridSize * gridSize) / gridSize;
	// 	int z = voxelIndex / (gridSize * gridSize);

	// 	float fx = 2.0f * float(x) / fGridSize - 1.0f;
	// 	float fy = 2.0f * float(y) / fGridSize - 1.0f;
	// 	float fz = 2.0f * float(z) / fGridSize - 1.0f;

	// 	// clear and make sphere and ground plane
	// 	// if(fx * fx + fy * fy + fz * fz < 0.1f){
	// 	// 	voxelGrid[voxelIndex] = 123;
	// 	// }else if(x > 10 && x < gridSize - 10 && z < 4){
	// 	// 	voxelGrid[voxelIndex] = 123;

	// 	// }else{
	// 	// 	voxelGrid[voxelIndex] = 0;
	// 	// }

	// 	// clear everything
	// 	voxelGrid[voxelIndex] = 0;
	// });


	// allocate framebuffer memory
	int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
	uint64_t* framebuffer = allocator->alloc<uint64_t*>(framebufferSize);
	uint64_t* fb_vr_left = allocator->alloc<uint64_t*>(int(uniforms.vr_left_width) * int(uniforms.vr_left_height) * sizeof(uint64_t));
	uint64_t* fb_vr_right = allocator->alloc<uint64_t*>(int(uniforms.vr_right_width) * int(uniforms.vr_right_height) * sizeof(uint64_t));
	
	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
	});

	if(uniforms.vrEnabled){
		processRange(0, uniforms.vr_left_width * uniforms.vr_left_height, [&](int pixelIndex){
			fb_vr_left[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
		});

		processRange(0, uniforms.vr_right_width * uniforms.vr_right_height, [&](int pixelIndex){
			fb_vr_right[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
		});
	}
	
	grid.sync();

	{ // generate and draw a ground plane
		int cells = 50;
		int numTriangles     = cells * cells * 2;
		int numVertices      = 3 * numTriangles;
		Triangles* triangles = allocator->alloc<Triangles*>(sizeof(Triangles));
		triangles->positions = allocator->alloc<float3*  >(sizeof(float3) * numVertices);
		triangles->uvs       = allocator->alloc<float2*  >(sizeof(float2) * numVertices);
		triangles->colors    = allocator->alloc<uint32_t*>(sizeof(uint32_t) * numVertices);

		triangles->numTriangles = numTriangles;
		
		processRange(0, cells * cells, [&](int cellIndex){

			int cx = cellIndex % cells;
			int cy = cellIndex / cells;

			float u0 = float(cx + 0) / float(cells);
			float v0 = float(cy + 0) / float(cells);
			float u1 = float(cx + 1) / float(cells);
			float v1 = float(cy + 1) / float(cells);

			int offset = 6 * cellIndex;

			uint32_t color = 0;
			uint8_t* rgb = (uint8_t*)&color;
			rgb[0] = 255.0f * u0;
			rgb[1] = 255.0f * v0;
			rgb[2] = 0;

			float s = 10.0f;
			float height = -0.5f;
			
			triangles->positions[offset + 0] = {s * u0 - s * 0.5f, s * v0 - s * 0.5f, height};
			triangles->positions[offset + 1] = {s * u1 - s * 0.5f, s * v0 - s * 0.5f, height};
			triangles->positions[offset + 2] = {s * u1 - s * 0.5f, s * v1 - s * 0.5f, height};
			triangles->positions[offset + 3] = {s * u0 - s * 0.5f, s * v0 - s * 0.5f, height};
			triangles->positions[offset + 4] = {s * u1 - s * 0.5f, s * v1 - s * 0.5f, height};
			triangles->positions[offset + 5] = {s * u0 - s * 0.5f, s * v1 - s * 0.5f, height};

			triangles->uvs[offset + 0] = {u0, v0};
			triangles->uvs[offset + 1] = {u1, v0};
			triangles->uvs[offset + 2] = {u1, v1};
			triangles->uvs[offset + 3] = {u0, v0};
			triangles->uvs[offset + 4] = {u1, v1};
			triangles->uvs[offset + 5] = {u0, v1};
		});

		Texture texture;
		texture.width = 512;
		texture.height = 512;
		texture.data = allocator->alloc<uint32_t*>(4 * texture.width * texture.height);

		grid.sync();

		processRange(0, texture.width * texture.height, [&](int index){
			
			int x = index % texture.width;
			int y = index / texture.width;

			uint32_t color;
			uint8_t* rgba = (uint8_t*)&color;

			if((x % 16) == 0 || (y % 16) == 0){
				color = 0x00000000;
			}else{
				color = 0x00aaaaaa;
			}

			texture.data[index] = color;

		});

		grid.sync();

		
		RasterizationSettings settings;
		settings.texture = nullptr;
		settings.colorMode = COLORMODE_TRIANGLE_ID;
		settings.world = mat4::identity();
		settings.view = uniforms.view;
		settings.proj = uniforms.proj;
		settings.width = uniforms.width;
		settings.height = uniforms.height;
		settings.texture = &texture;

		// when drawing time, due to normalization, everything needs to be colored by time
		// lets draw the ground with non-normalized time as well for consistency
		if(uniforms.colorMode == COLORMODE_TIME){
			settings.colorMode = COLORMODE_TIME_NORMALIZED;
		}else if(uniforms.colorMode == COLORMODE_TIME_NORMALIZED){
			settings.colorMode = COLORMODE_TIME_NORMALIZED;
		}

		settings.colorMode = COLORMODE_TEXTURE;

		// rasterizeTriangles(triangles, framebuffer, settings);

		if(uniforms.vrEnabled){
			settings.view = uniforms.vr_left_view;
			settings.proj = uniforms.vr_left_proj;
			settings.width = uniforms.vr_left_width;
			settings.height = uniforms.vr_left_height;
			rasterizeTriangles(triangles, fb_vr_left, settings);

			grid.sync();

			settings.view = uniforms.vr_right_view;
			settings.proj = uniforms.vr_right_proj;
			settings.width = uniforms.vr_right_width;
			settings.height = uniforms.vr_right_height;
			rasterizeTriangles(triangles, fb_vr_right, settings);
		}else{
			settings.view = uniforms.view;
			settings.proj = uniforms.proj;
			settings.width = uniforms.width;
			settings.height = uniforms.height;

			rasterizeTriangles(triangles, framebuffer, settings);
		}
	}

	grid.sync();

	// VOXEL PAINTING / CONTROLLER INPUT
	if(grid.thread_rank() == 0){

		// see openvr.h: EVRButtonId and ButtonMaskFromId
		uint64_t triggerMask = 1ull << 33ull;
		bool leftTriggerButtonDown = (uniforms.vr_left_controller_state.buttonPressedMask & triggerMask) != 0;
		bool rightTriggerButtonDown = (uniforms.vr_right_controller_state.buttonPressedMask & triggerMask) != 0;
		bool isTriggerDown = leftTriggerButtonDown || rightTriggerButtonDown;

		// printf("%llu \n", leftTriggerButtonDown);
		
		if(rightTriggerButtonDown){
			float brushRadius = 3.0;
			int iBrushRadius = ceil(brushRadius);
			float brushRadius2 = brushRadius * brushRadius;
			
			mat4 rot = mat4::rotate(0.5f * PI, {1.0f, 0.0f, 0.0f}).transpose();
			float4 pos = rot * uniforms.vr_right_controller_pose.transpose() * float4{0.0f, 0.0f, 0.0f, 1.0f};
			float3 boxSize = gridMax - gridMin;

			float fx = gridSize * (pos.x - gridMin.x) / boxSize.x;
			float fy = gridSize * (pos.y - gridMin.y) / boxSize.y;
			float fz = gridSize * (pos.z - gridMin.z) / boxSize.z;

			for(int ox = -iBrushRadius; ox <= brushRadius; ox++)
			for(int oy = -iBrushRadius; oy <= brushRadius; oy++)
			for(int oz = -iBrushRadius; oz <= brushRadius; oz++)
			{

				int ix = fx + float(ox);
				int iy = fy + float(oy);
				int iz = fz + float(oz);

				if(ix < 0 || ix >= gridSize) continue;
				if(iy < 0 || iy >= gridSize) continue;
				if(iz < 0 || iz >= gridSize) continue;

				int voxelIndex = ix + iy * gridSize + iz * gridSize * gridSize;

				float vcx = float(ix) + 0.5f;
				float vcy = float(iy) + 0.5f;
				float vcz = float(iz) + 0.5f;
				float dx = vcx - fx;
				float dy = vcy - fy;
				float dz = vcz - fz;
				float dd = dx * dx + dy * dy + dz * dz;

				if(dd < brushRadius2){
					voxelGrid[voxelIndex] = 123;
				}
			}
		}

		if(leftTriggerButtonDown){
			float brushRadius = 5.0;
			int iBrushRadius = ceil(brushRadius);
			float brushRadius2 = brushRadius * brushRadius;
			
			mat4 rot = mat4::rotate(0.5f * PI, {1.0f, 0.0f, 0.0f}).transpose();
			float4 pos = rot * uniforms.vr_left_controller_pose.transpose() * float4{0.0f, 0.0f, 0.0f, 1.0f};
			float3 boxSize = gridMax - gridMin;

			float fx = gridSize * (pos.x - gridMin.x) / boxSize.x;
			float fy = gridSize * (pos.y - gridMin.y) / boxSize.y;
			float fz = gridSize * (pos.z - gridMin.z) / boxSize.z;

			for(int ox = -iBrushRadius; ox <= brushRadius; ox++)
			for(int oy = -iBrushRadius; oy <= brushRadius; oy++)
			for(int oz = -iBrushRadius; oz <= brushRadius; oz++)
			{

				int ix = fx + float(ox);
				int iy = fy + float(oy);
				int iz = fz + float(oz);

				if(ix < 0 || ix >= gridSize) continue;
				if(iy < 0 || iy >= gridSize) continue;
				if(iz < 0 || iz >= gridSize) continue;

				int voxelIndex = ix + iy * gridSize + iz * gridSize * gridSize;

				float vcx = float(ix) + 0.5f;
				float vcy = float(iy) + 0.5f;
				float vcz = float(iz) + 0.5f;
				float dx = vcx - fx;
				float dy = vcy - fy;
				float dz = vcz - fz;
				float dd = dx * dx + dy * dy + dz * dz;

				if(dd < brushRadius2){
					voxelGrid[voxelIndex] = 0;
				}
			}
		}
	}

	grid.sync();

	{ // DRAW VOXEL GRID
		
		Triangles* triangles = marchingCubes(gridSize, voxelGrid);

		grid.sync();
		
		RasterizationSettings settings;
		settings.texture = nullptr;
		settings.colorMode = COLORMODE_VERTEXCOLOR;
		settings.world = mat4::identity();
		settings.view = uniforms.view;
		settings.proj = uniforms.proj;
		settings.width = uniforms.width;
		settings.height = uniforms.height;

		if(uniforms.vrEnabled){
			settings.view = uniforms.vr_left_view;
			settings.proj = uniforms.vr_left_proj;
			settings.width = uniforms.vr_left_width;
			settings.height = uniforms.vr_left_height;
			rasterizeTriangles(triangles, fb_vr_left, settings);

			grid.sync();

			settings.view = uniforms.vr_right_view;
			settings.proj = uniforms.vr_right_proj;
			settings.width = uniforms.vr_right_width;
			settings.height = uniforms.vr_right_height;
			rasterizeTriangles(triangles, fb_vr_right, settings);
		}else{
			settings.view = uniforms.view;
			settings.proj = uniforms.proj;
			settings.width = uniforms.width;
			settings.height = uniforms.height;

			rasterizeTriangles(triangles, framebuffer, settings);
		}
	}

	// grid.sync();

	// {

	// }

	grid.sync();

	{ // DRAW CONTROLLERS
		Triangles* triangles = allocator->alloc<Triangles*>(sizeof(Triangles));
		triangles->numTriangles = numTriangles;

		triangles->positions = positions;
		triangles->uvs = uvs;
		triangles->colors = colors;

		Texture texture;
		texture.width  = 1024;
		texture.height = 1024;
		texture.data   = textureData;

		RasterizationSettings settings;
		settings.texture = &texture;
		settings.colorMode = uniforms.colorMode;
		settings.world = uniforms.world;

		{
			float s = 0.8f;
			mat4 rot = mat4::rotate(0.5f * PI, {1.0f, 0.0f, 0.0f}).transpose();
			mat4 translate = mat4::translate(0.0f, 0.0f, 0.0f);
			mat4 scale = mat4::scale(s, s, s);
			mat4 wiggle = mat4::rotate(cos(5.0f * uniforms.time) * 0.1f, {0.0f, 1.0f, 0.0f}).transpose();
			mat4 wiggle_yaw = mat4::rotate(cos(5.0f * uniforms.time) * 0.1f, {0.0f, 0.0f, 1.0f}).transpose();
			
			settings.world = translate * wiggle * wiggle_yaw * rot * scale;

			
			if(uniforms.vrEnabled){

				float sController = 0.05f;
				if(uniforms.vr_left_controller_active){
					settings.world = rot * uniforms.vr_left_controller_pose.transpose() 
						* mat4::scale(sController, sController, sController);

					settings.view = uniforms.vr_left_view;
					settings.proj = uniforms.vr_left_proj;
					settings.width = uniforms.vr_left_width;
					settings.height = uniforms.vr_left_height;
					rasterizeTriangles(triangles, fb_vr_left, settings);

					grid.sync();

					settings.view = uniforms.vr_right_view;
					settings.proj = uniforms.vr_right_proj;
					settings.width = uniforms.vr_right_width;
					settings.height = uniforms.vr_right_height;
					rasterizeTriangles(triangles, fb_vr_right, settings);
				}

				if(uniforms.vr_right_controller_active){
					settings.world = rot * uniforms.vr_right_controller_pose.transpose() 
						* mat4::scale(sController, sController, sController);

					settings.view = uniforms.vr_left_view;
					settings.proj = uniforms.vr_left_proj;
					settings.width = uniforms.vr_left_width;
					settings.height = uniforms.vr_left_height;
					rasterizeTriangles(triangles, fb_vr_left, settings);

					grid.sync();

					settings.view = uniforms.vr_right_view;
					settings.proj = uniforms.vr_right_proj;
					settings.width = uniforms.vr_right_width;
					settings.height = uniforms.vr_right_height;
					rasterizeTriangles(triangles, fb_vr_right, settings);
				}
			}else{

				// auto projToWorld = [&](float4 pos){
				// 	float4 viewspace = uniforms.proj_inv * pos;
				// 	viewspace = viewspace / viewspace.w;

				// 	return uniforms.view_inv * viewspace;
				// };

				// float4 origin_projspace = uniforms.proj * float4{0.0f, 0.0f, 0.0f, 1.0f};
				// float4 dir_00_projspace = float4{-1.0f, -1.0f, 0.0f, 1.0f};
				// float4 dir_01_projspace = float4{-1.0f,  1.0f, 0.0f, 1.0f};
				// float4 dir_10_projspace = float4{ 1.0f, -1.0f, 0.0f, 1.0f};
				// float4 dir_11_projspace = float4{ 1.0f,  1.0f, 0.0f, 1.0f};

				// float4 origin_worldspace = projToWorld(origin_projspace);
				// float4 dir_00_worldspace = projToWorld(dir_00_projspace);
				// float4 dir_01_worldspace = projToWorld(dir_01_projspace);
				// float4 dir_10_worldspace = projToWorld(dir_10_projspace);
				// float4 dir_11_worldspace = projToWorld(dir_11_projspace);


				// // int x = pixelID % int(uniforms.width);
				// // int y = pixelID / int(uniforms.width);
				// int x = 700;
				// int y = 700;

				// float u = float(x) / uniforms.width;
				// float v = float(y) / uniforms.height;

				// float A_00 = (1.0f - u) * (1.0f - v);
				// float A_01 = (1.0f - u) *         v;
				// float A_10 =         u  * (1.0f - v);
				// float A_11 =         u  *         v;

				// float3 dir = make_float3(
				// 	A_00 * dir_00_worldspace + 
				// 	A_01 * dir_01_worldspace + 
				// 	A_10 * dir_10_worldspace + 
				// 	A_11 * dir_11_worldspace - origin_worldspace);
				// // dir = dir - origin_worldspace;
				// float3 origin = make_float3(origin_worldspace);



				// float4 tmp;
				// tmp.x = origin.x + dir.x * 10.0f;
				// tmp.y = origin.y + dir.y * 10.0f;
				// tmp.z = origin.z + dir.z * 10.0f;
				// tmp.w = 1.0f;
				// // // float4 tmp = dir_10_worldspace;


				// float s = 0.08f;
				// mat4 rot = mat4::rotate(0.5f * PI, {1.0f, 0.0f, 0.0f}).transpose();
				// // mat4 translate = mat4::translate(0.0f, 0.0f, 0.0f);
				// mat4 translate = mat4::translate(tmp.x, tmp.y, tmp.z);
				// mat4 scale = mat4::scale(s, s, s);
				// mat4 wiggle = mat4::rotate(cos(5.0f * uniforms.time) * 0.1f, {0.0f, 1.0f, 0.0f}).transpose();
				// mat4 wiggle_yaw = mat4::rotate(cos(5.0f * uniforms.time) * 0.1f, {0.0f, 0.0f, 1.0f}).transpose();
				
				// settings.world = translate * wiggle * wiggle_yaw * rot * scale;

				settings.view = uniforms.view;
				settings.proj = uniforms.proj;
				settings.width = uniforms.width;
				settings.height = uniforms.height;

				rasterizeTriangles(triangles, framebuffer, settings);
			}

			grid.sync();
		}
	}


	grid.sync();

	if(uniforms.vrEnabled){

		// TODO
		// drawSkybox(
		// 	uniforms.vr_left_proj, uniforms.vr_left_view, 
		// 	uniforms.vr_left_proj_inv, uniforms.vr_left_view_inv, 
		// 	framebuffer, 
		// 	uniforms.vr_left_width, uniforms.vr_left_height, 
		// 	skybox
		// );

		// drawSkybox(
		// 	uniforms.vr_right_proj, uniforms.vr_right_view, 
		// 	uniforms.vr_right_proj_inv, uniforms.vr_right_view_inv, 
		// 	framebuffer, 
		// 	uniforms.vr_right_width, uniforms.vr_right_height, 
		// 	skybox
		// );
		
		// if(grid.thread_rank() == 0){
		// 	mat4 mat = uniforms.vr_right_proj_inv;
		// 	printf("===========\n");
		// 	printf("%5.1f, %5.1f, %5.1f, %5.1f \n", mat[0].x, mat[0].y, mat[0].z, mat[0].w);
		// 	printf("%5.1f, %5.1f, %5.1f, %5.1f \n", mat[1].x, mat[1].y, mat[1].z, mat[1].w);
		// 	printf("%5.1f, %5.1f, %5.1f, %5.1f \n", mat[2].x, mat[2].y, mat[2].z, mat[2].w);
		// 	printf("%5.1f, %5.1f, %5.1f, %5.1f \n", mat[3].x, mat[3].y, mat[3].z, mat[3].w);
		// }

	}else{
		drawSkybox(
			uniforms.proj, uniforms.view, 
			uniforms.proj_inv, uniforms.view_inv, 
			framebuffer, 
			uniforms.width, uniforms.height, 
			skybox
		);
	}

	

	// { // SKYBOX
	// 	auto projToWorld = [&](float4 pos){
	// 		float4 viewspace = uniforms.proj_inv * pos;
	// 		viewspace = viewspace / viewspace.w;

	// 		return uniforms.view_inv * viewspace;
	// 	};

	// 	// rayPlaneIntersection from three.js
	// 	// https://github.com/mrdoob/three.js/blob/13a5874eabfe45fb8459e268e9786a20054bb6a2/src/math/Ray.js#L256
	// 	// LICENSE: https://github.com/mrdoob/three.js/blob/13a5874eabfe45fb8459e268e9786a20054bb6a2/LICENSE#L1-L21
	// 	//          (MIT)
	// 	auto rayPlaneIntersection = [&](float3 origin, float3 dir, float3 normal, float d){
			
	// 		float denominator = dot(normal, dir);

	// 		if(denominator == 0){
	// 			return 0.0f;
	// 		}

	// 		float t = -(dot(origin, normal) + d) / denominator;

	// 		return t;
	// 	};

	// 	float4 origin_projspace = uniforms.proj * float4{0.0f, 0.0f, 0.0f, 1.0f};
	// 	float4 dir_00_projspace = float4{-1.0f, -1.0f, 0.0f, 1.0f};
	// 	float4 dir_01_projspace = float4{-1.0f,  1.0f, 0.0f, 1.0f};
	// 	float4 dir_10_projspace = float4{ 1.0f, -1.0f, 0.0f, 1.0f};
	// 	float4 dir_11_projspace = float4{ 1.0f,  1.0f, 0.0f, 1.0f};

	// 	float4 origin_worldspace = projToWorld(origin_projspace);
	// 	float4 dir_00_worldspace = projToWorld(dir_00_projspace);
	// 	float4 dir_01_worldspace = projToWorld(dir_01_projspace);
	// 	float4 dir_10_worldspace = projToWorld(dir_10_projspace);
	// 	float4 dir_11_worldspace = projToWorld(dir_11_projspace);

	// 	processRange(uniforms.width * uniforms.height, [&](int pixelID){
			
	// 		int x = pixelID % int(uniforms.width);
	// 		int y = pixelID / int(uniforms.width);

	// 		float u = float(x) / uniforms.width;
	// 		float v = float(y) / uniforms.height;

	// 		float A_00 = (1.0f - u) * (1.0f - v);
	// 		float A_01 = (1.0f - u) *         v;
	// 		float A_10 =         u  * (1.0f - v);
	// 		float A_11 =         u  *         v;

	// 		float3 dir = make_float3(
	// 			A_00 * dir_00_worldspace + 
	// 			A_01 * dir_01_worldspace + 
	// 			A_10 * dir_10_worldspace + 
	// 			A_11 * dir_11_worldspace - origin_worldspace);
	// 		dir = normalize(dir);
	// 		// float3 origin = make_float3(origin_worldspace);
	// 		float3 origin = {0.0f, 0.0f, 0.0f};

	// 		float3 planes[6] = {
	// 			float3{ 1.0f,  0.0f,  0.0f},
	// 			float3{ 0.0f,  0.0f,  1.0f}, 
	// 			float3{ 0.0f,  1.0f,  0.0f},
	// 			float3{-1.0f,  0.0f,  0.0f},
	// 			float3{ 0.0f,  0.0f, -1.0f},
	// 			float3{ 0.0f, -1.0f,  0.0f},
	// 		};

	// 		// skybox:
	// 		// x: left-right
	// 		// y: bottom-top
	// 		// z: front-back
	// 		int planeIndex = 2 + 3;
	// 		float boxsize = 10.0f;

	// 		float closest_t = Infinity;
	// 		int closest_plane = 0;

	// 		// for(int i = 0; i < 6; i++)
	// 		for(int i : {0, 1, 2, 3, 4, 5})
	// 		{
	// 			float t = rayPlaneIntersection(origin, dir, planes[i], boxsize);

	// 			if(t > 0.0f && t < closest_t){
	// 				closest_t = t;
	// 				closest_plane = i;
	// 			}
	// 		}

	// 		float t = closest_t;
	// 		float3 I = t * dir;
	// 		float2 box_uv;

	// 		if(closest_plane == 0){
	// 			box_uv = {
	// 				0.5f * (I.y / boxsize) + 0.5f, 
	// 				0.5f * (I.z / boxsize) + 0.5f
	// 			};
	// 		}else if(closest_plane == 1){
	// 			box_uv = {
	// 				0.5f * (I.x / boxsize) + 0.5f, 
	// 				0.5f * (I.y / boxsize) + 0.5f
	// 			};
	// 		}else if(closest_plane == 2){
	// 			box_uv = {
	// 				1.0f - (0.5f * (I.x / boxsize) + 0.5f), 
	// 				0.5f * (I.z / boxsize) + 0.5f
	// 			};
	// 		}else if(closest_plane == 3){
	// 			box_uv = {
	// 				1.0f - (0.5f * (I.y / boxsize) + 0.5f), 
	// 				0.5f * (I.z / boxsize) + 0.5f
	// 			};
	// 		}else if(closest_plane == 4){
	// 			box_uv = {
	// 				0.5f * (I.x / boxsize) + 0.5f, 
	// 				1.0f - (0.5f * (I.y / boxsize) + 0.5f)
	// 			};
	// 		}else if(closest_plane == 5){
	// 			box_uv = {
	// 				0.5f * (I.x / boxsize) + 0.5f, 
	// 				0.5f * (I.z / boxsize) + 0.5f
	// 			};
	// 		}

	// 		if(t < 0.0f) return;
	// 		if(box_uv.x > 1.0f) return;
	// 		if(box_uv.x < 0.0f) return;
	// 		if(box_uv.y > 1.0f) return;
	// 		if(box_uv.y < 0.0f) return;

	// 		uint32_t color;
	// 		uint8_t* rgba = (uint8_t*)&color;

	// 		uint8_t* textureData = skybox.textures[closest_plane];
	// 		int tx = clamp(box_uv.x * skybox.width, 0.0f, skybox.width - 1.0f);
	// 		int ty = clamp((1.0f - box_uv.y) * skybox.height, 0.0f, skybox.height - 1.0f);
	// 		int texelIndex = tx + ty * skybox.width;

	// 		rgba[0] = textureData[4 * texelIndex + 0];
	// 		rgba[1] = textureData[4 * texelIndex + 1];
	// 		rgba[2] = textureData[4 * texelIndex + 2];

	// 		float depth = 100000000000.0f;
	// 		uint64_t idepth = *((uint32_t*)&depth);
	// 		uint64_t pixel = idepth << 32 | color;
			
	// 		atomicMin(&framebuffer[pixelID], pixel);
	// 	});


	// }

	grid.sync();

	// draws voxels as point sprites
	// now using marching cubes instead
	// { // DRAW VOXELS
	// 	int numTargets;
	// 	RenderTarget targets[3];

	// 	if(uniforms.vrEnabled){
	// 		numTargets = 2;

	// 		targets[0].view = uniforms.vr_left_view;
	// 		targets[0].proj = uniforms.vr_left_proj;
	// 		targets[0].transform = uniforms.vr_left_proj * uniforms.vr_left_view;
	// 		targets[0].framebuffer = fb_vr_left;
	// 		targets[0].width = uniforms.vr_left_width;
	// 		targets[0].height = uniforms.vr_left_height;

	// 		targets[1].view = uniforms.vr_right_view;
	// 		targets[1].proj = uniforms.vr_right_proj;
	// 		targets[1].transform = uniforms.vr_right_proj * uniforms.vr_right_view;
	// 		targets[1].framebuffer = fb_vr_right;
	// 		targets[1].width = uniforms.vr_right_width;
	// 		targets[1].height = uniforms.vr_right_height;
	// 	}else{
	// 		numTargets = 1;

	// 		targets[0].view = uniforms.view;
	// 		targets[0].proj = uniforms.proj;
	// 		targets[0].transform = uniforms.proj * uniforms.view;
	// 		targets[0].framebuffer = framebuffer;
	// 		targets[0].width = uniforms.width;
	// 		targets[0].height = uniforms.height;
	// 	}

	// 	rasterizeVoxels(gridSize, numCells, voxelGrid, numTargets, targets);
	// }

	grid.sync();

	uint32_t& maxNanos = *allocator->alloc<uint32_t*>(4);

	grid.sync();

	// TRANSFER TO OPENGL TEXTURE
	if(uniforms.vrEnabled){
		
		// left
		processRange(0, uniforms.vr_left_width * uniforms.vr_left_height, [&](int pixelIndex){
			int x = pixelIndex % int(uniforms.vr_left_width);
			int y = pixelIndex / int(uniforms.vr_left_width);

			uint64_t encoded = fb_vr_left[pixelIndex];
			uint32_t color = encoded & 0xffffffffull;
			uint8_t* rgba = (uint8_t*)&color;
			uint32_t idepth = (encoded >> 32);
			float depth = *((float*)&idepth);

			if(EDL_ENABLED){
				float edlRadius = 2.0f;
				float edlStrength = 0.4f;
				float2 edlSamples[4] = {
					{-1.0f,  0.0f},
					{ 1.0f,  0.0f},
					{ 0.0f,  1.0f},
					{ 0.0f, -1.0f}
				};

				float sum = 0.0f;
				for(int i = 0; i < 4; i++){
					float2 samplePos = {
						x + edlSamples[i].x,
						y + edlSamples[i].y
					};

					int sx = clamp(samplePos.x, 0.0f, uniforms.vr_left_width - 1.0f);
					int sy = clamp(samplePos.y, 0.0f, uniforms.vr_left_height - 1.0f);
					int samplePixelIndex = sx + sy * uniforms.vr_left_width;

					uint64_t sampleEncoded = fb_vr_left[samplePixelIndex];
					uint32_t iSampledepth = (sampleEncoded >> 32);
					float sampleDepth = *((float*)&iSampledepth);

					sum += max(0.0, depth - sampleDepth);
				}

				float shade = exp(-sum * 300.0 * edlStrength);

				rgba[0] = float(rgba[0]) * shade;
				rgba[1] = float(rgba[1]) * shade;
				rgba[2] = float(rgba[2]) * shade;
			}

			surf2Dwrite(color, gl_colorbuffer_vr_left, x * 4, y);
		});

		// right
		processRange(0, uniforms.vr_right_width * uniforms.vr_right_height, [&](int pixelIndex){
			int x = pixelIndex % int(uniforms.vr_right_width);
			int y = pixelIndex / int(uniforms.vr_right_width);

			uint64_t encoded = fb_vr_right[pixelIndex];
			uint32_t color = encoded & 0xffffffffull;
			uint8_t* rgba = (uint8_t*)&color;
			uint32_t idepth = (encoded >> 32);
			float depth = *((float*)&idepth);

			if(EDL_ENABLED){
				float edlRadius = 2.0f;
				float edlStrength = 0.4f;
				float2 edlSamples[4] = {
					{-1.0f,  0.0f},
					{ 1.0f,  0.0f},
					{ 0.0f,  1.0f},
					{ 0.0f, -1.0f}
				};

				float sum = 0.0f;
				for(int i = 0; i < 4; i++){
					float2 samplePos = {
						x + edlSamples[i].x,
						y + edlSamples[i].y
					};

					int sx = clamp(samplePos.x, 0.0f, uniforms.vr_right_width - 1.0f);
					int sy = clamp(samplePos.y, 0.0f, uniforms.vr_right_height - 1.0f);
					int samplePixelIndex = sx + sy * uniforms.vr_right_width;

					uint64_t sampleEncoded = fb_vr_right[samplePixelIndex];
					uint32_t iSampledepth = (sampleEncoded >> 32);
					float sampleDepth = *((float*)&iSampledepth);

					sum += max(0.0, depth - sampleDepth);
				}

				float shade = exp(-sum * 300.0 * edlStrength);

				rgba[0] = float(rgba[0]) * shade;
				rgba[1] = float(rgba[1]) * shade;
				rgba[2] = float(rgba[2]) * shade;
			}

			surf2Dwrite(color, gl_colorbuffer_vr_right, x * 4, y);
		});

		// blit vr displays to main window
		processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){

			int x = pixelIndex % int(uniforms.width);
			int y = pixelIndex / int(uniforms.width);

			float u = fmodf(2.0 * float(x) / uniforms.width, 1.0f);
			float v = float(y) / uniforms.height;

			uint32_t color = 0x000000ff;
			if(x < uniforms.width / 2.0){
				int vr_x = u * uniforms.vr_left_width;
				int vr_y = v * uniforms.vr_left_height;
				int vr_pixelIndex = vr_x + vr_y * uniforms.vr_left_width;

				uint64_t encoded = fb_vr_left[vr_pixelIndex];
				color = encoded & 0xffffffffull;
			}else{
				int vr_x = u * uniforms.vr_right_width;
				int vr_y = v * uniforms.vr_right_height;
				int vr_pixelIndex = vr_x + vr_y * uniforms.vr_right_width;

				uint64_t encoded = fb_vr_right[vr_pixelIndex];
				color = encoded & 0xffffffffull;
			}

			if(uniforms.colorMode == COLORMODE_TIME_NORMALIZED)
			if(color != BACKGROUND_COLOR)
			{
				color = color / (maxNanos / 255);
			}

			surf2Dwrite(color, gl_colorbuffer_main, x * 4, y);
		});

	}else{
		// blit custom cuda framebuffer to opengl texture
		processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){

			int x = pixelIndex % int(uniforms.width);
			int y = pixelIndex / int(uniforms.width);

			uint64_t encoded = framebuffer[pixelIndex];
			uint32_t color = encoded & 0xffffffffull;

			if(uniforms.colorMode == COLORMODE_TIME_NORMALIZED)
			if(color != BACKGROUND_COLOR)
			{
				color = color / (maxNanos / 255);
			}

			surf2Dwrite(color, gl_colorbuffer_main, x * 4, y);
		});
	}


}
