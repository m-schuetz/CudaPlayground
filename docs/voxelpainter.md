
# Abstract

<b>About: </b> A CUDA-based virtual reality voxel painter that does everything in a single kernel, including game logic, meshing (marching cubes), rendering engine and software rasterization.

<b>Method: </b> Everything is done by a single CUDA "megakernel". Cooperative Groups are used to distribute variable-sized workloads from different passes to all launched GPU threads and blocks; a 128³ voxel grid stores "inside" and "outside" voxels; the marching cubes algorithm is used to transform the voxel grid into a mesh; and a software rasterizer draws all the generated triangles, currently utilizing one block per triangle. 

<b>Results: </b> This voxel painter runs at 90fps at a resolution of 2468x2740 per eye on an RTX 3090. In non-VR mode, it runs at around 280fps (after some painting) to 360fps (nothing painted) at a resolution of 2560x1140.


Most relevant code sections are found in [main_vr.cpp](../modules/VR/main_vr.cpp) and especially [voxelpainter.cu](../modules/VR/voxelpainter.cu)


This tech demo was created during the [2023 global game jam](https://globalgamejam.org/2023/jam-sites/tu-wien-ggj23).

# Getting Started

* Clone the repository.
* Open build/CudaPlayground.sln in Visual Studio 2022
* Right click "Cuda_OpenGL_VR" and "Set as startup project", if not already the case
* Make sure that you compile&run in "Release" mode.
* Run the project! (ctrl + F5)
* Click "turn on VR" in the settings panel. 

# Documentation

This tech demo mostly takes place in [main_vr.cpp](../modules/VR/main_vr.cpp) and especially [voxelpainter.cu](../modules/VR/voxelpainter.cu). The latter is invoked each frame, keeps track and updates the voxel grid, draws the whole scene and then transfers the results to an OpenGL texture that can be shown on screen and also submitted to VR devices.

<img width="600px" src="./voxelpainter_overview.png" />

The following subsections briefly explain some details and design choices.

## Allocation

The CUDA kernel is called once per frame and needs to hold persistent memory that is reused in future kernel invocations (e.g. voxel grid) as well as temporary memory that is cleared and rebuilt from scratch each frame (e.g. framebuffers). To achieve this, we pass a large byte array to the kernel ``` kernel(..., uint32_t* buffer, ...) ``` and use a custom allocator that reserves memory from that buffer. 

```C++
// In each frame, "allocator" always starts handing out memory from byte 0 of the buffer. 
// Since voxelGrid is the first thing we allocate in a frame, 
// it is allways guaranteed to point to the first byte, and reserve 128³ * 4 bytes.
// Due to this, it is guaranteed that each frame uses the same voxelGrid data.
uint32_t* voxelGrid = allocator->alloc<uint32_t*>(numCells * sizeof(uint32_t));
```

```C++
// We also allocate bytes for main framebuffer and eyes at the start of a frame, 
// but their location in memory does not matter because we clear them anyway
uint64_t* framebuffer = allocator->alloc<uint64_t*>(width * height * sizeof(uint64_t);
uint64_t* fb_vr_left  = allocator->alloc<uint64_t*>(vr_width * vr_height * sizeof(uint64_t));
uint64_t* fb_vr_right = allocator->alloc<uint64_t*>(vr_width * vr_height * sizeof(uint64_t));
```

## Voxel Grid and Marching Cubes

This tech demo uses a voxel grid with a size of 128³ cells. Each cell can be either empty or filled. 

