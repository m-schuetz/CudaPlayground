
# Abstract

<b>About: </b> A CUDA-based virtual reality voxel painter that does everything in a single kernel, including game logic, meshing (marching cubes), rendering engine and software rasterization.

<b>Method: </b> Everything is done by a single CUDA "megakernel". Cooperative Groups are used to distribute variable-sized workloads from different passes to all launched GPU threads and blocks; a 128³ voxel grid stores "inside" and "outside" voxels; the marching cubes algorithm is used to transform the voxel grid into a mesh; and a software rasterizer draws all the generated triangles, currently utilizing one block per triangle. 

<b>Results: </b> This voxel painter runs at 90fps at a resolution of 2468x2740 per eye on an RTX 3090. In non-VR mode, it runs at around 280fps (after some painting) to 360fps (nothing painted) at a resolution of 2560x1140.

Most relevant code sections are found in [main_vr.cpp](../modules/VR/main_vr.cpp) and especially [voxelpainter.cu](../modules/VR/voxelpainter.cu)

This tech demo was created during the [2023 global game jam](https://globalgamejam.org/2023/jam-sites/tu-wien-ggj23).

https://user-images.githubusercontent.com/6705073/216948763-1754c40a-0244-453b-851b-b42ef4b0c852.mp4


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

## Voxel Grid and Marching Cubes

See [marchingCubes()](https://github.com/m-schuetz/CudaPlayground/blob/24aaca3eb8c63c2c2836127927aeedc843514755/modules/VR/voxelpainter.cu#L817)

This tech demo uses a voxel grid with a size of 128³ cells. Each cell can be either empty or filled. Marching Cubes[1,2] is then used to create triangles along borders between empty and filled voxels. 

## Triangle Rasterization

See [rasterizeTriangles()](https://github.com/m-schuetz/CudaPlayground/blob/24aaca3eb8c63c2c2836127927aeedc843514755/modules/VR/voxelpainter.cu#L463)

The triangle rasterizer utilizes one block of threads for each triangle. Each block grabs an unrendered triangle, computes its screen-space bounding box, and then iterates over all fragments utilizing one threads per fragment. If the fragment is within the triangle, then the corresponding pixel is colored accordingly. 

The reason for utilizing one thread block comprising 128 threads per triangle is that otherwise, huge triangles would lead to extremely low frame rates. With 128 threads per triangle, the worst-case times are far better, but the best-case times are far worse. Better approaches might instead split large triangles into smaller work packages and utilize one thread per package. Also, Nanite[3] has shown that utilizing one thread per triangle can lead to 3x performance improvements over the native hardware rasterizer in case of small triangles, so future work might be able to build competitive triangle rasterizers[4] or point rasterizers[5] in CUDA.

## Megakernels

Typically if you have multiple passes that depend on previous results, or passes with different workload sizes and parallelism, you'd invoke multiple kernels with the proper amount of workgroups and threads per workgroup. However, this application is written in one single kernel that is invoked once per frame, so syncing between passes and distributing workload to all available threads is necessary.

### Syncing
If one pass depends on the results of a previous pass, we can add a sync point to wait until all threads have finished with the previous pass. In CUDA, we can globaly sync all threads with cooperative groups using cooperative_groups::this_grid.snc() (shortened to grid.sync()). grid.sync() is used liberally throughout the application, for example after clearing the framebuffers to make sure they are fully cleared before they are used, or between updating and meshing the voxel grid.

### Parallelism / Amount of workgroups

The kernel is launched once per frame with as many workgroups as the streaming multiprocessors (SM) can keep active. We use ```cuOccupancyMaxActiveBlocksPerMultiprocessor``` to find how many blocks each SM can process concurently, and launch with ```cuLaunchCooperativeKernel```. When implementing custom global sync via spinlock, this would be important because otherwise some threads would spin forever, waiting for threads that never get their turn because the spinning threads occupy all the SMs. If we only launch as many workgroups/threads as the SMs can keep active, this type of deadlock won't happen. I asume that this is one of the reasons why cooperative groups require a special ```cuLaunchCooperativeKernel``` launch call that prevents you from launching too many workgroups.

### processRange helper function

Different passes have very different workloads, ranging from modifying one single global value to 128³ voxel cells to width * height pixels. With processRange(), we can distribute a variable amount of function calls to all the threads that we've spawned this frame.

```C++
// process all 128³ cells of the voxel grid.
// the 128³ function calls are distributed to all spawned workgroups and threads
processRange(0, 128 * 128 * 128, [&](int voxelIndex){ 
	// update voxel cell
});
```

```C++
// process all width * height pixels of the framebuffer
// the width * height function calls are distributed to all spawned workgroups and threads
processRange(0, 128 * 128 * 128, [&](int pixelIndex){ 
	// update pixel
});
```



## Allocations

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


# References

* [1] http://paulbourke.net/geometry/polygonise/
* [2] [Marching cubes: A high resolution 3D surface construction algorithm](https://dl.acm.org/doi/abs/10.1145/37402.37422)
* [3] [A Deep Dive into Nanite](https://www.youtube.com/watch?v=eviSykqSUUw)
* [4] [High-performance software rasterization on GPUs](https://dl.acm.org/doi/abs/10.1145/2018323.2018337)
* [5] [Software Rasterization of 2 Billion Points in Real Time](https://arxiv.org/abs/2204.01287)
