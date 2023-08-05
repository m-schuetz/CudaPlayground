

// #include <iostream>
// #include <filesystem>
// #include <locale.h>
// #include <string>
// #include <queue>
// #include <vector>
// #include <mutex>
// #include <thread>
// #include <format>

// #include "CudaModularProgram.h"
// #include "GLRenderer.h"
// #include "cudaGL.h"
// // #include "builtin_types.h"

// #include "unsuck.hpp"
// #include "ObjLoader.h"

// #include "pointRasterization/HostDeviceInterface.h"

// string path = "E:/resources/pointclouds/simlod/chiller.las";

// struct Point{
// 	float x;
// 	float y;
// 	float z;
// 	union{
// 		uint32_t color;
// 		uint8_t rgba[4];
// 	};
// };

// struct PointBatch{
// 	string file = "";
// 	int first = 0;
// 	int count = 0;
// 	shared_ptr<Buffer> points;
// };

// struct LasHeader{
// 	int32_t versionMajor;
// 	int32_t versionMinor;
// 	int32_t headerSize;
// 	uint64_t offsetToPointData;
// 	int32_t format;
// 	uint64_t recordLength;
// 	uint64_t numPoints;
// 	dvec3 scale;
// 	dvec3 offset;
// 	dvec3 min;
// 	dvec3 max;
// };

// using namespace std;
// using glm::mat4;
// using glm::transpose;

// LasHeader header;

// CUdeviceptr cptr_buffer, cptr_points;
// CUgraphicsResource cugl_colorbuffer;
// CudaModularProgram* cuda_program = nullptr;
// CUevent cevent_start, cevent_end;
// cudaStream_t stream_upload;

// mutex mtx_load;
// mutex mtx_loadQueue;
// mutex mtx_upload;

// deque<PointBatch> loadQueue;
// deque<PointBatch> loadedQueue;

// shared_ptr<Buffer> model;
// uint32_t numPointsLoaded = 0;

// void initCuda(){
// 	cuInit(0);
// 	CUdevice cuDevice;
// 	CUcontext context;
// 	cuDeviceGet(&cuDevice, 0);
// 	cuCtxCreate(&context, 0, cuDevice);

// 	cuStreamCreate(&stream_upload, CU_STREAM_NON_BLOCKING);
// }



// LasHeader readHeader(string path){
// 	shared_ptr<Buffer> headerBuffer = readBinaryFile(path, 0, 375);
// 	LasHeader header;

// 	header.versionMajor = headerBuffer->get<uint8_t>(24);
// 	header.versionMinor = headerBuffer->get<uint8_t>(25);
// 	header.headerSize = headerBuffer->get<uint16_t>(94);
// 	header.offsetToPointData = headerBuffer->get<uint32_t>(96);
// 	header.format = headerBuffer->get<uint8_t>(104);
// 	header.recordLength = headerBuffer->get<uint16_t>(105);
// 	header.numPoints = 0;
// 	if(header.versionMajor == 1 && header.versionMinor <= 2){
// 		header.numPoints = headerBuffer->get<uint32_t>(107);
// 	}else{
// 		header.numPoints = headerBuffer->get<uint64_t>(247);
// 	}
// 	header.scale = {
// 		headerBuffer->get<double>(131),
// 		headerBuffer->get<double>(139),
// 		headerBuffer->get<double>(147),
// 	};
// 	header.offset = {
// 		headerBuffer->get<double>(155),
// 		headerBuffer->get<double>(163),
// 		headerBuffer->get<double>(171),
// 	};
// 	header.min = {
// 		headerBuffer->get<double>(187),
// 		headerBuffer->get<double>(203),
// 		headerBuffer->get<double>(219),
// 	};
// 	header.max = {
// 		headerBuffer->get<double>(179),
// 		headerBuffer->get<double>(195),
// 		headerBuffer->get<double>(211),
// 	};

// 	return header;
// }

// shared_ptr<Buffer> loadLas(string path, uint32_t firstPoint, uint32_t numPoints){
	
// 	LasHeader header = readHeader(path);

// 	shared_ptr<Buffer> pointBuffer = readBinaryFile(path, 
// 		header.offsetToPointData + firstPoint * header.recordLength,
// 		numPoints * header.recordLength
// 	);
// 	shared_ptr<Buffer> targetBuffer = make_shared<Buffer>(16 * numPoints);

// 	int rgbOffset = 0;
// 	if(header.format == 2) rgbOffset = 20;
// 	if(header.format == 3) rgbOffset = 28;

// 	auto rescaleColor = [](uint32_t color){
// 		return color > 255 ? color / 256 : color;
// 	};

// 	for(int i = 0; i < numPoints; i++){
// 		int32_t X = pointBuffer->get<int32_t>(i * header.recordLength + 0);
// 		int32_t Y = pointBuffer->get<int32_t>(i * header.recordLength + 4);
// 		int32_t Z = pointBuffer->get<int32_t>(i * header.recordLength + 8);

// 		float x = double(X) * header.scale.x + header.offset.x - header.min.x;
// 		float y = double(Y) * header.scale.y + header.offset.y - header.min.y;
// 		float z = double(Z) * header.scale.z + header.offset.z - header.min.z;

// 		uint32_t R = pointBuffer->get<uint16_t>(i * header.recordLength + rgbOffset + 0);
// 		uint32_t G = pointBuffer->get<uint16_t>(i * header.recordLength + rgbOffset + 2);
// 		uint32_t B = pointBuffer->get<uint16_t>(i * header.recordLength + rgbOffset + 4);

// 		targetBuffer->set<float>(x, 16 * i + 0);
// 		targetBuffer->set<float>(y, 16 * i + 4);
// 		targetBuffer->set<float>(z, 16 * i + 8);
// 		targetBuffer->set<uint8_t>(rescaleColor(R), 16 * i + 12);
// 		targetBuffer->set<uint8_t>(rescaleColor(G), 16 * i + 13);
// 		targetBuffer->set<uint8_t>(rescaleColor(B), 16 * i + 14);
// 		targetBuffer->set<uint8_t>(255, 16 * i + 15);
// 	}

// 	return targetBuffer;
// }

// deque<PointBatch> createPointBatches(string path){

// 	LasHeader header = readHeader(path);

// 	deque<PointBatch> batches;

// 	uint64_t MAX_BATCH_SIZE = 1'000'000;
// 	for(uint64_t first = 0; first < header.numPoints; first += MAX_BATCH_SIZE){
// 		PointBatch batch;
// 		batch.file = path;
// 		batch.first = first;
// 		batch.count = std::min(header.numPoints - first, MAX_BATCH_SIZE);

// 		batches.push_back(batch);
// 	}

// 	return batches;
// }

// void spawnLoader(){
	
// 	thread t([&](){
		
// 		while(true){

// 			PointBatch batch;
// 			int t_reload_version = 0;

// 			mtx_load.lock();

// 			// get batch to load
// 			if(loadQueue.size() > 0){
// 				batch = loadQueue.front();
// 				loadQueue.pop_front();
// 			}

// 			mtx_load.unlock();

// 			if(batch.count > 0){
// 				// load points in batch

// 				shared_ptr<Buffer> points = loadLas(batch.file, batch.first, batch.count);
// 				batch.points = points;

// 				//cout << "loaded points: " << batch.first << ":" << batch.count;
// 				//cout << "; xyz: " << points->get<float>(0) << ", " << points->get<float>(1) << ", " << points->get<float>(2) << endl;

// 				numPointsLoaded += batch.count;

// 				unique_lock<mutex> lock2(mtx_load);
// 				loadedQueue.push_back(batch);
				
// 			}

// 			std::this_thread::sleep_for(1ms);
// 		}

// 	});
// 	t.detach();
// }

// void spawnUploader(){
// 	thread t([&](){

// 		while(true){

// 			double t_start = now();

// 			std::this_thread::sleep_for(100ns);

// 			// this lock ensures that we don't reset and upload at the same time
// 			lock_guard<mutex> lock_upload(mtx_upload);

// 			// acquire work, or keep spinning if there is none
// 			PointBatch batch;
// 			{
// 				lock_guard<mutex> lock_load(mtx_load);

// 				if(loadedQueue.size() > 0){
// 					batch = loadedQueue.front();
// 					loadedQueue.pop_front();
// 				}else{
// 					continue;
// 				}
// 			}

// 			auto target = cptr_points + uint64_t(16 * batch.first);
// 			auto source = batch.points->data;
// 			cuMemcpyHtoDAsync(target, source, batch.points->size, stream_upload);
// 		}

// 	});
// 	t.detach();
// }

// void renderCUDA(shared_ptr<GLRenderer> renderer){

// 	static bool registered = false;
// 	if(!registered){
// 		cuGraphicsGLRegisterImage(
// 			&cugl_colorbuffer, 
// 			renderer->view.framebuffer->colorAttachments[0]->handle, 
// 			GL_TEXTURE_2D, 
// 			CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

// 		registered = true;
// 	}

// 	CUresult resultcode = CUDA_SUCCESS;

// 	CUdevice device;
// 	int numSMs;
// 	cuCtxGetDevice(&device);
// 	cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

// 	int workgroupSize = 256;

// 	int numGroups;
// 	resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numGroups, cuda_program->kernels["kernel"], workgroupSize, 0);
// 	numGroups *= numSMs;
	
// 	//numGroups = 100;
// 	// make sure at least 10 workgroups are spawned)
// 	numGroups = std::clamp(numGroups, 10, 100'000);

// 	std::vector<CUgraphicsResource> dynamic_resources = {cugl_colorbuffer};
// 	cuGraphicsMapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
// 	CUDA_RESOURCE_DESC res_desc = {};
// 	res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
// 	cuGraphicsSubResourceGetMappedArray(&res_desc.res.array.hArray, cugl_colorbuffer, 0, 0);
// 	CUsurfObject output_surf;
// 	cuSurfObjectCreate(&output_surf, &res_desc);

// 	cuEventRecord(cevent_start, 0);

// 	float time = now();

// 	Uniforms uniforms;
// 	uniforms.width = renderer->width;
// 	uniforms.height = renderer->height;
// 	uniforms.time = now();
// 	uniforms.numPoints = header.numPoints;

// 	mat4 rotX = glm::rotate(glm::mat4(), 3.1415f * 0.5f, glm::vec3(1.0, 0.0, 0.0));

// 	mat4 world          = rotX;
// 	mat4 view           = renderer->camera->view;
// 	mat4 viewInverse    = inverse(view);
// 	mat4 camWorld       = renderer->camera->world;
// 	mat4 proj           = renderer->camera->proj;
// 	mat4 projInverse    = inverse(renderer->camera->proj);
// 	mat4 worldViewProj  = proj * view * world;
// 	world               = transpose(world);
// 	view                = transpose(view);
// 	viewInverse         = transpose(viewInverse);
// 	camWorld            = transpose(camWorld);
// 	proj                = transpose(proj);
// 	projInverse         = transpose(projInverse);
// 	worldViewProj       = transpose(worldViewProj);

// 	memcpy(&uniforms.world, &world, sizeof(world));
// 	memcpy(&uniforms.view, &view, sizeof(view));
// 	memcpy(&uniforms.viewInverse, &viewInverse, sizeof(viewInverse));
// 	memcpy(&uniforms.camWorld, &camWorld, sizeof(camWorld));
// 	memcpy(&uniforms.proj, &proj, sizeof(proj));
// 	memcpy(&uniforms.projInverse, &projInverse, sizeof(projInverse));
// 	memcpy(&uniforms.transform, &worldViewProj, sizeof(worldViewProj));

// 	float values[16];
// 	memcpy(&values, &worldViewProj, sizeof(worldViewProj));


// 	void* args[] = {
// 		&uniforms, &cptr_buffer, &cptr_points, &output_surf
// 	};

// 	auto res_launch = cuLaunchCooperativeKernel(cuda_program->kernels["kernel"],
// 		numGroups, 1, 1,
// 		workgroupSize, 1, 1,
// 		0, 0, args);

// 	if(res_launch != CUDA_SUCCESS){
// 		const char* str; 
// 		cuGetErrorString(res_launch, &str);
// 		printf("error: %s \n", str);
// 	}

// 	cuEventRecord(cevent_end, 0);
// 	cuSurfObjectDestroy(output_surf);
// 	cuGraphicsUnmapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
// }

// void initCudaProgram(shared_ptr<GLRenderer> renderer, shared_ptr<Buffer> model){

// 	cuMemAlloc(&cptr_buffer, 200'000'000);
// 	cuMemAlloc(&cptr_points, header.numPoints * 16);
// 	cuMemsetD32(cptr_points, 0, header.numPoints * 4);

// 	cuda_program = new CudaModularProgram({
// 		.modules = {
// 			"./modules/pointRasterization/rasterize.cu",
// 			"./modules/pointRasterization/utils.cu",
// 		},
// 		.kernels = {"kernel"}
// 	});

// 	cuEventCreate(&cevent_start, 0);
// 	cuEventCreate(&cevent_end, 0);

// 	cuGraphicsGLRegisterImage(&cugl_colorbuffer, renderer->view.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
// }


// int main(){

// 	cout << std::setprecision(2) << std::fixed;
// 	setlocale( LC_ALL, "en_AT.UTF-8" );

// 	auto renderer = make_shared<GLRenderer>();

// 	// position: 5.55246042571529, -2.0518362192283996, 4.1355448234314975 
// 	renderer->controls->yaw    = 0.713;
// 	renderer->controls->pitch  = -0.435;
// 	renderer->controls->radius = 5.132;
// 	renderer->controls->target = { 2.198, 1.470, 2.499, };

// 	// position: 35.32560960587908, -1.8992892713890797, 5.488428931253263 
// 	renderer->controls->yaw    = 1.095;
// 	renderer->controls->pitch  = -0.515;
// 	renderer->controls->radius = 31.387;
// 	renderer->controls->target = { 7.418, 10.600, -1.586, };



// 	header = readHeader(path);
	
// 	initCuda();
// 	initCudaProgram(renderer, model);

// 	spawnLoader();
// 	spawnLoader();
// 	spawnLoader();
// 	spawnLoader();
// 	spawnUploader();
// 	loadQueue = createPointBatches(path);

// 	auto update = [&](){
		
// 	};

// 	auto render = [&](){
// 		renderer->view.framebuffer->setSize(renderer->width, renderer->height);

// 		glBindFramebuffer(GL_FRAMEBUFFER, renderer->view.framebuffer->handle);

// 		renderCUDA(renderer);

// 		{ // INFO WINDOW

// 			ImGui::SetNextWindowPos(ImVec2(10, 280));
// 			ImGui::SetNextWindowSize(ImVec2(490, 180));

// 			ImGui::Begin("Infos");
			
// 			ImGui::BulletText("Cuda software rasterizer rendering 25 instances of the spot model \n(5856 triangles, each).");
// 			ImGui::BulletText("Each cuda block renders one triangle, \nwith each thread processing a different fragment.");
// 			ImGui::BulletText("Cuda Kernel: voxelRasterization/rasterize.cu");
// 			ImGui::BulletText("Spot model courtesy of Keenan Crane.");

// 			ImGui::End();
// 		}

// 		{ // SETTINGS WINDOW

// 			ImGui::SetNextWindowPos(ImVec2(10, 280 + 180 + 10));
// 			ImGui::SetNextWindowSize(ImVec2(490, 230));

// 			ImGui::Begin("Settings");

// 			// ImGui::Text("Color:");
// 			// ImGui::RadioButton("Texture", &colorMode, COLORMODE_TEXTURE);
// 			// ImGui::RadioButton("UVs", &colorMode, COLORMODE_UV);
// 			// ImGui::RadioButton("Triangle Index", &colorMode, COLORMODE_TRIANGLE_ID);
// 			// ImGui::RadioButton("Time", &colorMode, COLORMODE_TIME);
// 			// ImGui::RadioButton("Time (normalized)", &colorMode, COLORMODE_TIME_NORMALIZED);

// 			// ImGui::Text("Sampling:");
// 			// ImGui::RadioButton("Nearest", &sampleMode, SAMPLEMODE_NEAREST);
// 			// ImGui::RadioButton("Linear", &sampleMode, SAMPLEMODE_LINEAR);

// 			if(ImGui::Button("Copy Camera")){
// 				auto controls = renderer->controls;
// 				auto pos = controls->getPosition();
// 				auto target = controls->target;

// 				stringstream ss;
// 				ss<< std::setprecision(2) << std::fixed;
// 				ss << format("// position: {}, {}, {} \n", pos.x, pos.y, pos.z);
// 				ss << format("renderer->controls->yaw    = {:.3f};\n", controls->yaw);
// 				ss << format("renderer->controls->pitch  = {:.3f};\n", controls->pitch);
// 				ss << format("renderer->controls->radius = {:.3f};\n", controls->radius);
// 				ss << format("renderer->controls->target = {{ {:.3f}, {:.3f}, {:.3f}, }};\n", target.x, target.y, target.z);

// 				string str = ss.str();
				
// 				toClipboard(str);
// 			}

// 			ImGui::End();
// 		}
// 	};

// 	renderer->loop(update, render);

// 	return 0;
// }
