

#include <iostream>
#include <filesystem>
#include <locale.h>
#include <string>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>
#include <format>

#include "CudaModularProgram.h"
#include "GLRenderer.h"
#include "cudaGL.h"
// #include "builtin_types.h"

#include "unsuck.hpp"
#include "ObjLoader.h"

#include "gaussianRasterization/HostDeviceInterface.h"

string path = "E:/resources/gaussian_splats/garden/point_cloud/iteration_30000/point_cloud.ply";

struct AttributesIndices{
	uint32_t x = 0;
	uint32_t y = 0;
	uint32_t z = 0;
	uint32_t nx = 0;
	uint32_t ny = 0;
	uint32_t nz = 0;
	uint32_t dc_0 = 0;
	uint32_t dc_1 = 0;
	uint32_t dc_2 = 0;
	uint32_t opacity = 0;
	uint32_t scale_0 = 0;
	uint32_t scale_1 = 0;
	uint32_t scale_2 = 0;
	uint32_t rot_0 = 0;
	uint32_t rot_1 = 0;
	uint32_t rot_2 = 0;
	uint32_t rot_3 = 0;

	vector<uint32_t> rest;
};

struct Point{
	float x;
	float y;
	float z;
	union{
		uint32_t color;
		uint8_t rgba[4];
	};
};

struct PointBatch{
	string file = "";
	int first = 0;
	int count = 0;
	shared_ptr<Buffer> points;
};

using namespace std;
using glm::mat4;
using glm::transpose;

CUdeviceptr cptr_buffer, cptr_points;
CUgraphicsResource cugl_colorbuffer;
CudaModularProgram* cuda_program = nullptr;
CUevent cevent_start, cevent_end;
cudaStream_t stream_upload;

mutex mtx_load;
mutex mtx_loadQueue;
mutex mtx_upload;

deque<PointBatch> loadQueue;
deque<PointBatch> loadedQueue;

shared_ptr<Buffer> model;
uint32_t numPointsLoaded = 0;
uint32_t numPoints = 0;

void initCuda(){
	cuInit(0);
	CUdevice cuDevice;
	CUcontext context;
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&context, 0, cuDevice);

	cuStreamCreate(&stream_upload, CU_STREAM_NON_BLOCKING);
}

void renderCUDA(shared_ptr<GLRenderer> renderer){

	static bool registered = false;
	if(!registered){
		cuGraphicsGLRegisterImage(
			&cugl_colorbuffer, 
			renderer->view.framebuffer->colorAttachments[0]->handle, 
			GL_TEXTURE_2D, 
			CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

		registered = true;
	}

	CUresult resultcode = CUDA_SUCCESS;

	CUdevice device;
	int numSMs;
	cuCtxGetDevice(&device);
	cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

	int workgroupSize = 256;

	int numGroups;
	resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numGroups, cuda_program->kernels["kernel"], workgroupSize, 0);
	numGroups *= numSMs;
	
	//numGroups = 100;
	// make sure at least 10 workgroups are spawned)
	numGroups = std::clamp(numGroups, 10, 100'000);

	std::vector<CUgraphicsResource> dynamic_resources = {cugl_colorbuffer};
	cuGraphicsMapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
	CUDA_RESOURCE_DESC res_desc = {};
	res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
	cuGraphicsSubResourceGetMappedArray(&res_desc.res.array.hArray, cugl_colorbuffer, 0, 0);
	CUsurfObject output_surf;
	cuSurfObjectCreate(&output_surf, &res_desc);

	cuEventRecord(cevent_start, 0);

	float time = now();

	Uniforms uniforms;
	uniforms.width = renderer->width;
	uniforms.height = renderer->height;
	uniforms.time = now();
	uniforms.numPoints = numPoints;
	uniforms.stride = model->size / numPoints;

	mat4 rotX = glm::rotate(glm::mat4(), 3.1415f * 0.5f, glm::vec3(1.0, 0.0, 0.0));

	mat4 world          = rotX;
	mat4 view           = renderer->camera->view;
	mat4 viewInverse    = inverse(view);
	mat4 camWorld       = renderer->camera->world;
	mat4 proj           = renderer->camera->proj;
	mat4 projInverse    = inverse(renderer->camera->proj);
	mat4 worldViewProj  = proj * view * world;
	world               = transpose(world);
	view                = transpose(view);
	viewInverse         = transpose(viewInverse);
	camWorld            = transpose(camWorld);
	proj                = transpose(proj);
	projInverse         = transpose(projInverse);
	worldViewProj       = transpose(worldViewProj);

	memcpy(&uniforms.world, &world, sizeof(world));
	memcpy(&uniforms.view, &view, sizeof(view));
	memcpy(&uniforms.viewInverse, &viewInverse, sizeof(viewInverse));
	memcpy(&uniforms.camWorld, &camWorld, sizeof(camWorld));
	memcpy(&uniforms.proj, &proj, sizeof(proj));
	memcpy(&uniforms.projInverse, &projInverse, sizeof(projInverse));
	memcpy(&uniforms.transform, &worldViewProj, sizeof(worldViewProj));

	float values[16];
	memcpy(&values, &worldViewProj, sizeof(worldViewProj));


	void* args[] = {
		&uniforms, &cptr_buffer, &cptr_points, &output_surf
	};

	auto res_launch = cuLaunchCooperativeKernel(cuda_program->kernels["kernel"],
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);

	if(res_launch != CUDA_SUCCESS){
		const char* str; 
		cuGetErrorString(res_launch, &str);
		printf("error: %s \n", str);
	}

	cuEventRecord(cevent_end, 0);
	cuSurfObjectDestroy(output_surf);
	cuGraphicsUnmapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
}

void initCudaProgram(shared_ptr<GLRenderer> renderer, shared_ptr<Buffer> model){

	cuMemAlloc(&cptr_buffer, 1'000'000'000);
	cuMemAlloc(&cptr_points, model->size);
	cuMemcpyHtoD (cptr_points, model->data, model->size);

	cuda_program = new CudaModularProgram({
		.modules = {
			"./modules/gaussianRasterization/rasterize.cu",
			"./modules/gaussianRasterization/utils.cu",
		},
		.kernels = {"kernel"}
	});

	cuEventCreate(&cevent_start, 0);
	cuEventCreate(&cevent_end, 0);

	cuGraphicsGLRegisterImage(&cugl_colorbuffer, renderer->view.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
}

shared_ptr<Buffer> loadGaussianSplats(string path){

	AttributesIndices attributes;

	auto potentialHeaderData = readBinaryFile(path, 0, 10'000);
	std::string strPotentialHeader(potentialHeaderData->data_char, potentialHeaderData->size);

	uint64_t posStartEndHeader = strPotentialHeader.find("end_header");
	uint64_t posEndHeader = strPotentialHeader.find('\n', posStartEndHeader);

	auto headerData = readBinaryFile(path, 0, posEndHeader);
	string strHeader(headerData->data_char, headerData->size);

	vector<string> lines = split(strHeader, '\n');
	int numAttributesProcessed = 0;
	for(string line : lines){

		vector<string> tokens = split(line, ' ');

		if(tokens[0] == "element" && tokens[1] == "vertex"){
			numPoints = std::stoi(tokens[2]);
		}else if(tokens[0] == "property"){
			string name = tokens[2];

			if(name == "x"){
				attributes.x = numAttributesProcessed;
			}else if(name == "y"){
				attributes.y = numAttributesProcessed;
			}else if(name == "z"){
				attributes.z = numAttributesProcessed;
			}

			numAttributesProcessed++;
		}

	}

	printfmt("numPoints: {} \n", numPoints);
	printfmt("attributes.x: {} \n", attributes.x);
	printfmt("attributes.y: {} \n", attributes.y);
	printfmt("attributes.z: {} \n", attributes.z);

	if(numPoints == 0){

		printfmt("numPoints is 0 \n");

		exit(123);
	}

	// printfmt("{} \n", strHeader);

	auto modelData = readBinaryFile(path, posEndHeader + 1, 100'000'000'000ull);

	printfmt("model.numBytes: {} \n", modelData->size);
	printfmt("posStartEndHeader: {} \n", posStartEndHeader);
	printfmt("posEndHeader: {} \n", posEndHeader);

	//printfmt("data: {} {} {} {} {} ... \n", 
	//	uint32_t(modelData->get<uint8_t>(0)),
	//	uint32_t(modelData->get<uint8_t>(1)),
	//	uint32_t(model->get<uint8_t>(2)),
	//	uint32_t(model->get<uint8_t>(3)),
	//	uint32_t(model->get<uint8_t>(4))
	//);

	return modelData;
}


int main(){

	cout << std::setprecision(2) << std::fixed;
	setlocale( LC_ALL, "en_AT.UTF-8" );

	auto renderer = make_shared<GLRenderer>();

	// position: -0.9744237992110044, 1.5045242783687351, -3.1074736490284622 
	renderer->controls->yaw    = -0.175;
	renderer->controls->pitch  = 1.565;
	renderer->controls->radius = 6.210;
	renderer->controls->target = { 0.107, 1.540, 3.007, };


	model = loadGaussianSplats(path);

	initCuda();
	initCudaProgram(renderer, model);

	auto update = [&](){
		
	};

	auto render = [&](){
		renderer->view.framebuffer->setSize(renderer->width, renderer->height);

		glBindFramebuffer(GL_FRAMEBUFFER, renderer->view.framebuffer->handle);

		renderCUDA(renderer);

		{ // INFO WINDOW

			ImGui::SetNextWindowPos(ImVec2(10, 280));
			ImGui::SetNextWindowSize(ImVec2(490, 180));

			ImGui::Begin("Infos");
			
			ImGui::BulletText("Cuda software rasterizer rendering 25 instances of the spot model \n(5856 triangles, each).");
			ImGui::BulletText("Each cuda block renders one triangle, \nwith each thread processing a different fragment.");
			ImGui::BulletText("Cuda Kernel: voxelRasterization/rasterize.cu");
			ImGui::BulletText("Spot model courtesy of Keenan Crane.");

			ImGui::End();
		}

		{ // SETTINGS WINDOW

			ImGui::SetNextWindowPos(ImVec2(10, 280 + 180 + 10));
			ImGui::SetNextWindowSize(ImVec2(490, 230));

			ImGui::Begin("Settings");

			// ImGui::Text("Color:");
			// ImGui::RadioButton("Texture", &colorMode, COLORMODE_TEXTURE);
			// ImGui::RadioButton("UVs", &colorMode, COLORMODE_UV);
			// ImGui::RadioButton("Triangle Index", &colorMode, COLORMODE_TRIANGLE_ID);
			// ImGui::RadioButton("Time", &colorMode, COLORMODE_TIME);
			// ImGui::RadioButton("Time (normalized)", &colorMode, COLORMODE_TIME_NORMALIZED);

			// ImGui::Text("Sampling:");
			// ImGui::RadioButton("Nearest", &sampleMode, SAMPLEMODE_NEAREST);
			// ImGui::RadioButton("Linear", &sampleMode, SAMPLEMODE_LINEAR);

			if(ImGui::Button("Copy Camera")){
				auto controls = renderer->controls;
				auto pos = controls->getPosition();
				auto target = controls->target;

				stringstream ss;
				ss<< std::setprecision(2) << std::fixed;
				ss << format("// position: {}, {}, {} \n", pos.x, pos.y, pos.z);
				ss << format("renderer->controls->yaw    = {:.3f};\n", controls->yaw);
				ss << format("renderer->controls->pitch  = {:.3f};\n", controls->pitch);
				ss << format("renderer->controls->radius = {:.3f};\n", controls->radius);
				ss << format("renderer->controls->target = {{ {:.3f}, {:.3f}, {:.3f}, }};\n", target.x, target.y, target.z);

				string str = ss.str();
				
				toClipboard(str);
			}

			ImGui::End();
		}
	};

	renderer->loop(update, render);

	return 0;
}
