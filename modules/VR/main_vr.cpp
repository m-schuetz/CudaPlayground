

#include <iostream>
#include <filesystem>
#include <locale.h>
#include <string>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>
#include <format>

#include "OpenVRHelper.h"

#include "CudaModularProgram.h"
#include "GLRenderer.h"
#include "cudaGL.h"
// #include "builtin_types.h"

#include "unsuck.hpp"
#include "ObjLoader.h"

#include "HostDeviceInterface.h"

using namespace std;

CUdeviceptr cptr_buffer;
CUdeviceptr cptr_positions, cptr_uvs, cptr_colors;
CUdeviceptr cptr_texture;

View viewLeft;
View viewRight;

CUgraphicsResource cugl_main, cugl_vr_left, cugl_vr_right;
CudaModularProgram* cuda_program = nullptr;
CUevent cevent_start, cevent_end;

shared_ptr<ObjData> model;
// vector<uint32_t> colors;

int colorMode = COLORMODE_TEXTURE;
int sampleMode = SAMPLEMODE_LINEAR;

bool vrEnabled = false;
OpenVRHelper* ovr = nullptr;

void initCuda(){
	cuInit(0);
	CUdevice cuDevice;
	CUcontext context;
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&context, 0, cuDevice);
}

void renderCUDA(shared_ptr<GLRenderer> renderer){

	cuGraphicsGLRegisterImage(
		&cugl_main, 
		renderer->view.framebuffer->colorAttachments[0]->handle, 
		GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	cuGraphicsGLRegisterImage(
		&cugl_vr_left, 
		viewLeft.framebuffer->colorAttachments[0]->handle, 
		GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	cuGraphicsGLRegisterImage(
		&cugl_vr_right, 
		viewRight.framebuffer->colorAttachments[0]->handle, 
		GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	CUresult resultcode = CUDA_SUCCESS;

	CUdevice device;
	int numSMs;
	cuCtxGetDevice(&device);
	cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

	int workgroupSize = 128;

	int numGroups;
	resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numGroups, cuda_program->kernels["kernel"], workgroupSize, 0);
	numGroups *= numSMs;
	
	//numGroups = 100;
	// make sure at least 10 workgroups are spawned)
	numGroups = std::clamp(numGroups, 10, 100'000);

	vector<CUgraphicsResource> dynamic_resources = {cugl_main, cugl_vr_left, cugl_vr_right};
	cuGraphicsMapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

	CUDA_RESOURCE_DESC res_desc_main = {};
	res_desc_main.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
	cuGraphicsSubResourceGetMappedArray(&res_desc_main.res.array.hArray, cugl_main, 0, 0);
	CUsurfObject output_main;
	cuSurfObjectCreate(&output_main, &res_desc_main);

	CUDA_RESOURCE_DESC res_desc_vr_left = {};
	res_desc_vr_left.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
	cuGraphicsSubResourceGetMappedArray(&res_desc_vr_left.res.array.hArray, cugl_vr_left, 0, 0);
	CUsurfObject output_vr_left;
	cuSurfObjectCreate(&output_vr_left, &res_desc_vr_left);

	CUDA_RESOURCE_DESC res_desc_vr_right = {};
	res_desc_vr_right.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
	cuGraphicsSubResourceGetMappedArray(&res_desc_vr_right.res.array.hArray, cugl_vr_right, 0, 0);
	CUsurfObject output_vr_right;
	cuSurfObjectCreate(&output_vr_right, &res_desc_vr_right);

	cuEventRecord(cevent_start, 0);

	float time = now();

	Uniforms uniforms;
	uniforms.width = renderer->width;
	uniforms.height = renderer->height;
	uniforms.time = now();
	uniforms.colorMode = colorMode;
	uniforms.sampleMode = sampleMode;

	uniforms.vrEnabled = vrEnabled;
	uniforms.vr_left_width = viewLeft.framebuffer->width;
	uniforms.vr_left_height = viewLeft.framebuffer->height;
	uniforms.vr_right_width = viewRight.framebuffer->width;
	uniforms.vr_right_height = viewRight.framebuffer->height;

	{ // world view proj
		glm::mat4 rotX = glm::rotate(glm::mat4(), 3.1415f * 0.5f, glm::vec3(1.0, 0.0, 0.0));
		glm::mat4 world = rotX;
		glm::mat4 view = renderer->camera->view;
		glm::mat4 proj = renderer->camera->proj;
		glm::mat4 worldViewProj = proj * view * world;
		world = glm::transpose(world);
		view = glm::transpose(view);
		proj = glm::transpose(proj);
		worldViewProj = glm::transpose(worldViewProj);
		memcpy(&uniforms.world, &world, sizeof(world));
		memcpy(&uniforms.view, &view, sizeof(view));
		memcpy(&uniforms.proj, &proj, sizeof(proj));
		memcpy(&uniforms.transform, &worldViewProj, sizeof(worldViewProj));
	}

	{ // world view proj VR LEFT
		glm::mat4 rotX = glm::rotate(glm::mat4(), 3.1415f * 0.5f, glm::vec3(1.0, 0.0, 0.0));
		glm::mat4 world = rotX;
		glm::mat4 view = viewLeft.view;
		glm::mat4 proj = viewLeft.proj;
		glm::mat4 worldViewProj = proj * view * world;
		world = glm::transpose(world);
		view = glm::transpose(view);
		proj = glm::transpose(proj);
		worldViewProj = glm::transpose(worldViewProj);
		memcpy(&uniforms.vr_left_world, &world, sizeof(world));
		memcpy(&uniforms.vr_left_view, &view, sizeof(view));
		memcpy(&uniforms.vr_left_proj, &proj, sizeof(proj));
		memcpy(&uniforms.vr_left_transform, &worldViewProj, sizeof(worldViewProj));

		if(ovr->isActive()){
			Pose poseLeft = ovr->getLeftControllerPose();
			uniforms.vr_left_controller_active = poseLeft.valid;
			if(poseLeft.valid){
				glm::mat4 transform = poseLeft.transform;
				memcpy(&uniforms.vr_left_controller_pose, &transform, sizeof(transform));
			}

			Pose poseRight = ovr->getRightControllerPose();
			uniforms.vr_right_controller_active = poseRight.valid;
			if(poseRight.valid){
				glm::mat4 transform = poseRight.transform;
				memcpy(&uniforms.vr_right_controller_pose, &transform, sizeof(transform));
			}
		}
	}

	{ // world view proj VR RIGHT
		glm::mat4 rotX = glm::rotate(glm::mat4(), 3.1415f * 0.5f, glm::vec3(1.0, 0.0, 0.0));
		glm::mat4 world = rotX;
		glm::mat4 view = viewRight.view;
		glm::mat4 proj = viewRight.proj;
		glm::mat4 worldViewProj = proj * view * world;
		world = glm::transpose(world);
		view = glm::transpose(view);
		proj = glm::transpose(proj);
		worldViewProj = glm::transpose(worldViewProj);
		memcpy(&uniforms.vr_right_world, &world, sizeof(world));
		memcpy(&uniforms.vr_right_view, &view, sizeof(view));
		memcpy(&uniforms.vr_right_proj, &proj, sizeof(proj));
		memcpy(&uniforms.vr_right_transform, &worldViewProj, sizeof(worldViewProj));
	}


	void* args[] = {
		&uniforms, &cptr_buffer, 
		&output_main, &output_vr_left, &output_vr_right,
		&model->numTriangles, &cptr_positions, &cptr_uvs, &cptr_colors,
		&cptr_texture
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
	// cuEventSynchronize(cevent_end);

	// {
	// 	float total_ms;
	// 	cuEventElapsedTime(&total_ms, cevent_start, cevent_end);

	// 	cout << "CUDA durations: " << endl;
	// 	cout << std::format("total:     {:6.1f} ms", total_ms) << endl;
	// }

	cuCtxSynchronize();

	cuSurfObjectDestroy(output_main);
	cuSurfObjectDestroy(output_vr_left);
	cuSurfObjectDestroy(output_vr_right);
	cuGraphicsUnmapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
	cuGraphicsUnregisterResource(cugl_main);
	cuGraphicsUnregisterResource(cugl_vr_left);
	cuGraphicsUnregisterResource(cugl_vr_right);


}

void initCudaProgram(
	shared_ptr<GLRenderer> renderer, 
	shared_ptr<ObjData> model, 
	vector<uint32_t>& texture
){

	cuMemAlloc(&cptr_buffer, 500'000'000);

	int numVertices = model->numTriangles * 3;
	cuMemAlloc(&cptr_positions, numVertices * 12);
	cuMemAlloc(&cptr_uvs      , numVertices *  8);
	cuMemcpyHtoD(cptr_positions, model->xyz.data(), numVertices * 12);
	cuMemcpyHtoD(cptr_uvs      , model->uv.data() , numVertices *  8);
	
	cuMemAlloc(&cptr_texture   , 4 * 1024 * 1024);
	cuMemcpyHtoD(cptr_texture, texture.data(), 4 * 1024 * 1024);

	cuda_program = new CudaModularProgram({
		.modules = {
			"./modules/VR/rasterize.cu",
			"./modules/VR/utils.cu",
		},
		.kernels = {"kernel"}
	});

	cuEventCreate(&cevent_start, 0);
	cuEventCreate(&cevent_end, 0);

	// cuGraphicsGLRegisterImage(&cugl_main, renderer->view.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
	// cuGraphicsGLRegisterImage(&cugl_vr_left, viewLeft.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
	// cuGraphicsGLRegisterImage(&cugl_vr_right, viewRight.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
}


int main(){

	cout << std::setprecision(2) << std::fixed;
	setlocale( LC_ALL, "en_AT.UTF-8" );

	auto renderer = make_shared<GLRenderer>();

	renderer->controls->yaw = -2.6;
	renderer->controls->pitch = -0.4;
	renderer->controls->radius = 6.0;
	renderer->controls->target = {0.0f, 0.0f, 0.0f};

	//viewLeft.framebuffer = renderer->createFramebuffer(3000, 4000);
	//viewRight.framebuffer = renderer->createFramebuffer(3000, 4000);
	viewLeft.framebuffer = renderer->createFramebuffer(128, 128);
	viewRight.framebuffer = renderer->createFramebuffer(128, 128);

	ovr = OpenVRHelper::instance();

	auto flip = glm::dmat4(
		1.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		0.0, -1.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 1.0
	);


	initCuda();

	model = ObjLoader::load("./resources/spot/spot_triangulated.obj");
	auto ppmdata = readBinaryFile("./resources/spot/spot_texture.ppm", 40, 1000000000000);
	vector<uint32_t> colors(1024 * 1024, 0);

	for(int i = 0; i < 1024 * 1024; i++){
		uint32_t r = ppmdata->get<uint8_t>(3 * i + 0);
		uint32_t g = ppmdata->get<uint8_t>(3 * i + 1);
		uint32_t b = ppmdata->get<uint8_t>(3 * i + 2);
		uint32_t color = r | (g << 8) | (b << 16);

		colors[i] = color;
	}

	initCudaProgram(renderer, model, colors);


	auto update = [&](){

		if(ovr->isActive()){
			ovr->updatePose();
			ovr->processEvents();

			auto size = ovr->getRecommmendedRenderTargetSize();	
			int width = size[0];
			int height = size[1];

			bool needsResizeX = viewLeft.framebuffer->width != width;
			bool needsResizeY = viewLeft.framebuffer->height != height;
			bool needsResize = needsResizeX || needsResizeY;

			viewLeft.framebuffer->setSize(width, height);
			viewRight.framebuffer->setSize(width, height);

			glBindFramebuffer(GL_FRAMEBUFFER, viewLeft.framebuffer->handle);
			glViewport(0, 0, width, height);
			glBindFramebuffer(GL_FRAMEBUFFER, viewRight.framebuffer->handle);
			glViewport(0, 0, width, height);

			float near = 0.1;
			float far = 100'000.0;

			dmat4 projLeft = ovr->getProjection(vr::EVREye::Eye_Left, near, far);
			dmat4 projRight = ovr->getProjection(vr::EVREye::Eye_Right, near, far);

			auto poseHMD = ovr->hmdPose;
			auto poseLeft = ovr->getEyePose(vr::Hmd_Eye::Eye_Left);
			auto poseRight = ovr->getEyePose(vr::Hmd_Eye::Eye_Right);

			viewLeft.view = glm::inverse(flip * poseHMD * poseLeft);
			viewLeft.proj = ovr->getProjection(vr::Hmd_Eye::Eye_Left, 0.01, 10'000.0);

			viewRight.view = glm::inverse(flip * poseHMD * poseRight);
			viewRight.proj = ovr->getProjection(vr::Hmd_Eye::Eye_Right, 0.01, 10'000.0);

			
		}
		
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
			ImGui::BulletText("Cuda Kernel: rasterizeTriangles/rasterize.cu");
			ImGui::BulletText("Spot model courtesy of Keenan Crane.");

			ImGui::End();
		}

		{ // SETTINGS WINDOW

			ImGui::SetNextWindowPos(ImVec2(10, 280 + 180 + 10));
			ImGui::SetNextWindowSize(ImVec2(490, 230));

			ImGui::Begin("Settings");

			static int clicked = 0;
			string str = vrEnabled ? "turn off VR" : "turn on VR";
			if(ImGui::Button(str.c_str())){
				vrEnabled = !vrEnabled;

				if(vrEnabled){
					ovr->start();
				}else{
					ovr->stop();
				}
			}
			
			

			ImGui::End();
		}

		if(ovr->isActive()){
			
			// glBindFramebuffer(GL_FRAMEBUFFER, viewLeft.framebuffer->handle);
			// glClearColor(0.8, 0.2, 0.3, 1.0);
			// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			// glBindFramebuffer(GL_FRAMEBUFFER, viewRight.framebuffer->handle);
			// glClearColor(0.2, 0.8, 0.3, 1.0);
			// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			ovr->submit(
				viewLeft.framebuffer->colorAttachments[0]->handle, 
				viewRight.framebuffer->colorAttachments[0]->handle);
			ovr->postPresentHandoff();
		}

		glBindFramebuffer(GL_FRAMEBUFFER, renderer->view.framebuffer->handle);
	};

	renderer->loop(update, render);

	return 0;
}
