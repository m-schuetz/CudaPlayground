

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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "PxPhysicsAPI.h"

using namespace std;

CUdeviceptr cptr_buffer;
CUdeviceptr cptr_positions, cptr_uvs, cptr_colors;
CUdeviceptr cptr_texture;

struct CuFramebuffer{
	CUdeviceptr cptr = 0;
	int width  = 0;
	int height = 0;
};

CuFramebuffer framebuffer;
CuFramebuffer fb_vr_left;
CuFramebuffer fb_vr_right;

Skybox skybox;

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

struct{
	bool measureLaunchDurations = false;
} settings;

void initCuda(){
	cuInit(0);
	CUdevice cuDevice;
	CUcontext context;
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&context, 0, cuDevice);
}

Uniforms getUniforms(shared_ptr<GLRenderer> renderer){
	Uniforms uniforms;
	uniforms.width = renderer->width;
	uniforms.height = renderer->height;
	uniforms.time = now();
	uniforms.deltatime = renderer->timeSinceLastFrame;
	uniforms.frameCount = renderer->frameCount;
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
		glm::mat4 view_inv = glm::inverse(view);
		glm::mat4 proj_inv = glm::inverse(proj);

		world = glm::transpose(world);
		view = glm::transpose(view);
		proj = glm::transpose(proj);
		worldViewProj = glm::transpose(worldViewProj);
		view_inv = glm::transpose(view_inv);
		proj_inv = glm::transpose(proj_inv);

		memcpy(&uniforms.world, &world, sizeof(world));
		memcpy(&uniforms.view, &view, sizeof(view));
		memcpy(&uniforms.proj, &proj, sizeof(proj));
		memcpy(&uniforms.view_inv, &view_inv, sizeof(view_inv));
		memcpy(&uniforms.proj_inv, &proj_inv, sizeof(proj_inv));
		memcpy(&uniforms.transform, &worldViewProj, sizeof(worldViewProj));


		glm::inverse(view);
	}

	if(ovr->isActive()){
		{ // VR LEFT
			glm::mat4 rotX = glm::rotate(glm::mat4(), 3.1415f * 0.5f, glm::vec3(1.0, 0.0, 0.0));
			glm::mat4 world = rotX;
			glm::mat4 view = viewLeft.view;
			glm::mat4 proj = viewLeft.proj;
			glm::mat4 worldViewProj = proj * view * world;
			glm::mat4 view_inv = glm::inverse(view);
			glm::mat4 proj_inv = glm::inverse(proj);

			world = glm::transpose(world);
			view = glm::transpose(view);
			proj = glm::transpose(proj);
			worldViewProj = glm::transpose(worldViewProj);
			view_inv = glm::transpose(view_inv);
			proj_inv = glm::transpose(proj_inv);

			memcpy(&uniforms.vr_left_world, &world, sizeof(world));
			memcpy(&uniforms.vr_left_view, &view, sizeof(view));
			memcpy(&uniforms.vr_left_proj, &proj, sizeof(proj));
			memcpy(&uniforms.vr_left_transform, &worldViewProj, sizeof(worldViewProj));
			memcpy(&uniforms.vr_left_view_inv, &view_inv, sizeof(view_inv));
			memcpy(&uniforms.vr_left_proj_inv, &proj_inv, sizeof(proj_inv));
			
			Pose poseLeft = ovr->getLeftControllerPose();
			uniforms.vr_left_controller_active = poseLeft.valid;
			if(poseLeft.valid){
				glm::mat4 transform = poseLeft.transform;
				memcpy(&uniforms.vr_left_controller_pose, &transform, sizeof(transform));
			}
			
			auto state = ovr->getLeftControllerState();
			uniforms.vr_left_controller_state.packetNum = state.unPacketNum;
			uniforms.vr_left_controller_state.buttonPressedMask = state.ulButtonPressed;
			uniforms.vr_left_controller_state.buttonTouchedMask = state.ulButtonTouched;
		}

		{ // VR RIGHT
			glm::mat4 rotX = glm::rotate(glm::mat4(), 3.1415f * 0.5f, glm::vec3(1.0, 0.0, 0.0));
			glm::mat4 world = rotX;
			glm::mat4 view = viewRight.view;
			glm::mat4 proj = viewRight.proj;
			glm::mat4 worldViewProj = proj * view * world;
			glm::mat4 view_inv = glm::inverse(view);
			glm::mat4 proj_inv = glm::inverse(proj);

			world = glm::transpose(world);
			view = glm::transpose(view);
			proj = glm::transpose(proj);
			worldViewProj = glm::transpose(worldViewProj);
			view_inv = glm::transpose(view_inv);
			proj_inv = glm::transpose(proj_inv);

			memcpy(&uniforms.vr_right_world, &world, sizeof(world));
			memcpy(&uniforms.vr_right_view, &view, sizeof(view));
			memcpy(&uniforms.vr_right_proj, &proj, sizeof(proj));
			memcpy(&uniforms.vr_right_transform, &worldViewProj, sizeof(worldViewProj));
			memcpy(&uniforms.vr_right_view_inv, &view_inv, sizeof(view_inv));
			memcpy(&uniforms.vr_right_proj_inv, &proj_inv, sizeof(proj_inv));

			Pose poseRight = ovr->getRightControllerPose();
			uniforms.vr_right_controller_active = poseRight.valid;
			if(poseRight.valid){
				glm::mat4 transform = poseRight.transform;
				memcpy(&uniforms.vr_right_controller_pose, &transform, sizeof(transform));
			}

			auto state = ovr->getRightControllerState();
			uniforms.vr_right_controller_state.packetNum = state.unPacketNum;
			uniforms.vr_right_controller_state.buttonPressedMask = state.ulButtonPressed;
			uniforms.vr_right_controller_state.buttonTouchedMask = state.ulButtonTouched;
		}
	}

	return uniforms;
}

void resizeFramebuffer(CuFramebuffer& framebuffer, int width, int height){

	bool sizeChanged = width != framebuffer.width || height != framebuffer.height;

	if(sizeChanged){
		if(framebuffer.cptr != 0){
			cuMemFree(framebuffer.cptr);
		}

		framebuffer.width  = width;
		framebuffer.height = height;

		cuMemAlloc(&framebuffer.cptr, width * height * 8);
	}

}

void renderCUDA(shared_ptr<GLRenderer> renderer){

	resizeFramebuffer(framebuffer, renderer->width, renderer->height);
	resizeFramebuffer(fb_vr_left, viewLeft.framebuffer->width, viewLeft.framebuffer->height);
	resizeFramebuffer(fb_vr_right, viewRight.framebuffer->width, viewRight.framebuffer->height);

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

	Uniforms uniforms = getUniforms(renderer);

	void* args[] = {
		&uniforms, &cptr_buffer, 
		&output_main, &output_vr_left, &output_vr_right,
		&framebuffer.cptr, &fb_vr_left.cptr, &fb_vr_right.cptr,
		&model->numTriangles, &cptr_positions, &cptr_uvs, &cptr_colors,
		&cptr_texture, &skybox
	};

	OptionalLaunchSettings launchSettings = {
		.measureDuration = settings.measureLaunchDurations,
	};

	cuda_program->launch("kernel", args, launchSettings);
	cuda_program->launch("kernel_draw_skybox", args, launchSettings);
	cuda_program->launch("kernel_toOpenGL", args, launchSettings);

	cuEventRecord(cevent_end, 0);

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
	uint8_t* texture
){

	uint64_t bufferSize = 4'000'000'000;
	cuMemAlloc(&cptr_buffer, bufferSize);
	cuMemsetD8(cptr_buffer, 0, bufferSize); 

	int numVertices = model->numTriangles * 3;
	cuMemAlloc(&cptr_positions, numVertices * 12);
	cuMemAlloc(&cptr_uvs      , numVertices *  8);
	cuMemcpyHtoD(cptr_positions, model->xyz.data(), numVertices * 12);
	cuMemcpyHtoD(cptr_uvs      , model->uv.data() , numVertices *  8);
	
	cuMemAlloc(&cptr_texture   , 4 * 1024 * 1024);
	cuMemcpyHtoD(cptr_texture, texture, 4 * 1024 * 1024);

	cuda_program = new CudaModularProgram({
		.modules = {
			"./modules/VR/voxelpainter.cu",
			"./modules/VR/utils.cu",
		},
		.kernels = {
			"kernel", 
			"kernel_draw_skybox", 
			"kernel_toOpenGL"
		}
	});

	cuEventCreate(&cevent_start, 0);
	cuEventCreate(&cevent_end, 0);

	// cuGraphicsGLRegisterImage(&cugl_main, renderer->view.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
	// cuGraphicsGLRegisterImage(&cugl_vr_left, viewLeft.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
	// cuGraphicsGLRegisterImage(&cugl_vr_right, viewRight.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
}
//
// PHYSX ===============================================
using namespace physx;

static PxDefaultAllocator		gAllocator;
static PxDefaultErrorCallback	gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene		= NULL;
static PxMaterial*				gMaterial	= NULL;
static PxPvd*					gPvd        = NULL;

void initPhysx(){
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
}
// PHYSX ===============================================

int main(){

	initPhysx();

	

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

	// load spot
	int texWidth, texHeight, numChannels;
	unsigned char *colors = stbi_load("./resources/spot/spot_texture.png", &texWidth, &texHeight, &numChannels, 4);
	model = ObjLoader::load("./resources/spot/spot_triangulated.obj");

	{ // load skybox
		// - box with 6 textures, each the same size
		// nx, ny, nz, px, py, pz
		int width;
		int height;
		int numChannels;
		
		vector<unsigned char*> data(6, nullptr);
		data[0] = stbi_load("./resources/skybox2/nx.jpg", &width, &height, &numChannels, 4);
		data[1] = stbi_load("./resources/skybox2/ny.jpg", &width, &height, &numChannels, 4);
		data[2] = stbi_load("./resources/skybox2/nz.jpg", &width, &height, &numChannels, 4);
		data[3] = stbi_load("./resources/skybox2/px.jpg", &width, &height, &numChannels, 4);
		data[4] = stbi_load("./resources/skybox2/py.jpg", &width, &height, &numChannels, 4);
		data[5] = stbi_load("./resources/skybox2/pz.jpg", &width, &height, &numChannels, 4);

		for(int i = 0; i < 6; i++){
			CUdeviceptr cptr;
			cuMemAlloc(&cptr, width * height * 4);
			cuMemcpyHtoD(cptr, data[i], width * height * 4);
			memcpy(&skybox.textures[i], &cptr, sizeof(CUdeviceptr));
			stbi_image_free(data[i]);
		}

		skybox.width = width;
		skybox.height = height;
	}


	initCudaProgram(renderer, model, colors);
	stbi_image_free(colors);


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

			if(ImGuiCond_FirstUseEver){
				ImGui::SetNextWindowPos(ImVec2(10, 280));
				ImGui::SetNextWindowSize(ImVec2(490, 180));
			}

			ImGui::Begin("Infos");
			
			ImGui::BulletText("Cuda software rasterizer rendering 25 instances of the spot model \n(5856 triangles, each).");
			ImGui::BulletText("Each cuda block renders one triangle, \nwith each thread processing a different fragment.");
			ImGui::BulletText("Cuda Kernel: rasterizeTriangles/rasterize.cu");
			ImGui::BulletText("Spot model courtesy of Keenan Crane.");

			ImGui::End();
		}

		{ // SETTINGS WINDOW

			if(ImGuiCond_FirstUseEver){
				ImGui::SetNextWindowPos(ImVec2(10, 280 + 180 + 10));
				ImGui::SetNextWindowSize(ImVec2(490, 230));
			}

			ImGui::Begin("Settings");

			ImGui::Checkbox("Measure Kernel Durations", &settings.measureLaunchDurations);

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

		{ // STATS WINDOW
			if(ImGuiCond_FirstUseEver){
				ImGui::SetNextWindowSize(ImVec2(490, 230));
			}


			ImGui::Begin("Stats");

			for(string kernelName : cuda_program->kernelNames){
				float duration = cuda_program->last_launch_duration[kernelName];
				string msg = std::format("{}: {:.3f} ms", kernelName, duration);
				ImGui::Text(msg.c_str());
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
