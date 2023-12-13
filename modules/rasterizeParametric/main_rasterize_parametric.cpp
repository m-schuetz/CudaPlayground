

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

#include "HostDeviceInterface.h"

using namespace std;

struct{
	int method                = METHOD_32X32;
	int model                 = MODEL_FUNKY_PLANE;
	bool isPaused             = false;
	bool enableRefinement     = false;
	double timeSinceLastFrame = 0.0;
	bool lockFrustum          = false;
} settings;

float duration_scene;
float duration_patches;
float duration_rasterization;

CUdeviceptr cptr_buffer;
CUdeviceptr cptr_framebuffer;
CUdeviceptr cptr_models, cptr_numModels;
CUdeviceptr cptr_patches, cptr_numPatches;
CUdeviceptr cptr_stats;

Stats stats;
void* h_stats_pinned = nullptr;

CUgraphicsResource cugl_colorbuffer;
CudaModularProgram* cuda_program = nullptr;
// CudaModularProgram* cuda_program_generate_patches = nullptr;
CUevent cevent_start_scene, cevent_end_scene;
CUevent cevent_start_patches, cevent_end_patches;
CUevent cevent_start_rasterization, cevent_end_rasterization;

void initCuda(){
	cuInit(0);
	CUdevice cuDevice;
	CUcontext context;
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&context, 0, cuDevice);
}

void renderCUDA(shared_ptr<GLRenderer> renderer){

	cuGraphicsGLRegisterImage(
		&cugl_colorbuffer, 
		renderer->view.framebuffer->colorAttachments[0]->handle, 
		GL_TEXTURE_2D, 
		CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	CUresult resultcode = CUDA_SUCCESS;

	CUdevice device;
	int numSMs;
	cuCtxGetDevice(&device);
	cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

	std::vector<CUgraphicsResource> dynamic_resources = {cugl_colorbuffer};
	cuGraphicsMapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
	CUDA_RESOURCE_DESC res_desc = {};
	res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
	cuGraphicsSubResourceGetMappedArray(&res_desc.res.array.hArray, cugl_colorbuffer, 0, 0);
	CUsurfObject output_surf;
	cuSurfObjectCreate(&output_surf, &res_desc);

	// cuEventRecord(cevent_start, 0);

	static float time = 0;

	if(!settings.isPaused){
		time += settings.timeSinceLastFrame;
	}

	Uniforms uniforms;
	uniforms.width = renderer->width;
	uniforms.height = renderer->height;
	uniforms.time = time;
	uniforms.method = settings.method;
	uniforms.model = settings.model;
	uniforms.isPaused = settings.isPaused;
	uniforms.enableRefinement = settings.enableRefinement;
	uniforms.lockFrustum = settings.lockFrustum;

	glm::mat4 rotX = glm::rotate(glm::mat4(), 3.1415f * 0.5f, glm::vec3(1.0, 0.0, 0.0));

	static glm::mat4 locked_view;
	static glm::mat4 locked_transform;

	glm::mat4 world = rotX;
	glm::mat4 view = renderer->camera->view;
	glm::mat4 proj = renderer->camera->proj;
	glm::mat4 worldViewProj = proj * view * world;

	world = glm::transpose(world);
	view = glm::transpose(view);
	proj = glm::transpose(proj);
	worldViewProj = glm::transpose(worldViewProj);

	if(!settings.lockFrustum){
		locked_view = view;
		locked_transform = worldViewProj;
	}

	memcpy(&uniforms.world, &world, sizeof(world));
	memcpy(&uniforms.view, &view, sizeof(view));
	memcpy(&uniforms.proj, &proj, sizeof(proj));
	memcpy(&uniforms.transform, &worldViewProj, sizeof(worldViewProj));
	memcpy(&uniforms.locked_view, &locked_view, sizeof(locked_view));
	memcpy(&uniforms.locked_transform, &locked_transform, sizeof(locked_transform));

	float values[16];
	memcpy(&values, &worldViewProj, sizeof(worldViewProj));

	cuMemcpyDtoHAsync(h_stats_pinned, cptr_stats, sizeof(Stats), ((CUstream)CU_STREAM_DEFAULT));
	memcpy(&stats, h_stats_pinned, sizeof(Stats));


	{ // CLEAR FRAMEBUFFER
		int workgroupSize = 1024;
		int numGroups;
		auto& kernel = cuda_program->kernels["kernel_clear_framebuffer"];
		resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numGroups, kernel, workgroupSize, 0);
		numGroups *= numSMs;
		numGroups = std::clamp(numGroups, 10, 100'000);

		void* args[] = {
			&uniforms, &cptr_buffer, &cptr_framebuffer,
			&cptr_models, &cptr_numModels, 
			&cptr_patches, &cptr_numPatches, 
			&output_surf, &cptr_stats
		};
		
		auto res_launch = cuLaunchCooperativeKernel(kernel,
			numGroups, 1, 1,
			workgroupSize, 1, 1,
			0, 0, args);

		if(res_launch != CUDA_SUCCESS){
			const char* str; 
			cuGetErrorString(res_launch, &str);
			printf("error: %s \n", str);
		}
	}

	if(settings.method == METHOD_SAMPLEPERF_TEST){
		int workgroupSize = 1024;
		int numGroups;
		auto& kernel = cuda_program->kernels["kernel_sampleperf_test"];
		resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numGroups, kernel, workgroupSize, 0);
		numGroups *= numSMs;
		numGroups = std::clamp(numGroups, 10, 100'000);

		void* args[] = {
			&uniforms, &cptr_buffer, &cptr_framebuffer, 
			&cptr_models, &cptr_numModels, 
			&cptr_patches, &cptr_numPatches, 
			&output_surf, &cptr_stats
		};
		
		auto res_launch = cuLaunchCooperativeKernel(kernel,
			numGroups, 1, 1,
			workgroupSize, 1, 1,
			0, 0, args);

		if(res_launch != CUDA_SUCCESS){
			const char* str; 
			cuGetErrorString(res_launch, &str);
			printf("error: %s \n", str);
		}
	}else{

		{ // GENERATE SCENE
			cuEventRecord(cevent_start_scene, 0);

			int workgroupSize = 256;
			int numGroups;
			auto& kernel = cuda_program->kernels["kernel_generate_scene"];
			resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numGroups, kernel, workgroupSize, 0);
			numGroups *= numSMs;
			numGroups = std::clamp(numGroups, 10, 100'000);

			void* args[] = {
				&uniforms, &cptr_buffer, 
				&cptr_models, &cptr_numModels, 
				&cptr_patches, &cptr_numPatches, 
				&output_surf, &cptr_stats
			};

			auto res_launch = cuLaunchCooperativeKernel(kernel,
				numGroups, 1, 1,
				workgroupSize, 1, 1,
				0, 0, args);

			if(res_launch != CUDA_SUCCESS){
				const char* str; 
				cuGetErrorString(res_launch, &str);
				printf("error: %s \n", str);
			}

			cuEventRecord(cevent_end_scene, 0);
		}

		{ // GENERATE PATCHES
			cuEventRecord(cevent_start_patches, 0);

			int workgroupSize = 256;
			int numGroups;
			auto& kernel = cuda_program->kernels["kernel_generate_patches"];
			resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numGroups, kernel, workgroupSize, 0);
			numGroups *= numSMs;
			numGroups = std::clamp(numGroups, 10, 100'000);

			void* args[] = {
				&uniforms, &cptr_buffer, 
				&cptr_models, &cptr_numModels, 
				&cptr_patches, &cptr_numPatches, 
				&output_surf, &cptr_stats
			};

			auto res_launch = cuLaunchCooperativeKernel(kernel,
				numGroups, 1, 1,
				workgroupSize, 1, 1,
				0, 0, args);

			if(res_launch != CUDA_SUCCESS){
				const char* str; 
				cuGetErrorString(res_launch, &str);
				printf("error: %s \n", str);
			}

			cuEventRecord(cevent_end_patches, 0);
		}

		cuEventRecord(cevent_start_rasterization, 0);

		if(settings.method == METHOD_32X32){
			int workgroupSize = 1024;
			int numGroups;
			auto& kernel = cuda_program->kernels["kernel_rasterize_patches_32x32"];
			resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numGroups, kernel, workgroupSize, 0);
			numGroups *= numSMs;
			numGroups = std::clamp(numGroups, 10, 100'000);

			void* args[] = {
				&uniforms, &cptr_buffer, &cptr_framebuffer,
				&cptr_models, &cptr_numModels, 
				&cptr_patches, &cptr_numPatches, 
				&output_surf, &cptr_stats
			};
			
			auto res_launch = cuLaunchCooperativeKernel(kernel,
				numGroups, 1, 1,
				workgroupSize, 1, 1,
				0, 0, args);

			if(res_launch != CUDA_SUCCESS){
				const char* str; 
				cuGetErrorString(res_launch, &str);
				printf("error: %s \n", str);
			}
		}else if(settings.method == METHOD_RUNNIN_THRU){
			int workgroupSize = 128;
			int numGroups;
			auto& kernel = cuda_program->kernels["kernel_rasterize_patches_runnin_thru"];
			resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numGroups, kernel, workgroupSize, 0);
			numGroups *= numSMs;
			numGroups = std::clamp(numGroups, 10, 100'000);

			void* args[] = {
				&uniforms, &cptr_buffer, &cptr_framebuffer,
				&cptr_models, &cptr_numModels, 
				&cptr_patches, &cptr_numPatches, 
				&output_surf, &cptr_stats
			};
			
			auto res_launch = cuLaunchCooperativeKernel(kernel,
				numGroups, 1, 1,
				workgroupSize, 1, 1,
				0, 0, args);

			if(res_launch != CUDA_SUCCESS){
				const char* str; 
				cuGetErrorString(res_launch, &str);
				printf("error: %s \n", str);
			}
		}

		cuEventRecord(cevent_end_rasterization, 0);

	}

	{ // TRANSFER FRAMEBUFFER TO OPENGL TEXTURE
		int workgroupSize = 1024;
		int numGroups;
		auto& kernel = cuda_program->kernels["kernel_framebuffer_to_OpenGL"];
		resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numGroups, kernel, workgroupSize, 0);
		numGroups *= numSMs;
		numGroups = std::clamp(numGroups, 10, 100'000);

		void* args[] = {
			&uniforms, &cptr_buffer, &cptr_framebuffer,
			&cptr_models, &cptr_numModels, 
			&cptr_patches, &cptr_numPatches, 
			&output_surf, &cptr_stats
		};
		
		auto res_launch = cuLaunchCooperativeKernel(kernel,
			numGroups, 1, 1,
			workgroupSize, 1, 1,
			0, 0, args);

		if(res_launch != CUDA_SUCCESS){
			const char* str; 
			cuGetErrorString(res_launch, &str);
			printf("error: %s \n", str);
		}
	}

	cuCtxSynchronize();

	cuEventElapsedTime(&duration_scene, cevent_start_scene, cevent_end_scene);
	cuEventElapsedTime(&duration_patches, cevent_start_patches, cevent_end_patches);
	cuEventElapsedTime(&duration_rasterization, cevent_start_rasterization, cevent_end_rasterization);

	cuSurfObjectDestroy(output_surf);
	cuGraphicsUnmapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
	cuGraphicsUnregisterResource(cugl_colorbuffer);


}

void initCudaProgram(shared_ptr<GLRenderer> renderer){
	cuMemAlloc(&cptr_buffer, 500'000'000);
	cuMemAlloc(&cptr_framebuffer, 512'000'000); // up to 8000x8000 pixels
	cuMemAlloc(&cptr_models, 10'000'000);
	cuMemAlloc(&cptr_patches, 500'000'000);
	cuMemAlloc(&cptr_numModels, 8);
	cuMemAlloc(&cptr_numPatches, 8);

	cuMemAlloc(&cptr_stats, sizeof(Stats));
	cuMemAllocHost((void**)&h_stats_pinned , sizeof(Stats));

	cuda_program = new CudaModularProgram({
		.modules = {
			"./modules/rasterizeParametric/parametric.cu",
			"./modules/rasterizeParametric/utils.cu",
		},
		.kernels = {
			"kernel_sampleperf_test",
			"kernel_generate_scene", 
			"kernel_generate_patches", 
			"kernel_rasterize_patches_32x32",
			"kernel_rasterize_patches_runnin_thru",
			"kernel_clear_framebuffer",
			"kernel_framebuffer_to_OpenGL",
			"kernel_test"
		}
	});

	cuEventCreate(&cevent_start_scene, 0);
	cuEventCreate(&cevent_end_scene, 0);
	cuEventCreate(&cevent_start_patches, 0);
	cuEventCreate(&cevent_end_patches, 0);
	cuEventCreate(&cevent_start_rasterization, 0);
	cuEventCreate(&cevent_end_rasterization, 0);

	cuGraphicsGLRegisterImage(&cugl_colorbuffer, renderer->view.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
}


int main(){

	cout << std::setprecision(2) << std::fixed;
	setlocale( LC_ALL, "en_AT.UTF-8" );

	auto renderer = make_shared<GLRenderer>();

	renderer->controls->yaw = -2.6;
	renderer->controls->pitch = -0.4;
	renderer->controls->radius = 6.0;
	renderer->controls->target = {0.0f, 0.0f, 0.0f};

	initCuda();
	initCudaProgram(renderer);


	auto update = [&](){
		static double lastFrameTime = now();
		settings.timeSinceLastFrame = now() - lastFrameTime;

		lastFrameTime = now();
	};

	auto render = [&](){
		renderer->view.framebuffer->setSize(renderer->width, renderer->height);

		glBindFramebuffer(GL_FRAMEBUFFER, renderer->view.framebuffer->handle);

		renderCUDA(renderer);

		{ // INFO WINDOW

			ImGui::SetNextWindowPos(ImVec2(10, 280));
			ImGui::SetNextWindowSize(ImVec2(490, 240));

			ImGui::Begin("Infos");

			auto locale = getSaneLocale();

			auto toMS = [locale](double millies){
				string str = "-";

				if(millies > 0.0){
					str = format("{:.1Lf} ms", millies);
				}

				return leftPad(str, 15);
			};

			auto toM = [locale](double number){
				string str = format(locale, "{:.1Lf} M", number / 1'000'000.0);
				return leftPad(str, 14);
			};

			auto toB = [locale](double number) {
				string str = format(locale, "{:.1Lf} B", number / 1'000'000'000.0);
				return leftPad(str, 14);
			};

			auto toMB = [locale](double number){
				string str = format(locale, "{:.1Lf} MB", number / 1'000'000.0);
				return leftPad(str, 15);
			};
			auto toGB = [locale](double number){
				string str = format(locale, "{:.1Lf} GB", number / 1'000'000'000.0);
				return leftPad(str, 15);
			};

			auto toIntString = [locale](double number){
				string str = format(locale, "{:L}", number);
				return leftPad(str, 10);
			};

			vector<vector<string>> table = {
				{"Durations "             , "", ""},
				{"    generate scene"     , toMS(duration_scene)           , format("{:.1f}", duration_scene)},
				{"    generate patches"   , toMS(duration_patches)         , format("{:.1f}", duration_patches)},
				{"    rasterize patches"  , toMS(duration_rasterization)   , format("{:.1f}", duration_rasterization)},
				// {"time 0  ", toMS(stats.time_0)       , format("{:.1f}", stats.time_0)},
				// {"time 1", toMS(stats.time_1)       , format("{:.1f}", stats.time_1)},
				// {"time 2", toMS(stats.time_2)       , format("{:.1f}", stats.time_2)},
			};

			for(int i = 0; i < 14; i++){

				vector<string> entry = {
					format("#patches level {}", i), 
					format("{}", stats.numPatches[i]), 
					""
				};
				table.push_back(entry);
			}

			auto flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV;
			if (ImGui::BeginTable("table1", 3, flags)){
				ImGui::TableSetupColumn("AAA", ImGuiTableColumnFlags_WidthStretch);
				ImGui::TableSetupColumn("BBB", ImGuiTableColumnFlags_WidthStretch);
				ImGui::TableSetupColumn("CCC", ImGuiTableColumnFlags_WidthFixed);
				for (int row = 0; row < table.size(); row++){
					ImGui::TableNextRow();
					for (int column = 0; column < 2; column++){
						ImGui::TableSetColumnIndex(column);
						
						ImGui::Text(table[row][column].c_str());
					}

					ImGui::PushID(row);

					ImGui::TableSetColumnIndex(2);
					if (ImGui::SmallButton("c")) {
						string str = table[row][2];
						toClipboard(str);
					}

					ImGui::PopID();
				}
				ImGui::EndTable();
			}
		

			ImGui::End();
		}

		{ // SETTINGS WINDOW

			ImGui::SetNextWindowPos(ImVec2(10, 280 + 240 + 10), ImGuiCond_FirstUseEver);
			ImGui::SetNextWindowSize(ImVec2(490, 230), ImGuiCond_FirstUseEver);

			ImGui::Begin("Settings");

			string label = settings.isPaused ? "Resume" : "Pause";
			if (ImGui::Button(label.c_str())){
				settings.isPaused = !settings.isPaused;
			}

			ImGui::Checkbox("Enable Refinement",     &settings.enableRefinement);
			ImGui::Checkbox("lock frustum",          &settings.lockFrustum);

			ImGui::Text("Method:");
			ImGui::RadioButton("sampleperf test (100M samples)", &settings.method, METHOD_SAMPLEPERF_TEST);
			ImGui::RadioButton("32x32", &settings.method, METHOD_32X32);
			ImGui::RadioButton("128 runnin thru", &settings.method, METHOD_RUNNIN_THRU);

			ImGui::Text("Model:");
			ImGui::RadioButton("Plane", &settings.model, MODEL_PLANE);
			ImGui::RadioButton("Funky Plane", &settings.model, MODEL_FUNKY_PLANE);
			ImGui::RadioButton("Extra Funky Plane", &settings.model, MODEL_EXTRA_FUNKY_PLANE);
			ImGui::RadioButton("Sphere", &settings.model, MODEL_SPHERE);
			ImGui::RadioButton("Glyph", &settings.model, MODEL_GLYPH);

			ImGui::End();
		}
	};

	renderer->loop(update, render);

	return 0;
}
