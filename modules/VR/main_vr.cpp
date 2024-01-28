

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
#include <glm/gtx/matrix_decompose.hpp>

#include "unsuck.hpp"
#include "ObjLoader.h"

#include "HostDeviceInterface.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContext.h"
#include "cudamanager/PxCudaContextManager.h"
#include "extensions/PxRemeshingExt.h"
#include "extensions/PxParticleExt.h"

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

CuFramebuffer fb_points;
CuFramebuffer fb_points_vr_left;
CuFramebuffer fb_points_vr_right;

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

using namespace physx;
using namespace ExtGpu;

static PxDefaultAllocator		gAllocator;
static PxDefaultErrorCallback	gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene		= NULL;
static PxMaterial*				gMaterial	= NULL;
static PxPBDParticleSystem*		gParticleSystem = NULL;
static PxParticleBuffer*		gParticleBuffer = NULL;
static PxRigidDynamic*			gControllerBodies[2] = { NULL, NULL };

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

	resizeFramebuffer(fb_points, renderer->width, renderer->height);
	resizeFramebuffer(fb_points_vr_left, viewLeft.framebuffer->width, viewLeft.framebuffer->height);
	resizeFramebuffer(fb_points_vr_right, viewRight.framebuffer->width, viewRight.framebuffer->height);

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

	void* physx_positions = nullptr;
	uint32_t physx_numParticles = 0;

	if(gParticleSystem){
		physx_positions = (void*)gParticleBuffer->getPositionInvMasses();
		physx_numParticles = gParticleBuffer->getNbActiveParticles();
	}

	void* args_with_phsx[] = {
		&uniforms, &cptr_buffer, 
		&output_main, &output_vr_left, &output_vr_right,
		&framebuffer.cptr, &fb_vr_left.cptr, &fb_vr_right.cptr,
		&fb_points.cptr, &fb_points_vr_left.cptr, &fb_points_vr_right.cptr,
		&model->numTriangles, &cptr_positions, &cptr_uvs, &cptr_colors,
		&cptr_texture, &skybox,
		&physx_positions, &physx_numParticles
	};

	void* args_resolve_points[] = {
		&uniforms, &cptr_buffer, 
		&fb_points.cptr, &fb_points_vr_left.cptr, &fb_points_vr_right.cptr,
		&framebuffer.cptr, &fb_vr_left.cptr, &fb_vr_right.cptr,
		&physx_positions, &physx_numParticles
	};

	void* args[] = {
		&uniforms, &cptr_buffer, 
		&output_main, &output_vr_left, &output_vr_right,
		&framebuffer.cptr, &fb_vr_left.cptr, &fb_vr_right.cptr,
		&model->numTriangles, &cptr_positions, &cptr_uvs, &cptr_colors,
		&cptr_texture, &skybox,
	};

	OptionalLaunchSettings launchSettings = {
		.measureDuration = settings.measureLaunchDurations,
	};

	cuda_program->launch("kernel", args_with_phsx, {.blockSize = 256});
	cuda_program->launch("kernel_resolve_pointsToSpheres", args_resolve_points, {.blockSize = 256});
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

	uint64_t bufferSize = 2'000'000'000;
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
			"kernel_resolve_pointsToSpheres",
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


PxVec3 cubeVertices[] = { PxVec3(0.5f, -0.5f, -0.5f), PxVec3(0.5f, -0.5f, 0.5f),  PxVec3(-0.5f, -0.5f, 0.5f),  PxVec3(-0.5f, -0.5f, -0.5f),
	PxVec3(0.5f, 0.5f, -0.5f), PxVec3(0.5f, 0.5f, 0.5f), PxVec3(-0.5f, 0.5f, 0.5f), PxVec3(-0.5f, 0.5f, -0.5f) };

PxU32 cubeIndices[] = { 1, 2, 3,  7, 6, 5,  4, 5, 1,  5, 6, 2,  2, 6, 7,  0, 3, 7,  0, 1, 3,  4, 7, 5,  0, 4, 1,  1, 5, 2,  3, 2, 7,  4, 0, 7 };

enum ParticleTypes
{
	eWater = 0,
	eSand = 1,
};

static void initParticles(ParticleTypes particleType)
{
	PxCudaContextManager* cudaContextManager = gScene->getCudaContextManager();
	if (cudaContextManager == NULL)
		return;

	const PxVec3 position(0.0f, 0.0f, 0.0f);

	// fluid as default
	const PxU32 numX = 50;
	PxU32 numY = 120;  
	const PxU32 numZ = 30;
	PxReal particleSpacing = 0.1f;
	PxReal density = 1000.0f;

	// Material setup
	PxReal friction = 0.05f;
	PxReal damping = 0.05f;
	PxReal adhesion = 0.0f;
	PxReal viscosity = 0.001f;
	PxReal vorticityConfinement = 10.0f;
	PxReal surfaceTension = 0.00704f;
	PxReal cohesion = 0.0704f;
	PxReal lift = 0.0f;
	PxReal drag = 0.0f;

	PxParticlePhaseFlags phaseFlag = PxParticlePhaseFlags(
		PxParticlePhaseFlag::eParticlePhaseFluid | 
		PxParticlePhaseFlag::eParticlePhaseSelfCollide
	);

	if (particleType == ParticleTypes::eSand)
	{
		numY = 200;
		particleSpacing = 0.03f;
		density = 10'000.0f;
		friction = 1500000.0f;
		damping = 0.25f;
		adhesion = 10.1f;
		viscosity = 1000.0f;
		vorticityConfinement = 0.0f;
		surfaceTension = 0.0f;
		cohesion = 10000.0f;
		lift = 0.0f;
		drag = 0.0f;
		phaseFlag = PxParticlePhaseFlag::eParticlePhaseSelfCollide;  // no Fluid / pressure
	}

	PxPBDMaterial* fluidMat = gPhysics->createPBDMaterial(
		friction, damping, adhesion, viscosity, vorticityConfinement,
		surfaceTension, cohesion, lift, drag);


	PxPBDParticleSystem* particleSystem = gPhysics->createPBDParticleSystem(*cudaContextManager, 96);
	gParticleSystem = particleSystem;

	// General particle system setting
	const PxReal restOffset = 0.5f * particleSpacing / 0.6f;
	const PxReal solidRestOffset = restOffset;
	const PxReal fluidRestOffset = restOffset * 0.6f;
	const PxReal particleMass = density * 1.333f * 3.14159f * particleSpacing * particleSpacing * particleSpacing;
	particleSystem->setRestOffset(restOffset);
	particleSystem->setContactOffset(restOffset + 0.01f);
	particleSystem->setParticleContactOffset(fluidRestOffset / 0.6f);
	particleSystem->setSolidRestOffset(solidRestOffset);
	particleSystem->setFluidRestOffset(fluidRestOffset);
	particleSystem->enableCCD(false);
	particleSystem->setMaxVelocity(solidRestOffset * 100.f);

	gScene->addActor(*particleSystem);

	// Create particles and add them to the particle system
	const PxU32 particlePhase = particleSystem->createPhase(fluidMat, phaseFlag);

	const PxU32 maxParticles = numX * numY * numZ;
	PxU32* phase = cudaContextManager->allocPinnedHostBuffer<PxU32>(maxParticles);
	PxVec4* positionInvMass = cudaContextManager->allocPinnedHostBuffer<PxVec4>(maxParticles);
	PxVec4* velocity = cudaContextManager->allocPinnedHostBuffer<PxVec4>(maxParticles);

	PxReal x = position.x;
	PxReal y = position.y;
	PxReal z = position.z;

	for (PxU32 i = 0; i < numX; ++i)
	{
		for (PxU32 j = 0; j < numY; ++j)
		{
			for (PxU32 k = 0; k < numZ; ++k)
			{
				const PxU32 index = i * (numY * numZ) + j * numZ + k;

				PxVec4 pos(x, y, z, 1.0f / particleMass);
				phase[index] = particlePhase;
				positionInvMass[index] = pos;
				velocity[index] = PxVec4(0.0f);

				z += particleSpacing;
			}
			z = position.z;
			y += particleSpacing;
		}
		y = position.y;
		x += particleSpacing;
	}

	ExtGpu::PxParticleBufferDesc bufferDesc;
	bufferDesc.maxParticles = maxParticles;
	bufferDesc.numActiveParticles = maxParticles;

	bufferDesc.positions = positionInvMass;
	bufferDesc.velocities = velocity;
	bufferDesc.phases = phase;

	gParticleBuffer = physx::ExtGpu::PxCreateAndPopulateParticleBuffer(bufferDesc, cudaContextManager);
	gParticleSystem->addParticleBuffer(gParticleBuffer);

	cudaContextManager->freePinnedHostBuffer(positionInvMass);
	cudaContextManager->freePinnedHostBuffer(velocity);
	cudaContextManager->freePinnedHostBuffer(phase);
}

static void initObstacles()
{
	const float playgroundSize = 5.0f;
	gScene->addActor(*PxCreatePlane(*gPhysics, PxPlane(0.f, 1.f, 0.f, 0.0f), *gMaterial));  // ground plane
	gScene->addActor(*PxCreatePlane(*gPhysics, PxPlane(0.f, -1.f, 0.f, 100.0f), *gMaterial));  // sky plane
	gScene->addActor(*PxCreatePlane(*gPhysics, PxPlane(1.f, 0.f, 0.f, playgroundSize), *gMaterial));  // x plane
	gScene->addActor(*PxCreatePlane(*gPhysics, PxPlane(-1.f, 0.f, 0.f, playgroundSize), *gMaterial));  // -x plane
	gScene->addActor(*PxCreatePlane(*gPhysics, PxPlane(0.f, 0.f, 1.f, playgroundSize), *gMaterial));  // z plane
	gScene->addActor(*PxCreatePlane(*gPhysics, PxPlane(0.f, 0.f, -1.f, playgroundSize), *gMaterial));  // -z plane
}

static void initControllers()
{
	PxShape* controllerShape = gPhysics->createShape(PxSphereGeometry(0.5f), *gMaterial);

	for (int i = 0; i < 2; ++i)
	{
		PxRigidDynamic* body = gPhysics->createRigidDynamic(PxTransform(PxVec3(0.0f, 0.5f, 0.0f)));
		body->attachShape(*controllerShape);
		body->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
		gScene->addActor(*body);
		gControllerBodies[i] = body;
	}
	controllerShape->release();
}

void initPhysxScene()
{
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

	initObstacles();
	initControllers();
	initParticles(ParticleTypes::eWater);

	// Setup rigid bodies
	// const PxReal boxSize = 0.75f;
	// const PxReal boxMass = 0.25f;
	// PxShape* shape = gPhysics->createShape(PxBoxGeometry(0.5f * boxSize, 0.5f * boxSize, 0.5f * boxSize), *gMaterial);
	// for (int i = 0; i < 5; ++i)
	// {
	// 	PxRigidDynamic* body = gPhysics->createRigidDynamic(PxTransform(PxVec3(i - 2.0f, 10, 0.f)));
	// 	body->attachShape(*shape);
	// 	PxRigidBodyExt::updateMassAndInertia(*body, boxMass);
	// 	gScene->addActor(*body);
	// }
	// shape->release();
}

void initPhysx(){
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true);
	
	PxCudaContextManager* cudaContextManager = NULL;
	if (PxGetSuggestedCudaDeviceOrdinal(gFoundation->getErrorCallback()) >= 0)
	{
		// initialize CUDA
		PxCudaContextManagerDesc cudaContextManagerDesc;
		cudaContextManager = PxCreateCudaContextManager(*gFoundation, cudaContextManagerDesc, PxGetProfilerCallback());
		if (cudaContextManager && !cudaContextManager->contextIsValid())
		{
			cudaContextManager->release();
			cudaContextManager = NULL;
		}
	}
	if (cudaContextManager == NULL)
	{
		PxGetFoundation().error(PxErrorCode::eINVALID_OPERATION, PX_FL, "Failed to initialize CUDA!\n");
	}

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher = PxDefaultCpuDispatcherCreate(2);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;
	sceneDesc.cudaContextManager = cudaContextManager;
	sceneDesc.staticStructure = PxPruningStructureType::eDYNAMIC_AABB_TREE;
	sceneDesc.flags |= PxSceneFlag::eENABLE_PCM;
	sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
	sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
	sceneDesc.solverType = PxSolverType::eTGS;
	gScene = gPhysics->createScene(sceneDesc);

	initPhysxScene();
}

void onBeforeRenderParticles()
{
	if (gParticleSystem)
	{
		PxVec4* positions = gParticleBuffer->getPositionInvMasses();
		const PxU32 numParticles = gParticleBuffer->getNbActiveParticles();

		PxScene* scene;
		PxGetPhysics().getScenes(&scene, 1);
		PxCudaContextManager* cudaContextManager = scene->getCudaContextManager();

		cudaContextManager->acquireContext();

		PxCudaContext* cudaContext = cudaContextManager->getCudaContext();

		// TODO: replace with memcpyDtoD? directly copy to cuda buffer? even necessary?
		// cudaContext->memcpyDtoH(sPosBuffer.map(), CUdeviceptr(positions), sizeof(PxVec4) * numParticles);

		cudaContextManager->releaseContext();
	}
}

void updateActor(PxRigidDynamic* actor, PxTransform targetPose)
{

	if (ovr && ovr->isActive() && ovr->getLeftControllerPose().valid) {
		// Pose poseLeft = ovr->getLeftControllerPose();
		Pose pose = ovr->getLeftControllerPose();

		//PxTransform phsyxPose;
		glm::mat4 transform = pose.transform;

		// glm::dvec3 worldPos = ...;
		// glm::quat rot = glm::quat_cast(transform);

		glm::vec3 scale;
		glm::quat rotation;
		glm::vec3 translation;
		glm::vec3 skew;
		glm::vec4 perspective;
		glm::decompose(transform, scale, rotation, translation, skew, perspective);


		auto physx_world = PxVec3(translation.x, translation.y, translation.z);
		auto physx_rot   = PxQuat(rotation.x, rotation.y, rotation.z, rotation.w);
		
		PxTransform phsyxPose(physx_world, physx_rot);

		// memcpy(&uniforms.vr_right_controller_pose, &transform, sizeof(transform));

		actor->setKinematicTarget(phsyxPose);
		
	} else {
		actor->setKinematicTarget(targetPose);
	}


}

void animateActorToTarget(PxRigidDynamic* actor, const PxVec3& targetPos, const PxReal dt)
{
	const PxVec3 actorPos = actor->getGlobalPose().p;
	const PxVec3 dir = targetPos - actorPos;
	const PxReal dist = dir.magnitude();
	const PxReal maxSpeed = 2.0f;
	const PxReal maxDistDelta = maxSpeed * dt;
	if (dist > maxDistDelta)
		updateActor(actor, PxTransform(actorPos + dir * (maxDistDelta / dist)));
	else
		updateActor(actor, PxTransform(targetPos));
}

void handlePhysicsInputs(PxReal dt)
{
	float dist = 4.0f;
	float height = -0.3f;
	std::vector<PxVec3> targetPositions =
	{
			PxVec3(0.0f, height, 0.0f),
			PxVec3(-dist, height, -dist),
			PxVec3(dist, height, dist),
			PxVec3(0.0f, height, -dist),
			PxVec3(0.0f, height, -dist),
			PxVec3(dist, height, -dist)
	};
	static uint32_t targetPosId = 0;
	static PxReal dtAccu = 0;

	dtAccu += dt;
	if (dtAccu > 3.0f)
	{
		dtAccu = 0;
		targetPosId++;
	}

	// for (int controllerId = 0; controllerId < 2; ++controllerId)
	// {
	// 	if (gControllerBodies[controllerId])
	// 	{	
	// 		uint32_t targetPosIdController = targetPosId + controllerId % targetPositions.size();
	// 		animateActorToTarget(gControllerBodies[controllerId], targetPositions[targetPosIdController], dt);
	// 	}
	// }

	if (ovr && ovr->isActive()) {

		if(ovr->getLeftControllerPose().valid){
			Pose pose = ovr->getLeftControllerPose();

			glm::mat4 transform = pose.transform;

			glm::vec3 scale;
			glm::quat rotation;
			glm::vec3 translation;
			glm::vec3 skew;
			glm::vec4 perspective;
			glm::decompose(transform, scale, rotation, translation, skew, perspective);

			auto physx_world = PxVec3(translation.x, translation.y, translation.z);
			auto physx_rot   = PxQuat(rotation.x, rotation.y, rotation.z, rotation.w);
			
			PxTransform phsyxPose(physx_world, physx_rot);

			auto actor = gControllerBodies[0];
			actor->setKinematicTarget(phsyxPose);
		}

		if(ovr->getLeftControllerPose().valid){
			Pose pose = ovr->getRightControllerPose();

			glm::mat4 transform = pose.transform;
			glm::vec3 scale;
			glm::quat rotation;
			glm::vec3 translation;
			glm::vec3 skew;
			glm::vec4 perspective;
			glm::decompose(transform, scale, rotation, translation, skew, perspective);


			auto physx_world = PxVec3(translation.x, translation.y, translation.z);
			auto physx_rot   = PxQuat(rotation.x, rotation.y, rotation.z, rotation.w);
			
			PxTransform phsyxPose(physx_world, physx_rot);

			auto actor = gControllerBodies[1];
			actor->setKinematicTarget(phsyxPose);
		}

	}



	/*
	PxCudaContextManager* cudaContextManager = gScene->getCudaContextManager();
	cudaContextManager->acquireContext();

	PxU32 numActiveParticles = gParticleBuffer->getNbActiveParticles();
	PxVec4* posInvMass = gParticleBuffer->getPositionInvMasses();
	PxVec4* velocities = gParticleBuffer->getVelocities();

	// TODO: try PxVec4* posInvMassHost = cudaContextManager->allocPinnedHostBuffer<PxVec4>(numActiveParticles);
	PxVec4* posInvMassHost = new PxVec4[numActiveParticles];

	PxCudaContext* cudaContext = cudaContextManager->getCudaContext();
	cudaContext->memcpyDtoH(posInvMassHost, CUdeviceptr(posInvMass), numActiveParticles * sizeof(PxVec4));

	static PxReal particlesAccu = 0.0f;
	static PxReal particlesPerSecond = 500.0f;
	particlesAccu += particlesPerSecond * dt;
	uint32_t particlesThisStep = int(particlesAccu);
	for (PxU32 i = 0; i < particlesThisStep; ++i)
	{
		particlesAccu -= 1.0f;
		auto moveParticleId = std::rand() % gParticleBuffer->getNbActiveParticles();
		PxTransform targetPose = gControllerBodies[0]->getGlobalPose();
		//printf("try to set particle\n");
		posInvMassHost[moveParticleId].x = targetPose.p.x;
		posInvMassHost[moveParticleId].y = targetPose.p.y;
		posInvMassHost[moveParticleId].z = targetPose.p.z;
	}

	cudaContext->memcpyHtoDAsync(CUdeviceptr(posInvMass), posInvMassHost, numActiveParticles * sizeof(PxVec4), 0);
	cudaContext->streamSynchronize(0);
	cudaContextManager->releaseContext();
	*/
}

void updatePhysx(shared_ptr<GLRenderer> renderer) 
{
	onBeforeRenderParticles();

	static double accumulatedTime = 0.0f;
	double updateEvery = 1.0 / 120.0;
	const PxReal dt = static_cast<PxReal>(updateEvery);

	accumulatedTime += renderer->timeSinceLastFrame;

	while (accumulatedTime > updateEvery) {
		handlePhysicsInputs(updateEvery);

		// do update for <updateEvery> seconds
		gScene->simulate(dt);
		gScene->fetchResults(true);
		gScene->fetchResultsParticleSystem();

		accumulatedTime -= updateEvery;
	}

}

void renderPhysx() 
{
	// onBeforeRenderParticles(); // really before stepping? not here?

	// TODO: render particles

	PxScene* scene;
	PxGetPhysics().getScenes(&scene, 1);
	PxU32 nbActors = scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC | PxActorTypeFlag::eRIGID_STATIC);
	if (nbActors)
	{
		std::vector<PxRigidActor*> actors(nbActors);
		scene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC | PxActorTypeFlag::eRIGID_STATIC, reinterpret_cast<PxActor**>(&actors[0]), nbActors);
		// TODO: replace with own rendering
		// Snippets::renderActors(&actors[0], static_cast<PxU32>(actors.size()), true);
	}


}

void cleanupPhysics(bool /*interactive*/)
{
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gPhysics);
	PX_RELEASE(gFoundation);
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

		updatePhysx(renderer);

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

			// if(ImGuiCond_FirstUseEver){
			// 	ImGui::SetNextWindowPos(ImVec2(10, 280));
			// 	ImGui::SetNextWindowSize(ImVec2(490, 180));
			// }

			ImGui::Begin("Infos");
			
			ImGui::BulletText("Cuda software rasterizer rendering 25 instances of the spot model \n(5856 triangles, each).");
			ImGui::BulletText("Each cuda block renders one triangle, \nwith each thread processing a different fragment.");
			ImGui::BulletText("Cuda Kernel: rasterizeTriangles/rasterize.cu");
			ImGui::BulletText("Spot model courtesy of Keenan Crane.");

			ImGui::End();
		}

		{ // SETTINGS WINDOW

			// if(ImGuiCond_FirstUseEver){
			// 	ImGui::SetNextWindowPos(ImVec2(10, 280 + 180 + 10));
			// 	ImGui::SetNextWindowSize(ImVec2(490, 230));
			// }

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

	cleanupPhysics(true);

	return 0;
}
