#include "RenderManager.h"

RenderManager::RenderManager(SDL_Window* window, ImGui::FileBrowser* FileDialog, DirectionalLight* Light)
{
	this->window = window;
	context = SDL_GL_CreateContext(window);
	if (context == nullptr)
	{
		fprintf(stderr, "Cannot creater openGL context");
		exit(EXIT_FAILURE);
	}

	if (!gladLoadGLLoader(SDL_GL_GetProcAddress))
	{
		fprintf(stderr, "Glad cannot inizialize");
		exit(EXIT_FAILURE);
	}

	// Setup ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	ImGui::StyleColorsDark();

	fileDialog = FileDialog;

	// Setup Platform/Renderer bindings
	ImGui_ImplSDL2_InitForOpenGL(window, context);
	ImGui_ImplOpenGL3_Init("#version 410");

	SDL_GL_GetDrawableSize(window, &width, &height);

	glGenBuffers(1, &rayCastingPBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, rayCastingPBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

	glGenTextures(1, &rayCastingTexture);
	glBindTexture(GL_TEXTURE_2D, rayCastingTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

	// Set texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	
	glGenFramebuffers(1, &framebuffer);

	cudaGraphicsGLRegisterBuffer(&cudaRaycastingPBOResource, rayCastingPBO, cudaGraphicsMapFlagsWriteDiscard);

	activeCam = nullptr;

	raycaster = Raycaster();

	light = Light;
}



void RenderManager::CalculateRays()
{
	if (activeCam == nullptr)
		return;
	cudaGraphicsMapResources(1, &cudaRaycastingPBOResource);
	float4* d_ptr;
	cudaGraphicsResourceGetMappedPointer((void**)&d_ptr, nullptr, cudaRaycastingPBOResource);

	raycaster.Raycast(d_ptr, *activeCam, *light);

	cudaGraphicsUnmapResources(1, &cudaRaycastingPBOResource);
}

void RenderManager::RenderRaysData()
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, rayCastingPBO);
	glBindTexture(GL_TEXTURE_2D, rayCastingTexture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, nullptr);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, rayCastingPBO);

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rayCastingTexture, 0);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); 
	glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RenderManager::RenderImGui()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();

	ImGui::SetNextWindowPos(ImVec2(5, 5));

	ImGui::Begin("CSG RayCasting",0,ImGuiWindowFlags_::ImGuiWindowFlags_AlwaysAutoResize);

	ImGui::Text("FPS: %.1f", fps);
	if (ImGui::Button("Load CSGTree..."))
	{
		fileDialog->Open();
	}
	ImGui::Separator();
	ImGui::Combo("Algorithm", &renderingAlg, algorithmListBoxItems, 2);
	ImGui::Separator();
	ImGui::LabelText("","Camera");
	ImGui::Combo("Type", &cameraIdx, cameraListBoxItems, 2);
	if (ImGui::TreeNode("Settings")&&activeCam!=nullptr)
	{
		int fov = activeCam->fov/3.14f*180.f;
		ImGui::SliderInt("FOV", &fov, 30, 120, "%d deg");
		activeCam->setFOV((float)fov);

		if (activeCam->GetType() == Camera::CameraType::FreeRoamCamera)
		{
			ImGui::SliderFloat("Speed", &(((FreeRoamCamera*)activeCam)->speed), 0.1f, 5.f, "%.2f");
			ImGui::SliderFloat("Sensitivity", &(((FreeRoamCamera*)activeCam)->sensitivity), 0.1f, 5.f, "%.2f");
		}
		else if (activeCam->GetType() == Camera::CameraType::OrbitalCamera)
		{
			ImGui::SliderFloat("Speed", &(((OrbitalCamera*)activeCam)->speed), 0.1f, 5.f, "%.2f");
			ImGui::SliderFloat("Sensitivity", &(((OrbitalCamera*)activeCam)->sensitivity), 0.1f, 5.f, "%.2f");
			ImGui::SliderFloat("Zoom", &(((OrbitalCamera*)activeCam)->radius), 0.01f, 50.f, "%.2f");
		}

		ImGui::TreePop();
	}
	ImGui::Separator();
	ImGui::LabelText("","Light direction:");
	ImGui::SliderAngle("Polar", &light->polar, -180, 180);
	ImGui::SliderAngle("Azimuth", &light->azimuth, -180, 180);

	fileDialog->Display();

	ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void RenderManager::Render()
{
	if (renderingAlg != lastRenderingAlg)
	{
		lastRenderingAlg = renderingAlg;
		ChangeAlgorithm();
	}
	SetActiveCamera();

	int tmpWidth, tmpHeight;
	SDL_GL_GetDrawableSize(window, &tmpWidth, &tmpHeight);

	if (tmpWidth != width || tmpHeight != height)
	{
		width = tmpWidth;
		height = tmpHeight;
		ChangeSize();
	}

	if (treeSet)
	{
		CalculateRays();
		RenderRaysData();
	}
	RenderImGui();
	
}

void RenderManager::ChangeAlgorithm()
{
	if (lastRenderingAlg == 0)
		raycaster.ChangeAlg(tree, lastRenderingAlg);
	if (lastRenderingAlg == 1)
		raycaster.ChangeAlg(classicalTree, lastRenderingAlg);
}

void RenderManager::ChangeTree()
{
	raycaster.ChangeTree(tree);
	ChangeAlgorithm();
}

void RenderManager::ChangeSize()
{
	raycaster.ChangeSize(width, height);

	cudaGraphicsUnregisterResource(cudaRaycastingPBOResource);

	glDeleteTextures(1, &rayCastingTexture);
	glDeleteBuffers(1, &rayCastingPBO);

	glGenBuffers(1, &rayCastingPBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, rayCastingPBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

	glGenTextures(1, &rayCastingTexture);
	glBindTexture(GL_TEXTURE_2D, rayCastingTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	
	cudaGraphicsGLRegisterBuffer(&cudaRaycastingPBOResource, rayCastingPBO, cudaGraphicsMapFlagsWriteDiscard);
}

void RenderManager::SetTreeToRender(CSGTree newTree)
{
	tree = newTree;
	classicalTree = newTree;
	classicalTree.TransformForClassical();
	treeSet = true;
	ChangeSize();
	ChangeTree();
}


void RenderManager::CleanUp()
{
	raycaster.CleanUp();
	cudaGraphicsUnregisterResource(cudaRaycastingPBOResource);

	glDeleteTextures(1, &rayCastingTexture);
	glDeleteBuffers(1, &rayCastingPBO);
}

void RenderManager::SetActiveCamera()
{
	activeCam = cameras[cameraIdx];
}