#pragma once


#include <SDL.h>
#include <stdio.h>
#include <cstdlib>
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "../RayCasting/Raycaster.cuh"
#include "DirectionalLight.h"

#include "../../imgui/imgui.h"
#include "../../imgui/imgui_impl_opengl3.h"
#include "../../imgui/imgui_impl_sdl2.h"
#include "../../imgui/imfilebrowser.h"

class RenderManager
{
	SDL_Window* window;
	SDL_GLContext context;
	Raycaster raycaster;

	int width, height;

	GLuint rayCastingPBO;
	GLuint rayCastingTexture;
	GLuint framebuffer;
	
	cudaGraphicsResource* cudaRaycastingPBOResource;

	CSGTree tree;

	ImGui::FileBrowser* fileDialog;
	DirectionalLight* light;

	void ChangeSize();
	void CalculateRays();
	void RenderRaysData();
	void RenderImGui();

	public:
		float fps = 0;
		Camera cam;
		RenderManager(SDL_Window* window, ImGui::FileBrowser* FileDialog, DirectionalLight* Light);
		void Render();
		void SetTreeToRender(CSGTree tree);
};



