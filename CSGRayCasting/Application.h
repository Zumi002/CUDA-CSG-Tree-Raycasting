#pragma once

#include <string>
#include <SDL.h>
#include <glad/glad.h>
#include <fstream>
#include <iostream>

#include "Graphics/RenderManager/RenderManager.h"
#include "Graphics/RayCasting/CSGTree/CSGTree.cuh"
#include "Controls/InputManager.h"
#include "Utils/CyclicBuffer.h"
#include "Graphics/RenderManager/Camera/CameraController.h"

class Application
{
	int screenWidth = 800,
		screenHeight = 600;

	bool quit = false;


	SDL_Window* window;
	
	CSGTree tree;

	RenderManager* renderer;
	CameraController* camController;

	InputManager* inputManager;
	ImGui::FileBrowser fileDialog;

	CyclicFloatBuffer<30> cyclicFloatBuffer;

	Uint32 oldTime;

	DirectionalLight light;

	void Input();
	void checkGLError();
	public:
		Application(const std::string windowName);
		void CreateAppWindow(const std::string windowName);
		void Run();
		bool LoadCSGTree(const std::string& fileName);
		void SaveSettings();
		void CleanUp();
		void LoadCameraSettings(const std::string& fileName);
};