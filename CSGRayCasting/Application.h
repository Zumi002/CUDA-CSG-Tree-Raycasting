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
#include "Graphics/RenderManager/Camera/OrbitalCamera.h"
#include "Graphics/RenderManager/Camera/FreeRoamCamera.h"

#define MAX_TEST_TIME 20.f

class Application
{
	int screenWidth = 800,
		screenHeight = 600;

	bool quit = false;
	bool isInTestMode = false;

	std::chrono::time_point<std::chrono::steady_clock> start;
	float elapsed = 0.f;

	SDL_Window* window;
	
	CSGTree tree;

	RenderManager* renderer;
	FreeRoamCamera* freeRoamCamera;
	OrbitalCamera* orbitalCamera;

	InputManager* inputManager;
	ImGui::FileBrowser fileDialog;

	CyclicFloatBuffer<30> cyclicFloatBuffer;

	Uint32 oldTime;

	DirectionalLight light;

	void Input();
	void checkGLError();
	void TestModeCameraManipulation();
	public:
		Application(const std::string windowName);
		void CreateAppWindow(const std::string windowName);
		void Run();
		bool LoadCSGTree(const std::string& fileName);
		void SaveSettings();
		void CleanUp();
		void LoadCameraSettings(const std::string& fileName);
		void SetTestMode(int alg);
};