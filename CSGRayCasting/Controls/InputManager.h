#pragma once

#include <SDL.h>

#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_opengl3.h"
#include "../imgui/imgui_impl_sdl2.h"

struct CameraControls
{
	int forward = 0;
	int backward = 0;
	int right = 0;
	int left = 0;
	int up = 0;
	int down = 0;
};

struct MouseControls
{
	bool pressed = false;
	int relativeX = 0;
	int relativeY = 0;
};

struct InputManager
{
	CameraControls camControls;
	MouseControls mouseControls;
	bool quit = false;
	

public:
	void Input();

};