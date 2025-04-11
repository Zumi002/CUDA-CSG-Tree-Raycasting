#include "InputManager.h"

void InputManager::Input()
{
	SDL_Event e;

	// Process events
	while (SDL_PollEvent(&e) != 0)
	{

		ImGui_ImplSDL2_ProcessEvent(&e);
		if (e.type == SDL_QUIT)
		{
			quit = true;
		}
		if (e.type == SDL_MOUSEMOTION)
		{
			mouseControls.relativeX = e.motion.xrel;
			mouseControls.relativeY = e.motion.yrel;
		}
		if (e.type == SDL_MOUSEBUTTONDOWN && !ImGui::GetIO().WantCaptureMouse)
		{
			if (e.button.button == SDL_BUTTON_LEFT)
			{
				mouseControls.relativeX = 0;
				mouseControls.relativeY = 0;
				mouseControls.pressed = true;
				SDL_SetRelativeMouseMode(SDL_TRUE);
			}
		}
		if (e.type == SDL_MOUSEBUTTONUP && !ImGui::GetIO().WantCaptureMouse)
		{
			if (e.button.button == SDL_BUTTON_LEFT)
			{
				mouseControls.pressed = false;
				SDL_SetRelativeMouseMode(SDL_FALSE);
			}
		}
		if ((e.type == SDL_KEYDOWN || e.type == SDL_KEYUP) &&
			e.key.repeat == 0)
		{
			SDL_Keycode key = e.key.keysym.sym;
			int is_pressed = e.key.state == SDL_PRESSED;

			if (key == SDLK_w)
			{
				if (is_pressed)
					camControls.forward = 1;
				else
					camControls.forward = 0;
			}
			if (key == SDLK_s)
			{
				if (is_pressed)
					camControls.backward = 1;
				else
					camControls.backward = 0;
			}
			if (key == SDLK_a)
			{
				if (is_pressed)
					camControls.left = 1;
				else
					camControls.left = 0;
			}
			if (key == SDLK_d)
			{
				if (is_pressed)
					camControls.right = 1;
				else
					camControls.right = 0;
			}
			if (key == SDLK_SPACE)
			{
				if (is_pressed)
					camControls.up = 1;
				else
					camControls.up = 0;
			}
			if (key == SDLK_LSHIFT)
			{
				if (is_pressed)
					camControls.down = 1;
				else
					camControls.down = 0;
			}

		}
	}
}