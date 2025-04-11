#include "Application.h"


Application::Application(const std::string windowName)
{
	CreateAppWindow(windowName);
}

void Application::CreateAppWindow(const std::string windowName)
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		fprintf(stderr, "SDL cannot initialize video subsytem\n");
		exit(EXIT_FAILURE);
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 2);

	window = SDL_CreateWindow(windowName.c_str(), 100, 100,
		screenWidth, screenHeight,
		SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

	renderer = new RenderManager(window, &fileDialog, &light);
	orbitalCamera = new OrbitalCamera();
	freeRoamCamera = new FreeRoamCamera();

	renderer->cameras.push_back(freeRoamCamera);
	renderer->cameras.push_back(orbitalCamera);

	inputManager = new InputManager();

	fileDialog.SetTitle("Load CSG Tree");
	fileDialog.SetTypeFilters({ ".txt" });

	oldTime = SDL_GetTicks();

	SDL_GL_SetSwapInterval(1);
}

void Application::Run()
{
	checkGLError();
	while (!quit)
	{
		Input();
		renderer->Render();
		Uint32 newTime = SDL_GetTicks();
		float fps = 1000.0f / (newTime - oldTime);
		oldTime = newTime;
		cyclicFloatBuffer.add(fps);
		if (cyclicFloatBuffer.pos == 0)
			renderer->fps = cyclicFloatBuffer.average();
		checkGLError();
		SDL_GL_SwapWindow(window);
	}
}

void Application::LoadCameraSettings(const std::string& fileName)
{
	try
	{
		freeRoamCamera->LoadCameraSetting(fileName);
	}
	catch (const std::exception& exc)
	{
		fprintf(stderr, "Cannot load camera settings: %s\n", exc.what());
	}
}

bool Application::LoadCSGTree(const std::string& fileName)
{
	try
	{

		std::ifstream inputStream(fileName.c_str(), std::ios::in);
		if (!inputStream.is_open())
		{
			throw std::runtime_error("File not found, or couldn't be open");
		}

		std::stringstream buffer;
		buffer << inputStream.rdbuf();
		inputStream.close();

		CSGTree tmpTree = CSGTree::Parse(buffer.str());
		tree = tmpTree;

		renderer->SetTreeToRender(tree);
	}
	catch (const std::exception& exc)
	{
		fprintf(stderr, "Cannot load tree: %s\n", exc.what());
	}
}

void Application::SaveSettings()
{
	freeRoamCamera->SaveCameraSetting("camera.ini");
}

void Application::Input()
{

	inputManager->Input();
	quit = inputManager->quit;

	if (fileDialog.HasSelected())
	{
		LoadCSGTree(fileDialog.GetSelected().string());
		fileDialog.ClearSelected();
	}

	if (renderer->activeCam != nullptr)
	{
		renderer->activeCam->HandleInput(inputManager->camControls, inputManager->mouseControls);
	}

	inputManager->mouseControls.relativeX = 0;
	inputManager->mouseControls.relativeY = 0;
}

void Application::checkGLError() {
	GLenum err;
	while ((err = glGetError()) != GL_NO_ERROR) {
		std::cerr << "OpenGL error: " << err << std::endl;
	}
}

void Application::CleanUp()
{
	renderer->CleanUp();
	delete renderer;
	delete inputManager;
	SDL_DestroyWindow(window);
}


