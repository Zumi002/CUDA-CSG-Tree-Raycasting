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

	lastFrame = Clock::now();

	SDL_GL_SetSwapInterval(0);
}

void Application::Run()
{
	checkGLError();
	lastFrame = Clock::now();
	while (!quit)
	{
		if (isInTestMode)
		{
			auto now = std::chrono::steady_clock::now();
			elapsed = std::chrono::duration<float>(now - start).count();
			if (elapsed > MAX_TEST_TIME)
			{
				quit = true;
			}
		}
		Input();
		renderer->Render();
		TimePoint now = Clock::now();
		std::chrono::duration<float> frameDuration = now - lastFrame;
		lastFrame = now;
		float fps = 1.0f / frameDuration.count();
		cyclicFloatBuffer.add(fps);
		renderer->fps = fps;
		if(cyclicFloatBuffer.pos == 0)
		renderer->avgFps = cyclicFloatBuffer.average();
		checkGLError();
		SDL_GL_SwapWindow(window);
		if (isInTestMode)
		{
			if (collectsStatistics)
			{
				if (additionalStatistics.pixelsWithHit)
				{
					primitivesHits += (long long)additionalStatistics.primitivesHit;
					pixelThatHit += (long long)additionalStatistics.pixelsWithHit;
				}
			}
			elapsed = std::chrono::duration<float>(now - start).count();
			if (elapsed > MAX_TEST_TIME)
			{
				quit = true;
			}
		}
		if (saveResults)
		{
			fpsSamples.emplace_back(fps);
		}
	}
	if (saveResults)
	{
		std::sort(fpsSamples.begin(), fpsSamples.end());
		int onePercent = (int)((float)fpsSamples.size() * 0.01);
		float onePercentSum = 0;
		for (int i = 0; i < onePercent; i++)
		{
			onePercentSum += fpsSamples[i];
		}

		float sum = 0;
		for (int i = 0; i < fpsSamples.size(); i++)
		{
			sum += fpsSamples[i];
		}

		result = new BenchmarkResults(tree.treeName);
		result->FPS[renderer->GetRenderingAlgIndex()][0] = onePercent > 0 ? onePercentSum/onePercent : 0;
		result->FPS[renderer->GetRenderingAlgIndex()][1] = fpsSamples.size() > 0 ? sum / fpsSamples.size() : 0;

		if (collectsStatistics)
		{

			result->avgPrimitivesPerPixel = pixelThatHit ? (float)primitivesHits / (float)pixelThatHit : 0;
		}

		csvResults->SaveResult(*result, renderer->GetRenderingAlgIndex(), collectsStatistics);
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
		std::filesystem::path path = fileName;
		tree.treeName = path.stem().string();

		renderer->SetTreeToRender(tree);

		return true;
	}
	catch (const std::exception& exc)
	{
		fprintf(stderr, "Cannot load tree: %s\n", exc.what());
	}
	return false;
}

void Application::SaveSettings()
{
	freeRoamCamera->SaveCameraSetting("camera.ini");
}

void Application::Input()
{

	inputManager->Input();

	quit = quit || (inputManager->quit);

	if (!isInTestMode)
	{
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
	else
	{
		TestModeCameraManipulation();
	}
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
	if (saveResults)
	{
		delete csvResults;
		delete result;
	}
	SDL_DestroyWindow(window);
}

void Application::SetTestMode(int alg)
{
	if (alg < 0 || alg>2)
		return; //should never happen

	renderer->SetTestMode(alg);
	renderer->SelectCamera(1); //orbital
	isInTestMode = true;
	start = std::chrono::steady_clock::now();
}

void Application::TestModeCameraManipulation()
{
	OrbitalCamera* orbitalCamera = (OrbitalCamera*)(renderer->activeCam);
	orbitalCamera->radius = 2*(1.5f + sinf(elapsed / 3));
	orbitalCamera->SetOrbitRotation(360*(1.0+sinf(elapsed / 5)), 90*cosf(elapsed / 4));
	orbitalCamera->MoveCamera();
}

void Application::SetResults(const std::string& fileName)
{
	csvResults = new CSVResults(fileName);
	if (csvResults->error)
	{
		quit = true;
	}
	else if(isInTestMode)
	{
		saveResults = true;
		fpsSamples.reserve(40000);
	}
}

void Application::SetAdditionalStatistics()
{
	if (isInTestMode)
	{
		collectsStatistics = true;
		renderer->CollectStatistics(&additionalStatistics);
	}
}

