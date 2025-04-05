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

	//cleanup()
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

void Application::Input()
{
	float moveForward = 0, 
		  moveRight = 0,
		  moveUp = 0;

	inputManager->Input();
	quit = inputManager->quit;

	if (inputManager->camControls.forward)
		moveForward += 0.1f;
	if (inputManager->camControls.backward)
		moveForward -= 0.1f;
	if (inputManager->camControls.left)
		moveRight -= 0.1f;
	if (inputManager->camControls.right)
		moveRight += 0.1f;
	if (inputManager->camControls.up)
		moveUp += 0.1f;
	if (inputManager->camControls.down)
		moveUp -= 0.1f;

	
	if (inputManager->mouseControls.pressed)
	{
		float yaw = -0.005f * (inputManager->mouseControls.relativeX),
			  pitch = -0.005f * (inputManager->mouseControls.relativeY);

		renderer->cam.rotate(pitch, yaw);

		inputManager->mouseControls.relativeX = 0;
		inputManager->mouseControls.relativeY = 0;
	}

	renderer->cam.move(moveForward, moveRight, moveUp);

	if (fileDialog.HasSelected())
	{
		LoadCSGTree(fileDialog.GetSelected().string());
		fileDialog.ClearSelected();
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
	SDL_DestroyWindow(window);
}

void Application::SaveCameraPosition()
{
	std::ofstream file("camera.ini");

	if (!file.is_open()) {
		std::cerr << "Failed to open camera.ini for writing!" << std::endl;
		return;
	}

	file << std::fixed << std::setprecision(6); // consistent float formatting

	file << "[Camera]" << std::endl;
	file << "fov=" << renderer->cam.fov << std::endl;
	file << "x=" << renderer->cam.x << std::endl;
	file << "y=" << renderer->cam.y << std::endl;
	file << "z=" << renderer->cam.z << std::endl;
	file << "rotX=" << renderer->cam.rotX << std::endl;
	file << "rotY=" << renderer->cam.rotY << std::endl;

	file << "forward_x=" << renderer->cam.forward[0] << std::endl;
	file << "forward_y=" << renderer->cam.forward[1] << std::endl;
	file << "forward_z=" << renderer->cam.forward[2] << std::endl;

	file << "right_x=" << renderer->cam.right[0] << std::endl;
	file << "right_y=" << renderer->cam.right[1] << std::endl;
	file << "right_z=" << renderer->cam.right[2] << std::endl;

	file << "up_x=" << renderer->cam.up[0] << std::endl;
	file << "up_y=" << renderer->cam.up[1] << std::endl;
	file << "up_z=" << renderer->cam.up[2] << std::endl;

	file.close();

	std::cout << "Camera position saved to camera.ini" << std::endl;
}


void Application::LoadCameraPosition(const std::string& file_name)
{
	std::ifstream file(file_name);

	if (!file.is_open()) {
		std::cerr << "Failed to open camera.ini for reading!" << std::endl;
		return;
	}

	std::string line;
	while (std::getline(file, line)) {
		// Skip comments and section headers
		if (line.empty() || line[0] == '#' || line[0] == '[') continue;

		std::istringstream iss(line);
		std::string key;
		if (std::getline(iss, key, '=')) {
			std::string valueStr;
			if (std::getline(iss, valueStr)) {
				float value = std::stof(valueStr);

				if (key == "fov") renderer->cam.fov = value;
				else if (key == "x") renderer->cam.x = value;
				else if (key == "y") renderer->cam.y = value;
				else if (key == "z") renderer->cam.z = value;
				else if (key == "rotX") renderer->cam.rotX = value;
				else if (key == "rotY") renderer->cam.rotY = value;
				else if (key == "forward_x") renderer->cam.forward[0] = value;
				else if (key == "forward_y") renderer->cam.forward[1] = value;
				else if (key == "forward_z") renderer->cam.forward[2] = value;
				else if (key == "right_x") renderer->cam.right[0] = value;
				else if (key == "right_y") renderer->cam.right[1] = value;
				else if (key == "right_z") renderer->cam.right[2] = value;
				else if (key == "up_x") renderer->cam.up[0] = value;
				else if (key == "up_y") renderer->cam.up[1] = value;
				else if (key == "up_z") renderer->cam.up[2] = value;
			}
		}
	}

	file.close();
}

