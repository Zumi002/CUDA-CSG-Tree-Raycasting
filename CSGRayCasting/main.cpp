#include "Application.h"

void HandleCommandLineArguments(Application& app, int argc, char* argv[]);

int main(int argc, char* argv[])
{
    Application app("CSG Tree RayCasting");

    HandleCommandLineArguments(app, argc, argv);
	
    app.Run();
	app.SaveSettings();
    app.CleanUp();
    return 0;
}

void HandleCommandLineArguments(Application& app, int argc, char* argv[])
{
    if (argc >= 2)
    {
        app.LoadCSGTree(argv[1]);
    }

    if (argc >= 3)
    {
        app.LoadCameraSettings(argv[2]);
    }
}
