#include "CLI11/CLI11.hpp"
#include "Application.h"

int HandleCommandLineArguments(Application& app, int argc, char* argv[]);

int main(int argc, char* argv[])
{
    Application app("CSG Tree RayCasting");
    int code = HandleCommandLineArguments(app, argc, argv);
    if (code != 0)
    {
        return code;
    }

    app.Run();
	app.SaveSettings();
    app.CleanUp();
    return EXIT_SUCCESS;
}

int HandleCommandLineArguments(Application& app, int argc, char* argv[])
{
    CLI::App cliApp{ "CSG Tree RayCaster - Command Line Options" };

    std::string file;
    std::string camera;
    int test = -1; // -1 - no test selected

    cliApp.add_option("-f,--file", file, "Path to CSG tree file to load");
    cliApp.add_option("-c,--camera", camera, "Path to camera settings file to load");
    cliApp.add_option("-t,--test", test, "Test algorithm number (0-2)\n0 - Single Hit Algorithm"\
        "\n1 - Classic Algorithm\n2 - Raymarching Algorithm")
        ->check(CLI::Range(0, 2));

    try {
        (cliApp).parse(argc, argv);
    }
    catch (const CLI::CallForHelp& e)
    {
        fprintf(stdout, cliApp.help().c_str());
        return EXIT_FAILURE;
    }
    catch (const CLI::ParseError& e) 
    {
        return (cliApp).exit(e);
    }

    if (!file.empty() && !app.LoadCSGTree(file)) {
        return EXIT_FAILURE;
    }

    if (!camera.empty()) {
        app.LoadCameraSettings(camera);
    }

    if (test != -1) {

        if (file.empty()) {
            fprintf(stdout, "Option --test requires --file to be specified!");
            return EXIT_FAILURE;
        }
        app.SetTestMode(test);
    }

    return 0;
}