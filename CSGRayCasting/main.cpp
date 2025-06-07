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
    std::string resultFileName;
    bool stats = false;

    cliApp.add_option("-f,--file", file, "Path to CSG tree file to load");
    cliApp.add_option("-c,--camera", camera, "Path to camera settings file to load");
    cliApp.add_option("-t,--test", test, "Test algorithm number (0-2)\n0 - Single Hit Algorithm"\
        "\n1 - Classic Algorithm\n2 - Raymarching Algorithm")
        ->check(CLI::Range(0, 2));
    cliApp.add_option("-r,--result", resultFileName, "Path to file where to save benchmark results");
    cliApp.add_flag("-s,--stats", stats, "Collects addtional statistics");

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

        if (file.empty()) 
        {
            fprintf(stdout, "Option --test requires --file to be specified!");
            return EXIT_FAILURE;
        }
        app.SetTestMode(test);
    }

    if (!resultFileName.empty())
    {
        app.SetResults(resultFileName);
    }

    if (stats)
    {
        if (test == -1 || resultFileName.empty())
        {
            fprintf(stdout, "Flag --stats requires --result and --test to be specified!");
            return EXIT_FAILURE;
        }
        app.SetAdditionalStatistics();
    }

    return 0;
}