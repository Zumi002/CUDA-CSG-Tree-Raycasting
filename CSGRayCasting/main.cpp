#include "Application.h"


int main(int argc, char* argv[])
{
    Application app("CSG Tree RayCasting");
    app.Run();
    app.CleanUp();
    return 0;
}