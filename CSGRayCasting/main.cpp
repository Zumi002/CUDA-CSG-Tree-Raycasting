#include "Application.h"


int main(int argc, char* argv[])
{
    Application app("CSG Tree RayCasting");
    app.LoadCSGTree("test5.txt");
    app.Run();

    return 0;
}