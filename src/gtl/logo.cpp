/**
 * @file simpla_lib.cpp.cpp
 * @author salmon
 * @date 2015-10-21.
 */
#include "../sp_config.h"

#include "logo.h"
#include <cstdlib>
#include <iomanip>
#include <string>

#include "Log.h"

namespace simpla
{
#ifndef IDENTIFY
#	define IDENTIFY "UNKNOWN"
#endif
static const char SIMPLA_LOGO[] = "\n"
        "\t        ____ ___ __  __ ____  _       \n"
        "\t       / ___|_ _|  \\/  |  _ \\| | __ _ \n"
        "\t       \\___ \\| || |\\/| | |_) | |/ _` |\n"
        "\t        ___) | || |  | |  __/| | (_| |\n"
        "\t       |____/___|_|  |_|_|   |_|\\__,_|\n"
        "\n"
        "\t Anything that can go wrong, will go wrong. \n"
        "\t                           -- Murphy's law \n"
        "\n"
        " SimPla, Plasma Simulator  \n"
        " " COPYRIGHT "\n"
        " Build Date: " __DATE__ " " __TIME__"                   \n"
        " ID:" IDENTIFY "\n"
        " Author: " AUTHOR "\n";


std::string ShowLogo()
{
    return SIMPLA_LOGO;
}

std::string ShowVersion()
{
    return IDENTIFY;
}

std::string ShowCopyRight()
{
    return SIMPLA_LOGO;
}

void TheStart(int flag)
{
    switch (flag)
    {
        default:
            INFORM << SINGLELINE << std::endl;
            VERBOSE << "So far so good, let's start work! " << std::endl;
            INFORM << "[MISSOIN     START]: " << std::endl;
            INFORM << SINGLELINE << std::endl;
    }
}

void TheEnd(int flag)
{
    switch (flag)
    {
        case -2:
            INFORM << "Oop! Some thing wrong! Don't worry, maybe not your fault! " << std::endl
            << " Just maybe! Please Check your configure file again! " << std::endl;
            break;
        case -1:
            INFORM << "Sorry! I can't help you now! Please, Try again later!" << std::endl;
            break;
        case 0:
            break;
        case 1:
        default:
            LOGGER << "MISSION COMPLETED!" << std::endl;

            INFORM << SINGLELINE << std::endl;
            INFORM << "[MISSION COMPLETED]: " << std::endl;
            VERBOSE << "Job is Done!! " << std::endl;
            VERBOSE << "	I'm so GOOD!" << std::endl;
            VERBOSE << "		Thanks me please!" << std::endl;
            VERBOSE << "			Thanks me please!" << std::endl;
            VERBOSE << "You are welcome!" << std::endl;
            INFORM << SINGLELINE << std::endl;

    }
    exit(0);
}
}
// namespace simpla