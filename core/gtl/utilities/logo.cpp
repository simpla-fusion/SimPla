/**
 * @file simpla_lib.cpp.cpp
 * @author salmon
 * @date 2015-10-21.
 */

#include "logo.h"
#include <cstdlib>
#include <iomanip>
#include <string>

#include "log.h"

namespace simpla {

#ifndef IDENTIFY
#	define IDENTIFY "UNKNOWN"
#endif

#define  AUTHOR " YU Zhi <yuzhi@ipp.ac.cn> "

#ifndef COPYRIGHT
#define  COPYRIGHT "All rights reserved. (2009-2015 )"
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
    switch (flag) {
        default:
            INFORM << SINGLELINE;
            VERBOSE << "So far so good, let's start work! ";
            INFORM << "[MISSOIN     START]: ";
            INFORM << SINGLELINE;
    }
}

void TheEnd(int flag)
{
    switch (flag) {
        case -2:
            INFORM << "Oop! Some thing wrong! Don't worry, maybe not your fault!\n"
                    " Just maybe! Please Check your configure file again! ";
            break;
        case -1:
            INFORM << "Sorry! I can't help you now! Please, Try again later!";
            break;
        case 0:
            break;
        case 1:
        default:
            LOGGER << "MISSION COMPLETED!";

            INFORM << SINGLELINE;
            INFORM << "[MISSION COMPLETED]: ";
            VERBOSE << "Job is Done!! ";
            VERBOSE << "	I'm so GOOD!";
            VERBOSE << "		Thanks me please!";
            VERBOSE << "			Thanks me please!";
            VERBOSE << "You are welcome!";
            INFORM << SINGLELINE;

    }
    exit(0);
}
}
// namespace simpla