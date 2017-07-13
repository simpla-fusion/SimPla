/**
 * @file simpla_lib.cpp.cpp
 * @author salmon
 * @date 2015-10-21.
 */
#include "simpla/SIMPLA_config.h"

#include <cstdlib>
#include <iomanip>
#include <string>
#include "Logo.h"

#include "Log.h"

namespace simpla {

static const char SIMPLA_LOGO[] =
    "\n"
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
    " " COPYRIGHT
    "\n"
    " Build Date: " __DATE__ " " __TIME__
    "                   \n"
    " ID:" SIMPLA_VERSION_IDENTIFY
    "\n"
    " Author: " AUTHOR "\n";

std::string ShowLogo() { return SIMPLA_LOGO; }

std::string ShowVersion() { return SIMPLA_VERSION_IDENTIFY; }

void TheStart(int flag) {
    switch (flag) {
        default:
            VERBOSE << SINGLELINE << std::endl;
            INFORM << "[MISSION     START]" << std::endl;
    }
}

void TheEnd(int flag) {
    switch (flag) {
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
            INFORM << "[MISSION COMPLETED]" << std::endl;
            VERBOSE << SINGLELINE << std::endl;
            VERBOSE << "Job is Done!!  I'm so GOOD!" << std::endl;
            VERBOSE << "		Thanks me  !" << std::endl;
            VERBOSE << "			Thanks me  !" << std::endl;
            VERBOSE << "You are welcome!" << std::endl;
            VERBOSE << SINGLELINE << std::endl;
    }
    exit(0);
}
}
// namespace simpla