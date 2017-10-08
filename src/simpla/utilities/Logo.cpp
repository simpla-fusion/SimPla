/**
 * @file simpla_lib.cpp.cpp
 * @author salmon
 * @date 2015-10-21.
 */
#include "simpla/SIMPLA_config.h"

#include <simpla/parallel/MPIComm.h>
#include <cstdlib>
#include <iomanip>
#include <string>
#include "Logo.h"

#include "Log.h"

namespace simpla {

static const char SIMPLA_LOGO[] =
    R"(
             ____ ___ __  __ ____  _
            / ___|_ _|  \/  |  _ \| | __ _
            \___ \| || |\/| | |_) | |/ _` |
             ___) | || |  | |  __/| | (_| |
            |____/___|_|  |_|_|   |_|\__,_|

      Anything that can go wrong, will go wrong.
                                -- Murphy's law

     SimPla, Plasma Simulator
)"
    "\n " COPYRIGHT " Build Date: " __DATE__ " " __TIME__
    "\n ID:" SIMPLA_VERSION_IDENTIFY " Author: " AUTHOR " ";

std::string ShowLogo() { return SIMPLA_LOGO; }

std::string ShowVersion() { return SIMPLA_VERSION_IDENTIFY; }

void TheStart(int flag) {
#ifdef MPI_FOUND
    if (GLOBAL_COMM.rank() != 0) { return; }
#endif  // MPI_FOUND
    switch (flag) {
        default:
            VERBOSE << SINGLELINE;
            INFORM << "[MISSION     START]";
    }
}

void TheEnd(int flag) {
#ifdef MPI_FOUND
    if (GLOBAL_COMM.rank() != 0) { return; }
#endif  // MPI_FOUND

    switch (flag) {
        case -2:
            INFORM << "Oops! Some thing wrong! Don't worry, maybe not your fault! "
                   << "Please Check your configure file again! ";
            break;
        case -1:
            INFORM << "Sorry! I can't help you now! Please, Try again later!";
            break;
        case 0:
            break;
        case 1:
        default:
            INFORM << "[MISSION COMPLETED]";
            VERBOSE << SINGLELINE;
            VERBOSE << "Job is Done!!";
            VERBOSE << SINGLELINE;
    }
    exit(0);
}
}
// namespace simpla