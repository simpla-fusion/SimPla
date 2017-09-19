//
// Created by salmon on 17-9-4.
//
#include <simpla/utilities/Logo.h>
#include "simpla/parallel/MPIComm.h"
#include "simpla/parallel/Parallel.h"
#include "simpla/utilities/Log.h"
namespace simpla {
int Initialize(int argc, char **argv) {
#ifndef NDEBUG
    logger::set_stdout_level(1000);
#endif

    parallel::Initialize(argc, argv);

    GLOBAL_COMM.barrier();
    MESSAGE << std::endl << ShowLogo() << std::endl;
    GLOBAL_COMM.barrier();
    return SP_SUCCESS;
}
int Finalize() { return SP_SUCCESS; }
}  // namespace simpla