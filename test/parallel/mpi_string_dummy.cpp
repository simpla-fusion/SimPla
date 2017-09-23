//
// Created by salmon on 17-9-23.
//

#include <simpla/parallel/MPIComm.h>
#include <simpla/parallel/Parallel.h>
#include <iostream>
#include "simpla/SIMPLA_config.h"
using namespace simpla;
int main(int argc, char** argv) {
    parallel::Initialize(argc, argv);
    std::cout << "[" << GLOBAL_COMM.rank() << "]"
              << parallel::gather_string("rank[" + std::to_string(GLOBAL_COMM.rank()) + "]", 2) << std::endl;

    std::cout << "[" << GLOBAL_COMM.rank() << "]"
              << parallel::gather_string("rank[" + std::to_string(GLOBAL_COMM.rank()) + "]", -1) << std::endl;
    parallel::Finalize();
}