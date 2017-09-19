//
// Created by salmon on 17-9-19.
//
#include "simpla/parallel/MPIUpdater.h"
#include <simpla/parallel/MPIComm.h>
#include <simpla/parallel/Parallel.h>
#include "simpla/SIMPLA_config.h"
#include "simpla/utilities/SPDefines.h"
using namespace simpla;

int main(int argc, char** argv) {
    parallel::Initialize(argc, argv);
    index_box_type box{{4, 5, 6}, {10, 12, 14}};
    index_tuple gw{2, 2, 2};

    Array<Real> d(nullptr, box);

    auto updater = parallel::MPIUpdater::New<double>();
    updater->SetIndexBox(box);
    updater->SetGhostWidth(gw);
    updater->SetUp();

    d.SetUp();
    d = GLOBAL_COMM.rank();
    std::cout << d << std::endl;
    updater->Push(d);
    updater->Update();
    updater->Pop(d);
    std::cout << d << std::endl;

    parallel::Finalize();
}