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
    index_box_type box{{0, 0, 0}, {8, 6, 1}};
    index_tuple gw{2, 2, 0};
    index_box_type outer_box;
    std::get<0>(outer_box) = std::get<0>(box) - gw;
    std::get<1>(outer_box) = std::get<1>(box) + gw;

    auto updater = parallel::MPIUpdater::New<double>();
    updater->SetIndexBox(box);
    updater->SetGhostWidth(gw);

    Array<Real> a(outer_box);
    Array<Real> b(box);
    a.FillNaN();
    b.Fill(GLOBAL_COMM.rank());
    auto rank = GLOBAL_COMM.rank();
    b.Foreach([&](auto& v, index_type i, index_type j, index_type k) { v = v * 1000000 + i * 10000 + j * 100 + k; });
    a.CopyIn(b);

    auto center = a.GetSelection(box);

    GLOBAL_COMM.barrier();
    if (GLOBAL_COMM.rank() == 0) { std::cout << a << std::endl; }
    GLOBAL_COMM.barrier();

    for (int dir = 0; dir < 3; ++dir) {
        updater->SetDirection(dir);
        updater->SetUp();
        updater->Push(a);
        updater->SendRecv();
        updater->Pop(a);
        updater->TearDown();
    }

    GLOBAL_COMM.barrier();
    if (GLOBAL_COMM.rank() == 0) { std::cout << a << std::endl; }
    GLOBAL_COMM.barrier();
    if (GLOBAL_COMM.rank() == 1) { std::cout << a << std::endl; }
    GLOBAL_COMM.barrier();
    if (GLOBAL_COMM.rank() == 2) { std::cout << a << std::endl; }
    GLOBAL_COMM.barrier();
    if (GLOBAL_COMM.rank() == 3) { std::cout << a << std::endl; }
    GLOBAL_COMM.barrier();
    parallel::Finalize();
}