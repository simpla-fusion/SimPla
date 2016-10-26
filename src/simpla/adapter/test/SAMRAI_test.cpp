//
// Created by salmon on 16-10-24.
//

#include <simpla/adapter/Adapter.h>


using namespace simpla;

int main(int argc, char **argv)
{
    auto ctx = simpla::create_context("SAMRAIWorkerHyperbolic");


    ctx->setup(argc, argv);
    ctx->deploy();
    ctx->next_time_step(1.0);
    ctx->teardown();
    exit(0);
}