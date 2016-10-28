//
// Created by salmon on 16-10-24.
//
#include <simpla/toolbox/Log.h>
#include <simpla/adapter/Adapter.h>


int main(int argc, char **argv)
{
    using namespace simpla;

    auto ctx = simpla::create_context("SAMRAIWorkerHyperbolic");

    MESSAGE << "START\n";
    ctx->initialize(argc, argv);
    ctx->load(nullptr);
    ctx->deploy();
    ctx->next_time_step(1.0);
    ctx->teardown();

    MESSAGE << "DONE\n";
    exit(0);
}