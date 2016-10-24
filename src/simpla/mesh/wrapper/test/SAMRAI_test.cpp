//
// Created by salmon on 16-10-24.
//

#include <simpla/mesh/wrapper/Wrapper.h>


int main(int argc, char **argv)
{
    auto ctx = simpla::create_context("LinAdv");

    ctx->setup(argc, argv);
    ctx->run(1.0);
    ctx->teardown();
    exit(0);
}