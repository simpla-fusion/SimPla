//
// Created by salmon on 17-2-13.
//

#include <simpla/engine/Domain.h>

#include <iostream>
using namespace simpla;
using namespace simpla::engine;
template <typename THost>
struct DummyWorker {
    static std::string RegisterName() { return "DummyWorker"; }
};
template <typename THost>
struct DummyMesh {
    static std::string RegisterName() { return "DummyMesh"; }
};

int main(int argc, char** argv) {
    typedef engine::Domain<DummyWorker, DummyMesh> domain_type;

    std::cout << domain_type::RegisterName() << std::endl;
}