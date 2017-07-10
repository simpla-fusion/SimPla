//
// Created by salmon on 17-2-13.
//

#include <simpla/engine/Domain.h>

#include <simpla/algebra/Field.h>
#include <simpla/data/all.h>
#include "../../to_delete/Mesh.h"
#include <simpla/mesh/SMesh.h>
#include <simpla/mesh/StructuredMesh.h>
#include <simpla/predefine/physics/EMFluid.h>
#include <iostream>
using namespace simpla;
using namespace simpla::engine;
using namespace simpla::data;
template <typename THost>
struct DummyWorker {
    THost* m_host_ = nullptr;
    DummyWorker(THost* h) : m_host_(h) {}
    ~DummyWorker() = default;
    static std::string TypeName() { return "DummyWorker"; }

    Field<THost, Real, VERTEX> foo{m_host_, "name"_ = "foo"};
};
template <typename THost>
struct DummyMesh : public simpla::mesh::StructuredMesh {
    THost* m_host_ = nullptr;
    DummyMesh(THost* h) : m_host_(h) {}
    ~DummyMesh() = default;
    static std::string RegisterName() { return "DummyMesh"; }
    Field<THost, Real, EDGE> foo2{m_host_, "name"_ = "foo2"};
};

int main(int argc, char** argv) {
    typedef engine::Domain<EMFluid, simpla::mesh::SMesh> domain_type;

    domain_type Foo;

    std::cout << domain_type::TypeName() << std::endl;
    auto const& attrs = Foo.GetAllAttributes();
    for (auto const& attr : Foo.GetAllAttributes()) { std::cout << " Field:" << attr.first << std::endl; }
}