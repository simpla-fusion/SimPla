//
// Created by salmon on 17-2-13.
//
#include <simpla/data/DataTable.h>

#include <simpla/engine/AttributeView.h>
#include <simpla/engine/DomainView.h>
#include <simpla/engine/Patch.h>
#include <simpla/engine/Worker.h>

#include <iostream>
using namespace simpla::engine;
using namespace simpla::data;

struct Moo : public MeshView {
    SP_OBJECT_HEAD(Moo, MeshView)
    DataAttribute<Real, 2, 2> tags0{"tags0"};
    DataAttribute<Real> tags{"tags", this};
    void Initialize() final {}
};

struct Foo : public Worker {
    SP_OBJECT_HEAD(Foo, Worker)
    DataAttribute<Real> F{"rho0", this, "CHECK"_ = true};
    DataAttribute<Real> E{"E", this, "CHECK"_ = false};
    void Initialize() final {}
    void Process() final {}
};
int main(int argc, char** argv) {
    auto patch = std::make_shared<Patch>();
    DomainView domain;
    domain.SetMesh<Moo>();
    domain.AppendWorker<Foo>();
    domain.Dispatch(patch);
    domain.Update();
    std::cout << domain << std::endl;
}