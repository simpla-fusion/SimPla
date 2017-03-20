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

    Moo() {}
    DataAttribute<Real, 2, 2> tags0{this, "tags0"};
    DataAttribute<Real> tags{this, "tags"};
    DataAttribute<Real> rho0{this, "rho0", "CHECK"_ = false, "TAG"_ = 12.345};
    DataAttribute<Real> rho1{this, "s"};
    DataAttribute<Real> tE{this};

    void Initialize() final {}
};

struct Foo : public Worker {
    SP_OBJECT_HEAD(Foo, Worker)
    DataAttribute<Real> rho0{this, "rho0", "CHECK"_ = true};
    DataAttribute<Real> E{this, "E", "CHECK"_ = false};
    void Initialize() final {}
    void Process() final {}
};
int main(int argc, char** argv) {
    auto patch = std::make_shared<Patch>();
    DomainView domain;
    domain.SetMesh<Moo>();
    domain.AddWorker<Foo>();
    //    domain.Dispatch(patch);
    domain.Update();
    std::cout << domain << std::endl;
    AttributeDict db;
    domain.Register(db);
    std::cout << db << std::endl;
}