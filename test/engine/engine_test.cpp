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

struct Moo : public MeshView, public AttributeViewBundle {
    SP_OBJECT_HEAD(Moo, MeshView)
    DataAttribute<Real, 2, 2> tags0{this, "tags0"};
    DataAttribute<Real> tags{this, "tags"};
    DataAttribute<Real> rho0{this, "rho0", NORMAL, "CHECK"_ = false, "TAG"_ = 12.345};
    DataAttribute<Real> rho1{this, "s", NORMAL};
    DataAttribute<Real> tE{this};

    bool Initialize() final { return true; }
};

struct Foo : public Worker {
    SP_OBJECT_HEAD(Foo, Worker)
    DataAttribute<Real> rho0{this, "rho0", NORMAL, "CHECK"_ = true};
    DataAttribute<Real> E{this, "E", NORMAL, "CHECK"_ = false};
    bool Initialize() final {}
    void Process() final {}
};
int main(int argc, char** argv) {
    auto patch = std::make_shared<Patch>();
    DomainView domain;
//    domain.SetMesh<Moo>();
//    domain.AppendWorker<Foo>();
    domain.Dispatch(patch);
    domain.Update();
    std::cout << domain << std::endl;
    AttributeDict db;
    domain.RegisterAttribute(&db);
    std::cout << db << std::endl;
}