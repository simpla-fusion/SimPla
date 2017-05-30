//
// Created by salmon on 17-2-13.
//
#include <simpla/data/DataTable.h>

#include <simpla/engine/Attribute.h>
#include <simpla/engine/Domain.h>
#include <simpla/engine/Patch.h>
#include <simpla/engine/Task.h>

#include <iostream>
using namespace simpla::engine;
using namespace simpla::data;

struct Moo : public MeshBase {
    SP_OBJECT_HEAD(Moo, MeshView)

    Moo() : MeshBase(<#initializer#>) {}
    DataAttribute<Real, 2, 2> tags0{this, "tags0"};
    DataAttribute<Real> tags{this, "tags"};
    DataAttribute<Real> rho0{this, "ne", "CHECK"_ = false, "TAG"_ = 12.345};
    DataAttribute<Real> rho1{this, "s"};
    DataAttribute<Real> tE{this};

    void Initialize() final {}
};

struct Foo : public Task {
    SP_OBJECT_HEAD(Foo, Task)
    DataAttribute<Real> rho0{this, "ne", "CHECK"_ = true};
    DataAttribute<Real> E{this, "E", "CHECK"_ = false};
    void Initialize() final {}
    void Process() final {}
};
int main(int argc, char** argv) {
    auto patch = std::make_shared<Patch>();
    Domain domain(<#initializer#>);
    domain.SetMesh<Moo>();
    domain.SetWorker<Foo>();
    //    domain.Dispatch(patch);
    domain.UpdatePatch(nullptr, 0, 0);
    std::cout << domain << std::endl;
    AttributeDict db;
    domain.Register(db);
    std::cout << db << std::endl;
}