//
// Created by salmon on 17-2-13.
//
#include <simpla/data/DataTable.h>
#include <simpla/engine/AttributeDesc.h>
#include <simpla/engine/AttributeView.h>
#include <simpla/engine/DomainView.h>
#include <simpla/engine/Worker.h>
#include <iostream>
using namespace simpla::engine;
using namespace simpla::data;

// struct AttrData : public AttributeView {
//    template <typename... Args>
//    AttrData(Args&&... args) : AttributeView(std::forward<Args>(args)...) {}
//    AttrData(DomainView* d, std::initializer_list<simpla::data::KeyValue> const& param) : AttributeView(d, param) {}
//    ~AttrData() {}
//};
struct Moo : public MeshView {
    SP_OBJECT_HEAD(Moo, MeshView)
    AttributeView tags{this, "tags", {"CHECK"_ = true}};
    void Initialize() final {}
};

struct Foo : public Worker {
    SP_OBJECT_HEAD(Foo, Worker)

    AttributeView F{this, "rho0", {"CHECK"_ = true}};
    AttributeView EF{this, "E", {"CHECK"_ = false}};

    void Initialize() final {}
    void Process() final {}
};
int main(int argc, char** argv) {
    DomainView domain;
    domain.SetMesh<Moo>();
    domain.AppendWorker<Foo>();
    domain.Update();
    std::cout << domain << std::endl;
}