//
// Created by salmon on 17-2-13.
//
#include <simpla/data/DataTable.h>
#include <simpla/engine/AttributeDesc.h>
#include <simpla/engine/AttributeView.h>
#include <simpla/engine/DomainView.h>
#include <iostream>
using namespace simpla::engine;
using namespace simpla::data;

// struct AttrData : public AttributeView {
//    template <typename... Args>
//    AttrData(Args&&... args) : AttributeView(std::forward<Args>(args)...) {}
//    AttrData(DomainView* d, std::initializer_list<simpla::data::KeyValue> const& param) : AttributeView(d, param) {}
//    ~AttrData() {}
//};
struct Foo : public DomainView {
    AttributeView F{this, "rho0", {"CHECK"_ = true}};
    AttributeView EF{this, "E", {"CHECK"_ = false}};
};
int main(int argc, char** argv) {
    Foo domain;

    domain.F.Update();
    std::cout << domain << std::endl;
    std::cout << domain.F.description().name() << " = " << domain.F.db << std::endl;
}