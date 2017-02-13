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

struct AttrData : public AttributeView {
    template <typename... Args>
    AttrData(DomainView* d, Args&&... args) : AttributeView(d, std::forward<Args>(args)...) {}
    AttrData(DomainView* d, std::initializer_list<simpla::data::KeyValue> const& param) : AttributeView(d) {
        description().db.insert(param);
    }

    ~AttrData() {}
    AttributeDesc& description() { return m_desc_; }
    AttributeDesc const& description() const { return m_desc_; }
    AttributeDesc m_desc_;
};
struct Foo : public DomainView {
    AttrData F{this, {"name"_ = "rho0", "CHECK"}};
    AttrData EF{this, {"name"_ = "E", "CHECK"}};
};
int main(int argc, char** argv) {
    Foo domain;

    domain.F.Update();
    std::cout << domain << std::endl;
    std::cout << domain.F.description() << std::endl;
}