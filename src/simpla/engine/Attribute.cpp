//
// Created by salmon on 16-10-20.
//

#include "Attribute.h"
#include <simpla/data/DataBlock.h>
#include <set>
#include <typeindex>
#include "Domain.h"
#include "MeshBase.h"
#include "MeshBlock.h"
#include "Patch.h"
namespace simpla {
namespace engine {

struct AttributeGroup::pimpl_s {
    std::set<Attribute *> m_attributes_;
};

AttributeGroup::AttributeGroup() : m_pimpl_(new pimpl_s) {}
AttributeGroup::~AttributeGroup() {}
void AttributeGroup::RegisterAt(AttributeGroup *other) {
    for (Attribute *attr : m_pimpl_->m_attributes_) { attr->RegisterAt(other); }
};
void AttributeGroup::DeregisterFrom(AttributeGroup *other) {
    for (Attribute *attr : m_pimpl_->m_attributes_) { attr->DeregisterFrom(other); }
};
void AttributeGroup::Push(Patch *p) {
    for (auto *item : m_pimpl_->m_attributes_) { item->Push(p->Pop(item->GetGUID())); }
}
void AttributeGroup::Pop(Patch *p) {
    for (auto *item : m_pimpl_->m_attributes_) { p->Push(item->GetGUID(), item->Pop()); }
}
void AttributeGroup::Attach(Attribute *p) { m_pimpl_->m_attributes_.emplace(p); }
void AttributeGroup::Detach(Attribute *p) { m_pimpl_->m_attributes_.erase(p); }

std::set<Attribute *> const &AttributeGroup::GetAll() const { return m_pimpl_->m_attributes_; }

struct Attribute::pimpl_s {
    std::set<AttributeGroup *> m_bundle_;
};

Attribute::Attribute(std::shared_ptr<data::DataTable> const &t) : m_pimpl_(new pimpl_s), data::Configurable(t) {
    SetUp();
};

Attribute::~Attribute() {
    for (auto *b : m_pimpl_->m_bundle_) { DeregisterFrom(b); }
}

void Attribute::RegisterAt(AttributeGroup *attr_b) {
    if (attr_b != nullptr) {
        auto res = m_pimpl_->m_bundle_.emplace(attr_b);
        if (res.second) { attr_b->Attach(this); }
    }
}
void Attribute::DeregisterFrom(AttributeGroup *attr_b) {
    if (m_pimpl_->m_bundle_.erase(attr_b) > 0) { attr_b->Detach(this); };
}

bool Attribute::isNull() const { return false; }
void Attribute::SetUp() {
    std::string default_name = std::string("_") + std::to_string(GetGUID());
    SetName(db()->GetValue<std::string>("name", default_name));
    SPObject::SetUp();
};

}  //{ namespace engine
}  // namespace simpla
