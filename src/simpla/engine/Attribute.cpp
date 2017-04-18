//
// Created by salmon on 16-10-20.
//

#include "Attribute.h"
#include <simpla/data/DataBlock.h>
#include <set>
#include <typeindex>
#include "Domain.h"
#include "Mesh.h"
#include "MeshBlock.h"
#include "Patch.h"
namespace simpla {
namespace engine {

struct AttributeGroup::pimpl_s {
    Mesh const *m_mesh_;
    std::set<Attribute *> m_attributes_;
};

AttributeGroup::AttributeGroup() : m_pimpl_(new pimpl_s) {}
AttributeGroup::~AttributeGroup() {}
void AttributeGroup::Register(AttributeGroup *other) {
    for (Attribute *attr : m_pimpl_->m_attributes_) { other->Attach(attr); }
};
void AttributeGroup::Deregister(AttributeGroup *other) {
    for (Attribute *attr : m_pimpl_->m_attributes_) { other->Detach(attr); }
};
void AttributeGroup::Push(Patch *p) {
    for (auto *item : m_pimpl_->m_attributes_) { item->PushData(p->Pop(item->GetGUID())); }
}
void AttributeGroup::Pop(Patch *p) {
    for (auto *item : m_pimpl_->m_attributes_) { p->Push(item->GetGUID(), item->PopData()); }
}
void AttributeGroup::Attach(Attribute *p) { m_pimpl_->m_attributes_.emplace(p); }
void AttributeGroup::Detach(Attribute *p) { m_pimpl_->m_attributes_.erase(p); }

std::set<Attribute *> const &AttributeGroup::GetAll() const { return m_pimpl_->m_attributes_; }

struct Attribute::pimpl_s {
    std::set<AttributeGroup *> m_bundle_;
    Mesh const *m_mesh_ = nullptr;
    Range<EntityId> m_range_;
};
//Attribute::Attribute(std::shared_ptr<data::DataTable> const &t) : m_pimpl_(new pimpl_s), data::Configurable(t) {
//    SetUp();
//}

Attribute::Attribute(AttributeGroup *b, std::shared_ptr<data::DataTable> const &t)
    : m_pimpl_(new pimpl_s), data::Configurable(t) {
    Register(b);
    SetUp();
};
// Attribute::Attribute(Attribute const &other) : m_pimpl_(new pimpl_s)  {
//    for (auto *b : other.m_pimpl_->m_bundle_) { Register(b); }
//    m_pimpl_->m_mesh_ = other.m_pimpl_->m_mesh_;
//}
// Attribute::Attribute(Attribute &&other) : m_pimpl_(new pimpl_s)  {
//    for (auto *b : other.m_pimpl_->m_bundle_) { Register(b); }
//    for (auto *b : m_pimpl_->m_bundle_) { other.Deregister(b); }
//    m_pimpl_->m_mesh_ = other.m_pimpl_->m_mesh_;
//}
Attribute::~Attribute() {
    for (auto *b : m_pimpl_->m_bundle_) { Deregister(b); }
}

void Attribute::Register(AttributeGroup *attr_b) {
    if (attr_b != nullptr) {
        auto res = m_pimpl_->m_bundle_.emplace(attr_b);
        if (res.second) { attr_b->Attach(this); }
    }
}
void Attribute::Deregister(AttributeGroup *attr_b) {
    if (m_pimpl_->m_bundle_.erase(attr_b) > 0) { attr_b->Detach(this); };
}

void Attribute::SetMesh(Mesh const *m) { m_pimpl_->m_mesh_ = m; }
Mesh const *Attribute::GetMesh() const { return m_pimpl_->m_mesh_; }

// void Attribute::SetRange(Range<EntityId> const &r) { m_pimpl_->m_range_ = r; }
// Range<EntityId> const &Attribute::GetRange() const { return m_pimpl_->m_range_; }

bool Attribute::isNull() const { return false; }
void Attribute::SetUp() {
    SetName(db()->GetValue<std::string>("name", ""));
    SPObject::SetUp();
};

}  //{ namespace engine
}  // namespace simpla
