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

struct AttributeBundle::pimpl_s {
    Mesh const *m_mesh_;
    std::set<Attribute *> m_attributes_;
};

AttributeBundle::AttributeBundle() : m_pimpl_(new pimpl_s) {}
AttributeBundle::~AttributeBundle() {}
void AttributeBundle::Register(AttributeBundle *other) {
    for (Attribute *attr : m_pimpl_->m_attributes_) { other->Attach(attr); }
};
void AttributeBundle::Deregister(AttributeBundle *other) {
    for (Attribute *attr : m_pimpl_->m_attributes_) { other->Detach(attr); }
};

void AttributeBundle::Attach(Attribute *p) { m_pimpl_->m_attributes_.emplace(p); }
void AttributeBundle::Detach(Attribute *p) { m_pimpl_->m_attributes_.erase(p); }

void AttributeBundle::Push(const std::shared_ptr<Patch> &p) {
    for (auto *v : m_pimpl_->m_attributes_) { v->PushData(p->Pop(v->GetGUID())); }
}
std::shared_ptr<Patch> AttributeBundle::Pop() {
    auto res = std::make_shared<Patch>();
    for (auto *v : m_pimpl_->m_attributes_) { res->Push(v->GetGUID(), v->PopData()); }
    return res;
}

std::set<Attribute *> const &AttributeBundle::GetAll() const { return m_pimpl_->m_attributes_; }

struct Attribute::pimpl_s {
    std::set<AttributeBundle *> m_bundle_;
    Mesh const *m_mesh_ = nullptr;
    Range<mesh::MeshEntityId> m_range_;
};
Attribute::Attribute(std::shared_ptr<data::DataTable> const &t) : m_pimpl_(new pimpl_s), concept::Configurable(t) {}
Attribute::Attribute(AttributeBundle *b, std::shared_ptr<data::DataTable> const &t)
    : concept::Configurable(t), m_pimpl_(new pimpl_s) {
    Register(b);
};
Attribute::Attribute(Attribute const &other) : m_pimpl_(new pimpl_s), concept::Configurable(other) {
    for (auto *b : other.m_pimpl_->m_bundle_) { Register(b); }
    m_pimpl_->m_mesh_ = other.m_pimpl_->m_mesh_;
}
Attribute::Attribute(Attribute &&other) : m_pimpl_(new pimpl_s), concept::Configurable(other) {
    for (auto *b : other.m_pimpl_->m_bundle_) { Register(b); }
    for (auto *b : m_pimpl_->m_bundle_) { other.Deregister(b); }
    m_pimpl_->m_mesh_ = other.m_pimpl_->m_mesh_;
}
Attribute::~Attribute() {
    for (auto *b : m_pimpl_->m_bundle_) { Deregister(b); }
}

void Attribute::Register(AttributeBundle *attr_b) {
    if (attr_b != nullptr) {
        auto res = m_pimpl_->m_bundle_.emplace(attr_b);
        if (res.second) { attr_b->Attach(this); }
    }
}
void Attribute::Deregister(AttributeBundle *attr_b) {
    if (m_pimpl_->m_bundle_.erase(attr_b) > 0) { attr_b->Detach(this); };
}

id_type Attribute::GetGUID() const { return db()->GetValue<id_type>("GUID", NULL_ID); };

void Attribute::SetMesh(Mesh const *m) { m_pimpl_->m_mesh_ = m; }
Mesh const *Attribute::GetMesh() const { return m_pimpl_->m_mesh_; }

void Attribute::SetRange(Range<mesh::MeshEntityId> const &r) { m_pimpl_->m_range_ = r; }
Range<mesh::MeshEntityId> const &Attribute::GetRange() const { return m_pimpl_->m_range_; }

bool Attribute::isNull() const { return false; }

}  //{ namespace engine
}  // namespace simpla
