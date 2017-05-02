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
AttributeDesc::AttributeDesc(std::string const &s, int IFORM, int DOF, std::type_info const &t_info,
                             std::shared_ptr<data::DataTable> const &t_db)
    : data::Configurable(t_db), m_name_(s), m_iform_(IFORM), m_dof_(DOF), m_t_info_(t_info) {}

AttributeDesc::AttributeDesc(AttributeDesc const &other)
    : data::Configurable(other),
      m_name_(other.m_name_),
      m_iform_(other.m_iform_),
      m_dof_(other.m_dof_),
      m_t_info_(other.m_t_info_){};
AttributeDesc::AttributeDesc(AttributeDesc &&other)
    : data::Configurable(other),
      m_name_(other.m_name_),
      m_iform_(other.m_iform_),
      m_dof_(other.m_dof_),
      m_t_info_(other.m_t_info_){};
std::string AttributeDesc::GetName() const { return m_name_; };
int AttributeDesc::GetIFORM() const { return m_iform_; };
int AttributeDesc::GetDOF() const { return m_dof_; };
std::type_info const &AttributeDesc::value_type_info() const { return m_t_info_; };

id_type AttributeDesc::GetID() const {
    static std::hash<std::string> s_hasher;
    return s_hasher(GetName() +                         //
                    "." + value_type_info().name() +    //
                    "." + std::to_string(GetIFORM()) +  //
                    "." + std::to_string(GetDOF()));
}

std::shared_ptr<AttributeDesc> AttributeDesc::GetDescription() const {
    return std::make_shared<AttributeDesc>(GetName(), GetIFORM(), GetDOF(), value_type_info(), db());
};

struct AttributeGroup::pimpl_s {
    std::set<Attribute *> m_attributes_;
};

AttributeGroup::AttributeGroup() : m_pimpl_(new pimpl_s) {}
AttributeGroup::~AttributeGroup() {}
void AttributeGroup::RegisterDescription(std::map<std::string, std::shared_ptr<AttributeDesc>> *m) {
    for (Attribute *attr : m_pimpl_->m_attributes_) { (*m)[attr->GetName()] = attr->GetDescription(); }
};

void AttributeGroup::RegisterAt(AttributeGroup *other) {
    for (Attribute *attr : m_pimpl_->m_attributes_) { attr->RegisterAt(other); }
};
void AttributeGroup::DeregisterFrom(AttributeGroup *other) {
    for (Attribute *attr : m_pimpl_->m_attributes_) { attr->DeregisterFrom(other); }
};
void AttributeGroup::Push(Patch *p) {
    for (auto *item : m_pimpl_->m_attributes_) { item->Push(p->Pop(item->GetID())); }
}
void AttributeGroup::Pop(Patch *p) {
    for (auto *item : m_pimpl_->m_attributes_) { p->Push(item->GetID(), item->Pop()); }
}
void AttributeGroup::Attach(Attribute *p) { m_pimpl_->m_attributes_.emplace(p); }
void AttributeGroup::Detach(Attribute *p) { m_pimpl_->m_attributes_.erase(p); }

std::set<Attribute *> const &AttributeGroup::GetAll() const { return m_pimpl_->m_attributes_; }

struct Attribute::pimpl_s {
    std::set<AttributeGroup *> m_bundle_;
};

Attribute::Attribute(std::shared_ptr<data::DataTable> const &cfg, int IFORM, int DOF, std::type_info const &t_info)
    : AttributeDesc(((cfg != nullptr && cfg->has("name")) ? cfg->GetValue<std::string>("name")
                                                          : ("_" + std::to_string(SPObject::GetGUID()))),
                    IFORM, DOF, t_info, cfg),
      m_pimpl_(new pimpl_s){};
Attribute::Attribute(Attribute const &other) : AttributeDesc(other), m_pimpl_(new pimpl_s) {}
Attribute::Attribute(Attribute &&other) : AttributeDesc(other), m_pimpl_(std::move(other.m_pimpl_)) {}
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

bool Attribute::isNull() const { return true; }
void Attribute::SetUp() { SPObject::SetUp(); };

}  //{ namespace engine
}  // namespace simpla
