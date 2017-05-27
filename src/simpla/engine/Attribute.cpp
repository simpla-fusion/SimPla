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
AttributeDesc::AttributeDesc(std::string const &prefix, int IFORM, int DOF, std::type_info const &t_info,
                             std::shared_ptr<data::DataTable> const &t_db)
    : data::Configurable(t_db), m_prefix_(prefix), m_iform_(IFORM), m_dof_(DOF), m_t_info_(t_info) {}

AttributeDesc::AttributeDesc(AttributeDesc const &other)
    : data::Configurable(other),
      m_prefix_(other.m_prefix_),
      m_iform_(other.m_iform_),
      m_dof_(other.m_dof_),
      m_t_info_(other.m_t_info_){};
AttributeDesc::AttributeDesc(AttributeDesc &&other)
    : data::Configurable(other),
      m_prefix_(other.m_prefix_),
      m_iform_(other.m_iform_),
      m_dof_(other.m_dof_),
      m_t_info_(other.m_t_info_){};
std::string AttributeDesc::GetPrefix() const { return m_prefix_; };
int AttributeDesc::GetIFORM() const { return m_iform_; };
int AttributeDesc::GetDOF() const { return m_dof_; };
std::type_info const &AttributeDesc::value_type_info() const { return m_t_info_; };

id_type AttributeDesc::GetID() const {
    static std::hash<std::string> s_hasher;
    return s_hasher(GetPrefix() +                       //
                    "." + value_type_info().name() +    //
                    "." + std::to_string(GetIFORM()) +  //
                    "." + std::to_string(GetDOF()));
}

std::shared_ptr<AttributeDesc> AttributeDesc::GetDescription() const {
    return std::make_shared<AttributeDesc>(GetPrefix(), GetIFORM(), GetDOF(), value_type_info(), db());
};

struct AttributeGroup::pimpl_s {
    std::map<std::string, Attribute *> m_attributes_;
};

AttributeGroup::AttributeGroup() : m_pimpl_(new pimpl_s) {}
AttributeGroup::~AttributeGroup() {}
void AttributeGroup::RegisterDescription(std::map<std::string, std::shared_ptr<AttributeDesc>> *m) {
    for (auto &item : m_pimpl_->m_attributes_) { (*m)[item.first] = item.second->GetDescription(); }
};

void AttributeGroup::RegisterAt(AttributeGroup *other) {
    for (auto &item : m_pimpl_->m_attributes_) { item.second->RegisterAt(other); }
};
void AttributeGroup::DeregisterFrom(AttributeGroup *other) {
    for (auto &item : m_pimpl_->m_attributes_) { item.second->DeregisterFrom(other); }
};

void AttributeGroup::Attach(Attribute *p) { m_pimpl_->m_attributes_.emplace(p->GetPrefix(), p); }
void AttributeGroup::Detach(Attribute *p) { m_pimpl_->m_attributes_.erase(p->GetPrefix()); }
std::map<std::string, Attribute *> &AttributeGroup::GetAllAttributes() { return m_pimpl_->m_attributes_; };
std::map<std::string, Attribute *> const &AttributeGroup::GetAll() const { return m_pimpl_->m_attributes_; };
Attribute *AttributeGroup::Get(std::string const &k) {
    auto it = m_pimpl_->m_attributes_.find(k);
    Attribute *res = nullptr;
    if (it != m_pimpl_->m_attributes_.end()) {
        res = it->second;
    } else {
        VERBOSE << "Can not find field [" << k << "] in [";
        for (auto const &item : m_pimpl_->m_attributes_) { VERBOSE << item.first << ","; }
        VERBOSE << std::endl;
    }

    return res;
}
Attribute const *AttributeGroup::Get(std::string const &k) const {
    auto it = m_pimpl_->m_attributes_.find(k);
    Attribute *res = nullptr;
    if (it != m_pimpl_->m_attributes_.end()) { res = it->second; }
    if (res == nullptr) {
        VERBOSE << "Can not find field [" << k << "] in [";
        for (auto const &item : m_pimpl_->m_attributes_) { VERBOSE << item.first << ","; }
        VERBOSE << std::endl;
    }

    return res;
}

struct Attribute::pimpl_s {
    Domain *m_domain_;
    std::set<AttributeGroup *> m_bundle_;
};
Attribute::Attribute(int IFORM, int DOF, std::type_info const &t_info, Domain *d,
                     std::shared_ptr<data::DataTable> const &cfg)
    : SPObject((cfg != nullptr && cfg->has("name")) ? cfg->GetValue<std::string>("name") : ""),
      AttributeDesc(SPObject::GetName(), IFORM, DOF, t_info, cfg),
      m_pimpl_(new pimpl_s) {
    RegisterAt(d);
    m_pimpl_->m_domain_ = d;
};

Attribute::Attribute(Attribute const &other) : AttributeDesc(other), m_pimpl_(new pimpl_s), SPObject(other.GetName()) {}
Attribute::Attribute(Attribute &&other)
    : AttributeDesc(other), m_pimpl_(std::move(other.m_pimpl_)), SPObject(other.GetName()) {}
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

void Attribute::Push(const std::shared_ptr<DataBlock> &d, const EntityRange &r) {}
std::shared_ptr<data::DataBlock> Attribute::Pop() { return nullptr; }

Domain *Attribute::GetDomain() const { return m_pimpl_->m_domain_; }
bool Attribute::isNull() const { return true; }
void Attribute::SetUp() { SPObject::SetUp(); };

}  //{ namespace engine
}  // namespace simpla
