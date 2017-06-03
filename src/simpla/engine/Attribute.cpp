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
AttributeDesc::AttributeDesc(int IFORM, int DOF, std::type_info const &t_info, std::string const &s_prefix,
                             std::shared_ptr<data::DataTable> const &t_db)
    : data::Configurable(t_db), m_prefix_(s_prefix), m_iform_(IFORM), m_dof_(DOF), m_t_info_(t_info) {}

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
    return std::make_shared<AttributeDesc>(GetIFORM(), GetDOF(), value_type_info(), GetPrefix(), db());
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
    for (auto &item : m_pimpl_->m_attributes_) { item.second->Register(other); }
};
void AttributeGroup::DeregisterFrom(AttributeGroup *other) {
    for (auto &item : m_pimpl_->m_attributes_) { item.second->Deregister(other); }
};
void AttributeGroup::Push(Patch *p) {
    for (auto &item : GetAllAttributes()) {
        auto k = "." + std::string(EntityIFORMName[item.second->GetIFORM()]) + "_BODY";
        auto it = p->m_ranges.find(k);
        item.second->Push(p->Pop(item.second->GetID()), (it == p->m_ranges.end()) ? EntityRange{} : it->second);
    }
}
void AttributeGroup::Pop(Patch *p) {
    for (auto &item : GetAllAttributes()) { p->Push(item.second->GetID(), item.second->Pop()); }
}
void AttributeGroup::Attach(Attribute *p) { m_pimpl_->m_attributes_.emplace(p->GetPrefix(), p); }
void AttributeGroup::Detach(Attribute *p) { m_pimpl_->m_attributes_.erase(p->GetPrefix()); }
std::map<std::string, Attribute *> &AttributeGroup::GetAllAttributes() { return m_pimpl_->m_attributes_; };
std::map<std::string, Attribute *> const &AttributeGroup::GetAll() const { return m_pimpl_->m_attributes_; };
bool AttributeGroup::has(std::string const &k) const {
    return m_pimpl_->m_attributes_.find(k) != m_pimpl_->m_attributes_.end();
}
bool AttributeGroup::check(std::string const &k, std::type_info const &t_info) const {
    auto it = m_pimpl_->m_attributes_.find(k);
    return (it != m_pimpl_->m_attributes_.end() && it->second->isA(t_info));
}
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
    MeshBase *m_mesh_;
    std::set<AttributeGroup *> m_bundle_;
};
Attribute::Attribute(int IFORM, int DOF, std::type_info const &t_info, AttributeGroup *grp,
                     std::shared_ptr<DataTable> cfg)
    : SPObject((cfg != nullptr && cfg->has("name")) ? cfg->GetValue<std::string>("name") : ""),
      AttributeDesc(IFORM, DOF, t_info, SPObject::GetName(), cfg),
      m_pimpl_(new pimpl_s) {
    Register(grp);
};

Attribute::Attribute(Attribute const &other) : AttributeDesc(other), m_pimpl_(new pimpl_s) {}
Attribute::Attribute(Attribute &&other) : AttributeDesc(other), m_pimpl_(std::move(other.m_pimpl_)) {}
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

void Attribute::Push(const std::shared_ptr<DataBlock> &d, const EntityRange &r) {}
std::shared_ptr<data::DataBlock> Attribute::Pop() { return nullptr; }

const MeshBase *Attribute::GetMesh() const { return m_pimpl_->m_mesh_; }
bool Attribute::isNull() const { return true; }
void Attribute::Update() { SPObject::Update(); };

}  //{ namespace engine
}  // namespace simpla
