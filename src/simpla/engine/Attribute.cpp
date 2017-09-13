//
// Created by salmon on 16-10-20.
//

#include "Attribute.h"
#include <set>
#include <typeindex>
#include "Domain.h"
#include "MeshBlock.h"
#include "simpla/data/DataBlock.h"
namespace simpla {
namespace engine {

AttributeGroup::AttributeGroup() = default;

AttributeGroup::~AttributeGroup() {
    for (auto &item : m_attributes_) { item.second->Deregister(this); }
}
std::shared_ptr<data::DataNode> AttributeGroup::Serialize() const {
    auto res = data::DataNode::New(data::DataNode::DN_TABLE);
    for (auto const &item : m_attributes_) { res->Set(item.first, item.second->Serialize()); }
    return res;
}
void AttributeGroup::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    if (cfg == nullptr) { return; }
    for (auto const &item : m_attributes_) { item.second->Deserialize(cfg->Get(item.first)); }
}
void AttributeGroup::Push(const std::shared_ptr<data::DataNode> &p) {
    if (p != nullptr) {
        for (auto &item : m_attributes_) {
            if (auto attr = p->Get(item.second->db()->GetValue<id_type>("DescID", NULL_ID))) {
                item.second->Push(attr);
            }
        }
    }
}

std::shared_ptr<data::DataNode> AttributeGroup::Pop() const {
    auto res = data::DataNode::New();
    for (auto &item : m_attributes_) {
        res->Set(item.second->db()->GetValue<id_type>("DescID", NULL_ID), item.second->Pop());
    }
    return res;
}

void AttributeGroup::Attach(Attribute *p) { m_attributes_.emplace(p->GetName(), p); }
void AttributeGroup::Detach(Attribute *p) { m_attributes_.erase(p->GetName()); }
std::shared_ptr<data::DataNode> AttributeGroup::RegisterAttributes() {
    auto res = data::DataNode::New();
    for (auto &item : m_attributes_) { res->Set(item.first, item.second->db()); }
    return res;
}
std::shared_ptr<data::DataNode> AttributeGroup::GetAttributeDescription(std::string const &k) const {
    auto it = m_attributes_.find(k);
    return it != m_attributes_.end() ? it->second->db() : nullptr;
}

struct Attribute::pimpl_s {
    std::set<AttributeGroup *> m_bundle_;
};

Attribute::Attribute() : m_pimpl_(new pimpl_s) {}
Attribute::~Attribute() {
    for (auto *grp : m_pimpl_->m_bundle_) { grp->Detach(this); }
    delete m_pimpl_;
}
std::shared_ptr<Attribute> Attribute::Duplicate() const {
    FIXME;
    return nullptr;
}
void Attribute::ReRegister(std::shared_ptr<Attribute> const &attr) const {
    for (auto &g : m_pimpl_->m_bundle_) { attr->Register(g); }
}

std::shared_ptr<data::DataNode> Attribute::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("_DATA_", const_cast<this_type *>(this)->Pop());
    return res;
}
void Attribute::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    Push(cfg->Get("_DATA_"));
}
void Attribute::DoSetUp() { base_type::DoSetUp(); };
void Attribute::DoUpdate() { base_type::DoUpdate(); };
void Attribute::DoTearDown() { base_type::DoTearDown(); };
void Attribute::Register(AttributeGroup *attr_b) {
    if (attr_b == nullptr) {
        static std::hash<std::string> s_hasher;
        auto id = s_hasher(db()->GetValue<std::string>("name", "unnamed") +  //
                           "." + value_type_info().name() +                  //
                           "." + std::to_string(GetIFORM()) +                //
                           "." + std::to_string(GetRank()));
        db()->SetValue("DescId", id);
        for (auto *item : m_pimpl_->m_bundle_) { Register(item); }
    } else {
        auto res = m_pimpl_->m_bundle_.emplace(attr_b);
        if (res.second) { attr_b->Attach(this); }
    }
}
void Attribute::Deregister(AttributeGroup *attr_b) {
    if (attr_b != nullptr) {
        attr_b->Detach(this);
        m_pimpl_->m_bundle_.erase(attr_b);
    }
}

}  //{ namespace engine
}  // namespace simpla

// void AttributeGroup::RegisterDescription(std::map<std::string, std::shared_ptr<AttributeDesc>> *m) const {
//    for (auto &item : m_pimpl_->m_attributes_) { (*m)[item.first] = item.m_node_->GetDescription(); }
//};
//
// void AttributeGroup::RegisterAt(AttributeGroup *other) {
//    for (auto *item : m_pimpl_->m_attributes_) { item->Register(other); }
//};
// void AttributeGroup::DeregisterFrom(AttributeGroup *other) {
//    for (auto *item : m_pimpl_->m_attributes_) { item->Deregister(other); }
//};
//
// std::map<std::string, Attribute *> &AttributeGroup::GetAllAttributes() { return m_pimpl_->m_attributes_; };
// std::map<std::string, Attribute *> const &AttributeGroup::GetAll() const { return m_pimpl_->m_attributes_; };
// bool AttributeGroup::has(std::string const &k) const {
//    return m_pimpl_->m_attributes_.find(k) != m_pimpl_->m_attributes_.end();
//}
// bool AttributeGroup::check(std::string const &k, std::type_info const &t_info) const {
//    auto it = m_pimpl_->m_attributes_.find(k);
//    return (it != m_pimpl_->m_attributes_.end() && it->m_node_->isA(t_info));
//}
// Attribute *AttributeGroup::GetPatch(std::string const &k) {
//    auto it = m_pimpl_->m_attributes_.find(k);
//    Attribute *res = nullptr;
//    if (it != m_pimpl_->m_attributes_.end()) {
//        res = it->m_node_;
//    } else {
//        VERBOSE << "Can not find field [" << k << "] in [";
//        for (auto const &item : m_pimpl_->m_attributes_) { VERBOSE << item.first << ","; }
//        VERBOSE << std::endl;
//    }
//
//    return res;
//}
// Attribute const *AttributeGroup::GetPatch(std::string const &k) const {
//    auto it = m_pimpl_->m_attributes_.find(k);
//    Attribute *res = nullptr;
//    if (it != m_pimpl_->m_attributes_.end()) { res = it->m_node_; }
//    if (res == nullptr) {
//        VERBOSE << "Can not find field [" << k << "] in [";
//        for (auto const &item : m_pimpl_->m_attributes_) { VERBOSE << item.first << ","; }
//        VERBOSE << std::endl;
//    }
//
//    return res;
//}
//
// Attribute::Attribute(Attribute const &other) : SPObject(other), AttributeDesc(other) {
//    for (auto *grp : other.m_bundle_) { Register(grp); }
//    Initialize();
//}
// Attribute::Attribute(Attribute &&other) noexcept : SPObject(std::move(other)), AttributeDesc(std::move(other)) {
//    for (auto *grp : m_bundle_) {
//        grp->Detach(&other);
//        grp->Attach(this);
//    }
//}