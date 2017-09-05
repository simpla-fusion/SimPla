//
// Created by salmon on 16-10-20.
//

#include "Attribute.h"
#include <set>
#include <typeindex>
#include "Domain.h"
#include "MeshBlock.h"
#include "Patch.h"
#include "simpla/data/DataBlock.h"
//#include "simpla/mesh/MeshBase.h"
namespace simpla {
namespace engine {

AttributeGroup::AttributeGroup(){};

AttributeGroup::~AttributeGroup() {
    for (auto &item : m_attributes_) { item.second->Deregister(this); }
}

int AttributeGroup::Push(const std::shared_ptr<data::DataNode> &p) {
    int count = 0;
    for (auto &item : m_attributes_) {
        if (auto blk = p->Get(item.second->db()->GetValue<id_type>("DescID", NULL_ID))) {
            count += item.second->Push(blk);
        }
    }
    return count;
}

std::shared_ptr<data::DataNode> AttributeGroup::Pop() {
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
struct Attribute::pimpl_s {
    std::set<AttributeGroup *> m_bundle_{};
    std::shared_ptr<data::DataBlock> m_data_block_ = nullptr;
};

Attribute::Attribute() : m_pimpl_(new pimpl_s) {}
Attribute::~Attribute() {
    for (auto *grp : m_pimpl_->m_bundle_) { grp->Detach(this); }
    delete m_pimpl_;
}

std::shared_ptr<data::DataNode> Attribute::Serialize() const { return base_type::Serialize(); }
void Attribute::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

void Attribute::Register(AttributeGroup *attr_b) {
    if (attr_b == nullptr) {
        static std::hash<std::string> s_hasher;
        auto id = s_hasher(db()->GetValue<std::string>("name", "unnamed") +  //
                           "." + value_type_info().name() +                  //
                           "." + std::to_string(GetIFORM()) +                //
                           "." + std::to_string(GetDOF()));
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
int Attribute::Push(const std::shared_ptr<data::DataNode> &d) {
    //    m_pimpl_->m_data_block_ = d;
    Update();
    return 0;
}
std::shared_ptr<data::DataNode> Attribute::Pop() {
    auto res = data::DataNode::New(data::DataNode::DN_ENTITY);
    res->SetEntity(m_pimpl_->m_data_block_);
    TearDown();
    return res;
}
std::shared_ptr<DataBlock> Attribute::GetDataBlock() { return m_pimpl_->m_data_block_; }
std::shared_ptr<const DataBlock> Attribute::GetDataBlock() const { return m_pimpl_->m_data_block_; }
void Attribute::Clear() { FIXME; }
bool Attribute::isNull() const { return m_pimpl_ == nullptr || m_pimpl_->m_data_block_ == nullptr; }

}  //{ namespace engine
}  // namespace simpla
