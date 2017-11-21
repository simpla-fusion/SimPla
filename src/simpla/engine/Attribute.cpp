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
struct AttributeGroup::pimpl_s {
    std::set<Attribute *> m_attributes_;
    bool m_is_initiazlied_ = false;
};
AttributeGroup::AttributeGroup() : m_pimpl_(new pimpl_s){};

AttributeGroup::~AttributeGroup() {
    for (auto &item : m_pimpl_->m_attributes_) { item->Deregister(this); }
    delete m_pimpl_;
}
bool AttributeGroup::IsInitialized() const { return m_pimpl_->m_is_initiazlied_; }

std::shared_ptr<data::DataEntry> AttributeGroup::Serialize() const {
    auto res = data::DataLightT<std::string *>::New();
    for (auto const &item : m_pimpl_->m_attributes_) { res->push_back(item->GetName()); }
    return data::DataEntry::New(res);
}
void AttributeGroup::Deserialize(std::shared_ptr<const data::DataEntry> const &cfg) {
    if (cfg == nullptr) { return; }
    for (auto const &item : m_pimpl_->m_attributes_) { item->Deserialize(cfg->Get(item->GetName())); }
}

std::set<Attribute *> &AttributeGroup::GetAttributes() { return m_pimpl_->m_attributes_; }
std::set<Attribute *> const &AttributeGroup::GetAttributes() const { return m_pimpl_->m_attributes_; }
// void AttributeGroup::Push(const std::shared_ptr<data::DataEntry> &p) {
//    if (p != nullptr) {
//        for (auto &item : m_pimpl_->m_attributes_) {
//            if (auto patch = p->Get(item->GetName())) { item->Push(patch); }
//        }
//    }
//}
//
// std::shared_ptr<data::DataEntry> AttributeGroup::Pop() const {
//    auto res = data::DataEntry::New();
//    for (auto &item : m_pimpl_->m_attributes_) { res->Set(item->GetName(), item->Pop()); }
//    return res;
//}
void AttributeGroup::Push(const std::shared_ptr<Patch> &p) {
    if (p == nullptr) { return; }
    m_pimpl_->m_is_initiazlied_ = true;
    for (auto &item : m_pimpl_->m_attributes_) {
        if (auto blk = p->GetDataBlock(item->GetName())) {
            item->Push(blk);
        } else {
            m_pimpl_->m_is_initiazlied_ = false;
        }
    }
}

std::shared_ptr<Patch> AttributeGroup::Pop() const {
    auto res = Patch::New();
    for (auto &item : m_pimpl_->m_attributes_) {
        if (!item->isNull()) { res->SetDataBlock(item->GetName(), item->Pop()); }
    }
    m_pimpl_->m_is_initiazlied_ = false;
    return res;
}
void AttributeGroup::Attach(Attribute *p) {
    if (p != nullptr) { m_pimpl_->m_attributes_.insert(p); }
}
void AttributeGroup::Detach(Attribute *p) {
    if (p != nullptr) { m_pimpl_->m_attributes_.erase(p); }
}

struct Attribute::pimpl_s {
    std::set<AttributeGroup *> m_bundle_;
};

Attribute::Attribute() : m_pimpl_(new pimpl_s) {}
Attribute::~Attribute() {
    for (auto *grp : m_pimpl_->m_bundle_) { grp->Detach(this); }
    delete m_pimpl_;
}
std::shared_ptr<Attribute> Attribute::Copy() const {
    FIXME;
    return nullptr;
}
void Attribute::ReRegister(std::shared_ptr<Attribute> const &attr) const {
    for (auto &g : m_pimpl_->m_bundle_) { attr->Register(g); }
}
namespace detail {
template <typename U, int IFORM>
std::shared_ptr<Attribute> NewAttribute1(std::shared_ptr<simpla::data::DataEntry> const &cfg) {
    if (cfg == nullptr) { return nullptr; }
    std::shared_ptr<Attribute> res = nullptr;
    if (auto p = std::dynamic_pointer_cast<data::DataLightT<int>>(cfg->GetEntity())) {
        switch (p->value()) {
            case 1:
                res = AttributeT<U, IFORM, 1>::New(cfg);
                break;
            case 2:
                res = AttributeT<U, IFORM, 2>::New(cfg);
                break;
            case 3:
                res = AttributeT<U, IFORM, 3>::New(cfg);
                break;
            case 4:
                res = AttributeT<U, IFORM, 4>::New(cfg);
                break;
            case 5:
                res = AttributeT<U, IFORM, 5>::New(cfg);
                break;
            case 6:
                res = AttributeT<U, IFORM, 6>::New(cfg);
                break;
            case 7:
                res = AttributeT<U, IFORM, 7>::New(cfg);
                break;
            case 8:
                res = AttributeT<U, IFORM, 8>::New(cfg);
                break;
            case 9:
                res = AttributeT<U, IFORM, 9>::New(cfg);
                break;
            default:
                UNIMPLEMENTED;
                break;
        }
    } else {
        UNIMPLEMENTED;
    }
    return res;
}
template <typename U>
std::shared_ptr<Attribute> NewAttribute(std::shared_ptr<simpla::data::DataEntry> const &cfg) {
    if (cfg == nullptr) { return nullptr; }
    std::shared_ptr<Attribute> res = nullptr;

    switch (cfg->GetValue<int>("IFORM", NODE)) {
        case NODE:
            res = NewAttribute1<U, NODE>(cfg);
            break;
        case CELL:
            res = NewAttribute1<U, CELL>(cfg);
            break;
        case EDGE:
            res = NewAttribute1<U, EDGE>(cfg);
            break;
        case FACE:
            res = NewAttribute1<U, FACE>(cfg);
            break;
        case FIBER:
            res = NewAttribute1<U, FIBER>(cfg);
            break;
        default:
            UNIMPLEMENTED;
            break;
    }
    return res;
}
}
std::shared_ptr<Attribute> Attribute::New(std::shared_ptr<simpla::data::DataEntry> const &cfg) {
    if (cfg == nullptr) { return nullptr; }
    std::shared_ptr<Attribute> res = nullptr;
    auto v_type = cfg->GetValue<std::string>("ValueType");
    if (v_type == simpla::traits::type_name<double>::value()) {
        res = detail::NewAttribute<double>(cfg);
    } else if (v_type == simpla::traits::type_name<float>::value()) {
        res = detail::NewAttribute<float>(cfg);
    } else if (v_type == simpla::traits::type_name<int>::value()) {
        res = detail::NewAttribute<int>(cfg);
    } else if (v_type == simpla::traits::type_name<long>::value()) {
        res = detail::NewAttribute<long>(cfg);
    } else if (v_type == simpla::traits::type_name<unsigned int>::value()) {
        res = detail::NewAttribute<unsigned int>(cfg);
    } else if (v_type == simpla::traits::type_name<unsigned long>::value()) {
        res = detail::NewAttribute<unsigned long>(cfg);
    }

    return res;
}

void Attribute::Register(AttributeGroup *attr_b) {
    if (attr_b == nullptr) {
        static std::hash<std::string> s_hasher;
        auto id = s_hasher(GetProperty<std::string>("name", "unnamed") +  //
                           "." + value_type_info().name() +               //
                           "." + std::to_string(GetIFORM()) +             //
                           "." + std::to_string(GetRank()));
        SetProperty("DescId", id);
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
