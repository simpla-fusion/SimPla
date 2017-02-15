//
// Created by salmon on 16-10-20.
//

#include "AttributeView.h"
#include <set>
#include <typeindex>
#include "DataBlock.h"
#include "DomainView.h"
#include "MeshView.h"

namespace simpla {
namespace engine {

struct AttributeViewBundle::pimpl_s {
    id_type m_current_block_id_ = NULL_ID;
    DomainView const *m_domain_ = nullptr;
    std::set<AttributeView *> m_attr_views_;
};

AttributeViewBundle::AttributeViewBundle(DomainView const *d) : m_pimpl_(new pimpl_s) { SetDomain(d); }
AttributeViewBundle::~AttributeViewBundle() {}
std::ostream &AttributeViewBundle::Print(std::ostream &os, int indent) const {
    for (AttributeView *attr : m_pimpl_->m_attr_views_) { os << "\"" << attr->name() << "\" ,"; }
    return os;
}

void AttributeViewBundle::insert(AttributeView *attr) { m_pimpl_->m_attr_views_.insert(attr); }
void AttributeViewBundle::erase(AttributeView *attr) { m_pimpl_->m_attr_views_.erase(attr); }
void AttributeViewBundle::insert(AttributeViewBundle *attr_bundle) {
    m_pimpl_->m_attr_views_.insert(attr_bundle->m_pimpl_->m_attr_views_.begin(),
                                   attr_bundle->m_pimpl_->m_attr_views_.end());
}
id_type AttributeViewBundle::current_block_id() const { return m_pimpl_->m_current_block_id_; }
bool AttributeViewBundle::isUpdated() const {
    return GetDomain() == nullptr || GetDomain()->current_block_id() == current_block_id();
}
void AttributeViewBundle::Update() const {
    if (isUpdated()) { return; }
    for (AttributeView *attr : m_pimpl_->m_attr_views_) {
        attr->SetDomain(GetDomain());
        attr->Update();
    }
    if (GetDomain() != nullptr) {
        m_pimpl_->m_current_block_id_ = GetDomain()->current_block_id();
    } else {
        m_pimpl_->m_current_block_id_ = NULL_ID;
    }
}
void AttributeViewBundle::SetDomain(DomainView const *d) { m_pimpl_->m_domain_ = d; };
DomainView const *AttributeViewBundle::GetDomain() const { return m_pimpl_->m_domain_; }

void AttributeViewBundle::for_each(std::function<void(AttributeView *)> const &fun) const {
    for (auto &attr : m_pimpl_->m_attr_views_) { fun(attr); }
}

struct AttributeView::pimpl_s {
   public:
    pimpl_s(std::string const &name_s, std::type_info const &t_id, int IFORM, int DOF);
    ~pimpl_s();
    AttributeViewBundle *m_bundle_ = nullptr;
    DomainView const *m_domain_ = nullptr;
    std::shared_ptr<DataBlock> m_data_;
    MeshView const *m_mesh_ = nullptr;
    id_type m_current_block_id_ = NULL_ID;
    const std::string m_name_;
    const std::type_index m_t_idx_;
    const int m_iform_;
    const int m_dof_;
    const id_type m_GUID_ = NULL_ID;
};

id_type GeneratorAttrGUID(std::string const &name_s, std::type_info const &t_id, int IFORM, int DOF) {
    std::string str = name_s + '.' + t_id.name() + '.' + static_cast<char>(IFORM + '0') + static_cast<char>(DOF + '0');
    return static_cast<id_type>(std::hash<std::string>{}(str));
}
AttributeView::pimpl_s::pimpl_s(std::string const &name_s, std::type_info const &t_id, int IFORM, int DOF)
    : m_name_(name_s == "" ? ("unnamed_" + std::to_string(reinterpret_cast<size_t>(this)))
                           : name_s),  // address pointer is used as UUID
      m_t_idx_(t_id),
      m_iform_(IFORM),
      m_dof_(DOF),
      m_GUID_(GeneratorAttrGUID(name_s, t_id, IFORM, DOF)) {}
AttributeView::pimpl_s::~pimpl_s() {}
AttributeView::AttributeView(std::string const &name_s, std::type_info const &t_id, int IFORM, int DOF)
    : m_pimpl_(new pimpl_s(name_s, t_id, IFORM, DOF)) {}

AttributeView::~AttributeView() { Disconnect(); }

void AttributeView::Connect(AttributeViewBundle *b) {
    b->insert(this);
    m_pimpl_->m_bundle_ = b;
}
void AttributeView::Disconnect() {
    if (m_pimpl_->m_bundle_ != nullptr) m_pimpl_->m_bundle_->erase(this);
    m_pimpl_->m_bundle_ = nullptr;
}
id_type AttributeView::GUID() const { return m_pimpl_->m_GUID_; };
std::string const &AttributeView::name() const { return m_pimpl_->m_name_; }
std::type_index AttributeView::value_type_index() const { return m_pimpl_->m_t_idx_; }
int AttributeView::iform() const { return m_pimpl_->m_iform_; }
int AttributeView::dof() const { return m_pimpl_->m_dof_; }

std::type_index AttributeView::mesh_type_index() const { return std::type_index(typeid(MeshView)); }

void AttributeView::SetDomain(DomainView const *d) { m_pimpl_->m_domain_ = d; };
DomainView const *AttributeView::GetDomain() const { return m_pimpl_->m_domain_; }

id_type AttributeView::current_block_id() const { return m_pimpl_->m_current_block_id_; }
bool AttributeView::isUpdated() const {
    return m_pimpl_->m_domain_ == nullptr || m_pimpl_->m_domain_->current_block_id() == current_block_id();
}
void AttributeView::Update() {
    if (isUpdated()) { return; }
    Initialize();
    m_pimpl_->m_current_block_id_ = m_pimpl_->m_domain_ == nullptr ? NULL_ID : m_pimpl_->m_domain_->current_block_id();
}

void AttributeView::Initialize() {
    if (m_pimpl_->m_domain_ != nullptr) {
        m_pimpl_->m_data_ = m_pimpl_->m_domain_->data_block(GUID());
        m_pimpl_->m_mesh_ = m_pimpl_->m_domain_->GetMesh().get();
    }
}

bool AttributeView::isNull() const { return m_pimpl_->m_data_ == nullptr; }
const std::shared_ptr<DataBlock> &AttributeView::data_block() const { return m_pimpl_->m_data_; }
std::shared_ptr<DataBlock> &AttributeView::data_block() { return m_pimpl_->m_data_; }

std::ostream &AttributeView::Print(std::ostream &os, int indent) const {
    os << std::setw(indent + 1) << " " << name();
    return os;
};

}  //{ namespace engine
}  // namespace simpla
