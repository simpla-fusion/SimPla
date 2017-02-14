//
// Created by salmon on 16-10-20.
//

#include "AttributeView.h"
#include <set>
#include <typeindex>
#include "AttributeDesc.h"
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
void AttributeViewBundle::insert(AttributeView *attr) { m_pimpl_->m_attr_views_.insert(attr); }
void AttributeViewBundle::erase(AttributeView *attr) { m_pimpl_->m_attr_views_.erase(attr); }
void AttributeViewBundle::insert(AttributeViewBundle *attr_bundle) {
    m_pimpl_->m_attr_views_.insert(attr_bundle->m_pimpl_->m_attr_views_.begin(),
                                   attr_bundle->m_pimpl_->m_attr_views_.end());
}

bool AttributeViewBundle::isUpdated() const {
    return m_pimpl_->m_domain_ != nullptr &&                                            //
           m_pimpl_->m_domain_->current_block_id() == m_pimpl_->m_current_block_id_ &&  //
           m_pimpl_->m_current_block_id_ != NULL_ID;
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
id_type AttributeViewBundle::current_block_id() const { return m_pimpl_->m_current_block_id_; }

void AttributeViewBundle::for_each(std::function<void(AttributeView *)> const &fun) const {
    for (auto &attr : m_pimpl_->m_attr_views_) { fun(attr); }
}

struct AttributeView::pimpl_s {
    AttributeViewBundle *m_bundle_;
    DomainView const *m_domain_ = nullptr;
    std::shared_ptr<AttributeDesc> m_desc_;
    std::shared_ptr<DataBlock> m_data_;
    MeshView const *m_mesh_;
    id_type m_current_block_id_ = NULL_ID;
};

AttributeView::AttributeView(std::shared_ptr<AttributeDesc> const &desc) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_desc_ = desc;
};
AttributeView::AttributeView(std::string const &name_s)
    : AttributeView(std::make_shared<AttributeDesc>(name_s, value_type_index(), iform(), dof())) {}
AttributeView::AttributeView(std::string const &name_s, std::initializer_list<data::KeyValue> const &param)
    : AttributeView(name_s) {
    db.insert(param);
};
AttributeView::AttributeView(AttributeViewBundle *b, std::string const &name_s,
                             std::initializer_list<data::KeyValue> const &param)
    : AttributeView(name_s, param) {
    b->insert(this);
    m_pimpl_->m_bundle_ = b;
};
AttributeView::~AttributeView() {
    if (m_pimpl_->m_domain_ != nullptr) { m_pimpl_->m_bundle_->erase(this); }
}
std::type_index AttributeView::value_type_index() const { return std::type_index(typeid(Real)); }
std::type_index AttributeView::mesh_type_index() const { return std::type_index(typeid(MeshView)); }
int AttributeView::iform() const { return VERTEX; }
int AttributeView::dof() const { return 1; }

AttributeDesc const &AttributeView::description() const { return *m_pimpl_->m_desc_; }
void AttributeView::SetDomain(DomainView const *d) { m_pimpl_->m_domain_ = d; };
DomainView const *AttributeView::GetDomain() const { return m_pimpl_->m_domain_; }

//    if (m_pimpl_->m_domain_ != nullptr) { m_pimpl_->m_domain_->RemoveAttribute(this); }

bool AttributeView::isUpdated() const {
    return m_pimpl_->m_domain_ == nullptr || m_pimpl_->m_domain_->current_block_id() == m_pimpl_->m_current_block_id_;
    //    return (m_pimpl_->m_domain_ != nullptr &&                                            //
    //            m_pimpl_->m_domain_->current_block_id() == m_pimpl_->m_current_block_id_ &&  //
    //            m_pimpl_->m_current_block_id_ != NULL_ID) ||
    //           (m_pimpl_->m_domain_ == nullptr && m_pimpl_->m_data_ != nullptr);
}
void AttributeView::Update() {
    if (isUpdated()) { return; }
    ASSERT(m_pimpl_->m_desc_ != nullptr);
    Initialize();
    m_pimpl_->m_current_block_id_ = m_pimpl_->m_domain_->current_block_id();
}
void AttributeView::Initialize() {
    if (m_pimpl_->m_domain_ != nullptr) {
        ASSERT(m_pimpl_->m_desc_ != nullptr);
        m_pimpl_->m_data_ = m_pimpl_->m_domain_->data_block(m_pimpl_->m_desc_->id());
        m_pimpl_->m_mesh_ = m_pimpl_->m_domain_->GetMesh().get();
    }
}

bool AttributeView::isNull() const { return m_pimpl_->m_data_ == nullptr; }
const std::shared_ptr<DataBlock> &AttributeView::data_block() const { return m_pimpl_->m_data_; }
std::shared_ptr<DataBlock> &AttributeView::data_block() { return m_pimpl_->m_data_; }

}  //{ namespace engine
}  // namespace simpla
