//
// Created by salmon on 16-10-20.
//

#include "AttributeView.h"
#include <typeindex>
#include "AttributeDesc.h"
#include "DataBlock.h"
#include "DomainView.h"
#include "MeshView.h"

namespace simpla {
namespace engine {

struct AttributeView::pimpl_s {
    DomainView *m_domain_ = nullptr;
    AttributeDesc const *m_desc_;
    std::shared_ptr<DataBlock> m_data_;
    mesh::MeshView const *m_mesh_;
    id_type m_current_block_id_ = NULL_ID;
};
AttributeView::AttributeView(DomainView *w, AttributeDesc const *desc) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_domain_ = w;
    m_pimpl_->m_desc_ = desc;
    m_pimpl_->m_data_ = nullptr;
    SetDomain(w);
};
AttributeView::~AttributeView() {
    if (m_pimpl_->m_domain_ != nullptr) { m_pimpl_->m_domain_->RemoveAttribute(this); }
}
AttributeDesc const &AttributeView::description() const { return *m_pimpl_->m_desc_; }

void AttributeView::SetDomain(DomainView *d) { m_pimpl_->m_domain_ = d; };
DomainView const *AttributeView::GetDomain() const { return m_pimpl_->m_domain_; }
void AttributeView::UnsetDomain() { m_pimpl_->m_domain_ = nullptr; }

bool AttributeView::isUpdated() const {
    return (m_pimpl_->m_domain_ != nullptr &&                                            //
            m_pimpl_->m_domain_->current_block_id() == m_pimpl_->m_current_block_id_ &&  //
            m_pimpl_->m_current_block_id_ != NULL_ID) ||
           (m_pimpl_->m_domain_ == nullptr && m_pimpl_->m_data_ != nullptr);
}
void AttributeView::Update() {
    if (isUpdated()) { return; }
    if (m_pimpl_->m_desc_ == nullptr) { m_pimpl_->m_desc_ = &description(); }
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
mesh::MeshView const *AttributeView::mesh_view() const { return m_pimpl_->m_mesh_; }
const std::shared_ptr<DataBlock> &AttributeView::data_block() const { return m_pimpl_->m_data_; }
std::shared_ptr<DataBlock> &AttributeView::data_block() { return m_pimpl_->m_data_; }

}  //{ namespace engine
}  // namespace simpla
