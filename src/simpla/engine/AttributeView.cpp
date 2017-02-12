//
// Created by salmon on 16-10-20.
//

#include "AttributeView.h"
#include <typeindex>
#include "AttributeDesc.h"
#include "DataBlock.h"
#include "DomainView.h"

namespace simpla {
namespace engine {

struct AttributeView::pimpl_s {
    DomainView *m_domain_ = nullptr;
    AttributeDesc const *m_desc_;
    std::shared_ptr<DataBlock> m_data_;
    mesh::MeshView const *m_mesh_;
};
AttributeView::AttributeView(DomainView *w, AttributeDesc const *desc) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_domain_ = w;
    m_pimpl_->m_desc_ = desc;
    m_pimpl_->m_data_ = nullptr;
    Connect(w);
};
AttributeView::~AttributeView() { Disconnect(); }
AttributeDesc const &AttributeView::description() const { return *m_pimpl_->m_desc_; }
void AttributeView::Connect(DomainView *d) {
    Disconnect();
    if (d != nullptr) {
        m_pimpl_->m_domain_ = d;
        m_pimpl_->m_domain_->Connect(this);
    }
};
void AttributeView::Disconnect(DomainView *d) {
    if (d != nullptr) {
        d->Disconnect(this);
    } else if (m_pimpl_->m_domain_ != nullptr) {
        m_pimpl_->m_domain_->Disconnect(this);
        m_pimpl_->m_domain_ = nullptr;
    }
}
void AttributeView::Update() {
    if (m_pimpl_->m_desc_ == nullptr) { m_pimpl_->m_desc_ = &description(); }
    ASSERT(m_pimpl_->m_desc_ != nullptr);
    Load();
}
std::shared_ptr<DataBlock> AttributeView::CreateDataBlock() const { return nullptr; }

void AttributeView::Load() {
    if (m_pimpl_->m_domain_ == nullptr) {
        m_pimpl_->m_data_ = CreateDataBlock();
    } else {
        ASSERT(m_pimpl_->m_desc_ != nullptr);
        m_pimpl_->m_data_ = m_pimpl_->m_domain_->data_block(m_pimpl_->m_desc_->id());
        m_pimpl_->m_mesh_ = m_pimpl_->m_domain_->mesh();
    }
}
void AttributeView::Unload(bool do_dump) {
    if (m_pimpl_->m_domain_ == nullptr) {
        m_pimpl_->m_data_.reset();
    } else if (do_dump) {
        ASSERT(m_pimpl_->m_desc_ != nullptr);
        m_pimpl_->m_domain_->data_block(m_pimpl_->m_desc_->id(), m_pimpl_->m_data_);
    }
}
bool AttributeView::isNull() const { return m_pimpl_->m_data_ == nullptr; }
mesh::MeshView const *AttributeView::mesh_view() const { return m_pimpl_->m_mesh_; }
DataBlock *AttributeView::data_block() { return m_pimpl_->m_data_.get(); }
DataBlock const *AttributeView::data_block() const { return m_pimpl_->m_data_.get(); }
void AttributeView::data_block(std::shared_ptr<DataBlock> const &d) { m_pimpl_->m_data_ = d; };

}  //{ namespace engine
}  // namespace simpla
