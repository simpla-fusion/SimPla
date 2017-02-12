//
// Created by salmon on 17-2-12.
//
#include "DomainView.h"
#include "AttributeView.h"
#include <simpla/mesh/MeshView.h>
namespace simpla {
namespace engine {
void DomainView::Dispatch(Domain const &d) {
    m_mesh_->Update();
    for (auto &attr : m_attrs_) { attr->Update(); }
};

void DomainView::Connect(AttributeView *attr) {
    attr->Connect(this);
    m_attrs_.insert(attr);
};
void DomainView::Disconnect(AttributeView *attr) {
    attr->Disconnect(this);
    m_attrs_.erase(attr);
}

std::shared_ptr<DataBlock> DomainView::data_block(id_type) const {}
void DomainView::data_block(id_type, std::shared_ptr<DataBlock>) {}
}  // namespace engine
}  // namespace simpla