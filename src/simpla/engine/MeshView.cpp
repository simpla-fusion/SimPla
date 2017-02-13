//
// Created by salmon on 16-11-24.
//
#include "MeshView.h"
#include <simpla/model/Model.h>
#include "DomainView.h"
namespace simpla {
namespace engine {

struct MeshView::pimpl_s {
    std::shared_ptr<MeshBlock> m_mesh_block_;
    DomainView *m_domain_ = nullptr;
    id_type m_current_block_id_ = NULL_ID;
};
MeshView::MeshView(DomainView *w) : m_pimpl_(new pimpl_s) {}

MeshView::~MeshView() {}

std::ostream &MeshView::Print(std::ostream &os, int indent) const {
    os << std::setw(indent + 1) << " "
       << "MeshView = { ";
    os << "Type = \"" << getClassName() << "\",";
    if (m_pimpl_->m_mesh_block_ != nullptr) {
        os << std::endl;
        os << std::setw(indent + 1) << " "
           << " Block = {";
//        m_pimpl_->m_mesh_block_->Print(os, indent + 1);
        os << std::setw(indent + 1) << " "
           << "},";
    }
    os << std::setw(indent + 1) << " "
       << "}," << std::endl;

    os << std::setw(indent + 1) << " "
       << "AttributeDesc Description= { ";

    os << std::setw(indent + 1) << " "
       << "} , " << std::endl;

    return os;
};
void MeshView::SetDomain(DomainView *d) { m_pimpl_->m_domain_ = d; };
DomainView const *MeshView::GetDomain() const { return m_pimpl_->m_domain_; }
void MeshView::UnsetDomain() { m_pimpl_->m_domain_ = nullptr; }
void MeshView::Dispatch(std::shared_ptr<MeshBlock> const &m) { m_pimpl_->m_mesh_block_ = m; }
std::shared_ptr<MeshBlock> const &MeshView::mesh_block() const { return m_pimpl_->m_mesh_block_; }
id_type MeshView::current_block_id() const { return m_pimpl_->m_current_block_id_; }
bool MeshView::isUpdated() const {
    return m_pimpl_->m_domain_ == nullptr || m_pimpl_->m_domain_->current_block_id() == current_block_id();
}
void MeshView::Update() {
    if (isUpdated()) { return; }
    m_pimpl_->m_current_block_id_ = m_pimpl_->m_mesh_block_->id();
    Initialize();
}
void MeshView::Initialize(){};

}  // {namespace mesh
}  // namespace simpla
