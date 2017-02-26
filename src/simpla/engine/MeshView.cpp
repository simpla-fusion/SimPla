//
// Created by salmon on 16-11-24.
//
#include "MeshView.h"
#include <simpla/model/Model.h>
#include "AttributeView.h"
#include "DomainView.h"
#include "MeshBlock.h"
namespace simpla {
namespace engine {

struct MeshView::pimpl_s {
    std::shared_ptr<MeshBlock> m_mesh_block_;
    DomainView *m_domain_ = nullptr;
};
MeshView::MeshView() : m_pimpl_(new pimpl_s) {}
MeshView::~MeshView() {}

std::ostream &MeshView::Print(std::ostream &os, int indent) const {
    os << std::setw(indent + 1) << "type = \"" << getClassName() << "\",";
    if (m_pimpl_->m_mesh_block_ != nullptr) {
        os << std::endl;
        os << std::setw(indent + 1) << " "
           << " Block = {";
        //        m_pimpl_->m_mesh_block_->Print(os, indent + 1);
        os << std::setw(indent + 1) << " "
           << "},";
    }

    return os;
};
void MeshView::Connect(DomainView *b) {
    if (m_pimpl_->m_domain_ != b) {
        Disconnect();
        m_pimpl_->m_domain_ = b;
        if (b != nullptr) { b->SetMesh(this); }
    }
}
void MeshView::Disconnect() {
    if (m_pimpl_->m_domain_ != nullptr && m_pimpl_->m_domain_->GetMesh() == this) {
        m_pimpl_->m_domain_->SetMesh(nullptr);
    }
    m_pimpl_->m_domain_ = nullptr;
}
void MeshView::OnNotify() { SetMeshBlock(GetDomain()->GetMeshBlock()); }
DomainView const *MeshView::GetDomain() const { return m_pimpl_->m_domain_; }
void MeshView::Update() {
    if (!isModified()) { return; }
    concept::StateCounter::Tag();
    Initialize();
}

id_type MeshView::GetMeshBlockId() const {
    return m_pimpl_->m_mesh_block_ == nullptr ? NULL_ID : m_pimpl_->m_mesh_block_->id();
}
std::shared_ptr<MeshBlock> const &MeshView::GetMeshBlock() const { return m_pimpl_->m_mesh_block_; }
void MeshView::SetMeshBlock(std::shared_ptr<MeshBlock> const &m) {
    if (m == m_pimpl_->m_mesh_block_) { return; }
    Finalize();
    m_pimpl_->m_mesh_block_ = m;
    Click();
}
void MeshView::Initialize() {}
void MeshView::Finalize() {}
}  // {namespace mesh
}  // namespace simpla
