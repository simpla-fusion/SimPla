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
        //        m_backend_->m_mesh_block_->Print(os, indent + 1);
        os << std::setw(indent + 1) << " "
           << "},";
    }

    return os;
};
void MeshView::Connect(DomainView *b) {
    //    if (m_backend_->m_domain_ != b) {
    //        Disconnect();
    //        m_backend_->m_domain_ = b;
    ////        if (b != nullptr) { b->SetMesh(this); }
    //    }
}
void MeshView::Disconnect() {
    //    if (m_backend_->m_domain_ != nullptr && m_backend_->m_domain_->GetMesh() == this) {
    //        m_backend_->m_domain_->SetMesh(nullptr);
    //    }
    //    m_backend_->m_domain_ = nullptr;
}
void MeshView::OnNotify() { /*SetMeshBlock(GetDomainWithMaterial()->GetMeshBlock());*/
}
DomainView const *MeshView::GetDomain() const { return m_pimpl_->m_domain_; }
bool MeshView::Update() { return SPObject::Update(); }

id_type MeshView::GetMeshBlockId() const {
    return m_pimpl_->m_mesh_block_ == nullptr ? NULL_ID : m_pimpl_->m_mesh_block_->GetGUID();
}
std::shared_ptr<MeshBlock> const &MeshView::GetMeshBlock() const { return m_pimpl_->m_mesh_block_; }
void MeshView::SetMeshBlock(std::shared_ptr<MeshBlock> const &m) {
    if (m == m_pimpl_->m_mesh_block_) {
        return;
    } else
        m_pimpl_->m_mesh_block_ = m;
    Click();
}
void MeshView::Initialize() { SPObject::Initialize(); }

Real MeshView::GetDt() const { return 1.0; }

}  // {namespace mesh
}  // namespace simpla