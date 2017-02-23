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
};
MeshView::MeshView() : m_pimpl_(new pimpl_s) { AttributeViewBundle::SetMesh(this); }
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

    os << std::setw(indent + 1) << " attributes = { ";
    AttributeViewBundle::Print(os, indent);
    os << "}  ";
    return os;
};
bool MeshView::isUpdated() const {
    return (!concept::StateCounter::isModified()) &&
           ((GetDomain() == nullptr) || (GetDomain()->GetMeshBlockId() == GetMeshBlockId()));
}
void MeshView::Update() {
    if (isUpdated()) { return; }
    AttributeViewBundle::SetMesh(this);
    if (GetDomain() != nullptr) { SetMeshBlock(GetDomain()->GetMeshBlock()); }
    AttributeViewBundle::Update();
    concept::StateCounter::Recount();

    Initialize();
}

id_type MeshView::GetMeshBlockId() const {
    return m_pimpl_->m_mesh_block_ == nullptr ? NULL_ID : m_pimpl_->m_mesh_block_->id();
}
std::shared_ptr<MeshBlock> const &MeshView::GetMeshBlock() const { return m_pimpl_->m_mesh_block_; }
void MeshView::SetMeshBlock(std::shared_ptr<MeshBlock> const &m) {
    if (m == m_pimpl_->m_mesh_block_) { return; }
    Finalize();
    Click();
    m_pimpl_->m_mesh_block_ = m;
}
void MeshView::Initialize() {}
void MeshView::Finalize() {}
}  // {namespace mesh
}  // namespace simpla
