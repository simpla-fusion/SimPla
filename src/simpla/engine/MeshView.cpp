//
// Created by salmon on 16-11-24.
//
#include "MeshView.h"
#include <simpla/model/Model.h>
#include "AttributeView.h"
#include "DomainView.h"
namespace simpla {
namespace engine {

struct MeshView::pimpl_s {
    std::shared_ptr<MeshBlock> m_mesh_block_;
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

    os << std::setw(indent + 1) << " attributes = { ";
    AttributeViewBundle::Print(os, indent);
    os << "}  ";
    return os;
};

void MeshView::Update() {
    if (AttributeViewBundle::isUpdated()) { return; }
    m_pimpl_->m_mesh_block_ = AttributeViewBundle::GetDomain()->mesh_block();
    AttributeViewBundle::Update();
    Initialize();
}
std::shared_ptr<MeshBlock> const &MeshView::mesh_block() const { return m_pimpl_->m_mesh_block_; }

}  // {namespace mesh
}  // namespace simpla
