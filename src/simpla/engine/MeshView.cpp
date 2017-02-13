//
// Created by salmon on 16-11-24.
//
#include "MeshView.h"
#include <simpla/mesh/MeshBlock.h>
#include <simpla/model/Model.h>
namespace simpla {
namespace engine {
struct MeshView::pimpl_s {
    std::shared_ptr<mesh::MeshBlock> m_mesh_block_;
};
MeshView::MeshView(DomainView *w = nullptr) : m_pimpl_(new pimpl_s) {}

MeshView::~MeshView() {}

std::ostream &MeshView::Print(std::ostream &os, int indent) const {
    os << std::setw(indent + 1) << " "
       << "MeshView = { ";
    os << "Type = \"" << getClassName() << "\",";
    if (m_mesh_block_ != nullptr) {
        os << std::endl;
        os << std::setw(indent + 1) << " "
           << " Block = {";
        m_mesh_block_->Print(os, indent + 1);
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
bool MeshView::isUpdated() const { return false; }
void MeshView::Update() {}
void MeshView::Initialize(){};

void MeshView::mesh_block(std::shared_ptr<MeshBlock> m) {
    Finalize();
    m_mesh_block_ = m;
    Initialize();
}

void MeshView::PreProcess() {
    Object::PreProcess();
    ASSERT(m_mesh_block_ != nullptr);
}

void MeshView::PostProcess() {
    m_mesh_block_.reset();
    Object::PostProcess();
}
}  // {namespace mesh
}  // namespace simpla
