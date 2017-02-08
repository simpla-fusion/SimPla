//
// Created by salmon on 16-11-24.
//
#include "MeshView.h"
#include <simpla/model/Model.h>
#include "AttributeView.h"
#include "MeshBlock.h"
#include "Patch.h"
namespace simpla {
namespace mesh {

MeshView::MeshView(Worker *w) : m_owner_(w) {}

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
       << "Attribute Description= { ";

    os << std::setw(indent + 1) << " "
       << "} , " << std::endl;

    return os;
};
void MeshView::Accept(std::shared_ptr<Patch> const &) {}
void MeshView::Initialize() { Object::Initialize(); };
void MeshView::Finalize() { Object::Finalize(); };

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
