//
// Created by salmon on 16-11-24.
//
#include "Mesh.h"
#include <simpla/model/Model.h>
#include "Attribute.h"
#include "MeshBlock.h"
#include "Patch.h"
namespace simpla {
namespace mesh {

Mesh::Mesh(Worker *w) : m_owner_(w) {}

Mesh::~Mesh() {}

std::ostream &Mesh::Print(std::ostream &os, int indent) const {
    os << std::setw(indent + 1) << " "
       << "Mesh = { ";
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
void Mesh::Deploy() { Object::Deploy(); };
void Mesh::Destroy() { Object::Destroy(); };

void Mesh::mesh_block(std::shared_ptr<MeshBlock> m) {
    Finalize();
    m_mesh_block_ = m;
    Initialize();
}

void Mesh::PreProcess() {
    Object::PreProcess();
    ASSERT(m_mesh_block_ != nullptr);
}

void Mesh::PostProcess() {
    m_mesh_block_.reset();
    Object::PostProcess();
}

void Mesh::Initialize() { Object::Initialize(); }

void Mesh::Finalize() { Object::Finalize(); }
}
}  // namespace simpla {namespace mesh
