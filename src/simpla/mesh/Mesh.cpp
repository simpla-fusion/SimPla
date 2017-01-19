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

Mesh::Mesh() : m_model_(nullptr) {}

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
void Mesh::Deploy() {
    if (m_mesh_block_ != nullptr) { m_mesh_block_->Deploy(); }
    if (m_model_ == nullptr) { m_model_ = std::make_unique<simpla::model::Model>(this); }
};

// bool Mesh::is_a(std::type_info const &info) const { return typeid(Mesh) == info; }

void Mesh::Accept(Patch *p) {
    PostProcess();
    m_mesh_block_ = p->mesh();
    for (auto attr : m_attrs_) { attr->Accept(p->data(attr->description().id())); }
    PreProcess();
};

void Mesh::Initialize(Real data_time, Real dt) { PreProcess(); }

void Mesh::Finalize(Real data_time, Real dt) { PostProcess(); }

void Mesh::PreProcess() { ASSERT(m_mesh_block_ != nullptr); }

void Mesh::PostProcess() { m_mesh_block_.reset(); }
}
}  // namespace simpla {namespace mesh
