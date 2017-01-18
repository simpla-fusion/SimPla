//
// Created by salmon on 16-11-24.
//
#include "Mesh.h"
#include <simpla/mesh/Attribute.h>
#include <simpla/mesh/MeshBlock.h>
#include "Patch.h"

namespace simpla {
namespace mesh {

Mesh::Mesh() {}

Mesh::~Mesh() {}

std::ostream &Mesh::print(std::ostream &os, int indent) const {
    os << std::setw(indent + 1) << " "
       << "Mesh = { ";
    os << "Type = \"" << get_class_name() << "\",";
    if (m_mesh_block_ != nullptr) {
        os << std::endl;
        os << std::setw(indent + 1) << " "
           << " Block = {";
        m_mesh_block_->print(os, indent + 1);
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
void Mesh::deploy() {
    if (m_mesh_block_ != nullptr) { m_mesh_block_->deploy(); }
};

// bool Mesh::is_a(std::type_info const &info) const { return typeid(Mesh) == info; }

void Mesh::accept(Patch *p) {
    post_process();
    m_mesh_block_ = p->mesh();
    for (auto attr : m_attrs_) { attr->accept(p->data(attr->description().id())); }
    pre_process();
};

void Mesh::initialize(Real data_time, Real dt) { pre_process(); }

void Mesh::finalize(Real data_time, Real dt) { post_process(); }

void Mesh::pre_process() { ASSERT(m_mesh_block_ != nullptr); }

void Mesh::post_process() { m_mesh_block_.reset(); }
}
}  // namespace simpla {namespace mesh
