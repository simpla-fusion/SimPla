//
// Created by salmon on 16-11-24.
//
#include "Mesh.h"
#include <simpla/geometry/GeoObject.h>
#include "Attribute.h"
#include "Domain.h"
#include "MeshBlock.h"
#include "Model.h"
#include "Patch.h"
namespace simpla {
namespace engine {

struct Mesh::pimpl_s {
    std::shared_ptr<MeshBlock> m_mesh_block_;
    std::shared_ptr<geometry::GeoObject> m_geo_obj_;
    std::shared_ptr<Chart> m_chart_;
};
Mesh::Mesh() : m_pimpl_(new pimpl_s) {}

Mesh::~Mesh() {}

// std::ostream &Mesh::Print(std::ostream &os, int indent) const {
//    os << std::setw(indent + 1) << "value_type_info = \"" << GetClassName() << "\",";
//    if (m_pimpl_->m_mesh_block_ != nullptr) {
//        os << std::endl;
//        os << std::setw(indent + 1) << " "
//           << " Block = {";
//        //        m_backend_->m_mesh_block_->Print(os, indent + 1);
//        os << std::setw(indent + 1) << " "
//           << "},";
//    }
//    return os;
//};

void Mesh::SetBlock(std::shared_ptr<MeshBlock> m) { m_pimpl_->m_mesh_block_ = m; }
std::shared_ptr<MeshBlock> Mesh::GetBlock() const { return m_pimpl_->m_mesh_block_; }
id_type Mesh::GetBlockId() const {
    return m_pimpl_->m_mesh_block_ == nullptr ? NULL_ID : m_pimpl_->m_mesh_block_->GetGUID();
}

void Mesh::SetGeoObject(std::shared_ptr<geometry::GeoObject> g) { m_pimpl_->m_geo_obj_ = g; }
std::shared_ptr<geometry::GeoObject> Mesh::GetGeoObject() const { return m_pimpl_->m_geo_obj_; }

void Mesh::SetChart(std::shared_ptr<Chart> c) { m_pimpl_->m_chart_ = c; }
std::shared_ptr<Chart> Mesh::GetChart() const { return m_pimpl_->m_chart_; }

void Mesh::Update() {}
void Mesh::Initialize() {}
void Mesh::Finalize() {}

std::shared_ptr<data::DataTable> Mesh::Serialize() const {
    auto p = std::make_shared<data::DataTable>();
    p->SetValue("Type", GetClassName());
    return p;
}
void Mesh::Deserialize(std::shared_ptr<data::DataTable>) {}
Range<mesh::MeshEntityId> Mesh::GetRange(int iform) const { return Range<mesh::MeshEntityId>(); };
void Mesh::Push(Patch *p) {
    m_pimpl_->m_mesh_block_ = p->GetBlock();
    AttributeGroup::Push(p);
}
void Mesh::Pop(Patch *p) {
    p->SetBlock(GetBlock());
    AttributeGroup::Pop(p);
}
}  // {namespace mesh
}  // namespace simpla
