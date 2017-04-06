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

struct MeshViewFactory::pimpl_s {
    std::map<std::string, std::function<Mesh *(std::shared_ptr<data::DataTable> const &)>> m_mesh_factory_;
};

MeshViewFactory::MeshViewFactory() : m_pimpl_(new pimpl_s){};
MeshViewFactory::~MeshViewFactory(){};

bool MeshViewFactory::RegisterCreator(std::string const &k,
                                      std::function<Mesh *(std::shared_ptr<data::DataTable> const &)> const &fun) {
    auto res = m_pimpl_->m_mesh_factory_.emplace(k, fun).second;
    if (res) { LOGGER << "Mesh Creator [ " << k << " ] is registered!" << std::endl; }
    return res;
};

Mesh *MeshViewFactory::Create(std::shared_ptr<data::DataTable> const &config) {
    Mesh *res = nullptr;
    try {
        if (config != nullptr) {
            res = m_pimpl_->m_mesh_factory_.at(config->GetValue<std::string>("name", ""))(config);
        }

    } catch (std::out_of_range const &) {
        RUNTIME_ERROR << "Mesh creator ["
                      << "] is missing!" << std::endl;
        return nullptr;
    }
    if (res != nullptr) { LOGGER << "Mesh [" << res->name() << "] is created!" << std::endl; }
    return res;
}

struct Mesh::pimpl_s {
    std::shared_ptr<MeshBlock> m_mesh_block_;
    std::shared_ptr<geometry::GeoObject> m_geo_obj_;
};
Mesh::Mesh(std::shared_ptr<data::DataTable> const &t) : concept::Configurable(t), m_pimpl_(new pimpl_s) {
    AttributeBundle::SetMesh(this);
}
Mesh::~Mesh() {}

std::ostream &Mesh::Print(std::ostream &os, int indent) const {
    os << std::setw(indent + 1) << "value_type_info = \"" << GetClassName() << "\",";
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
void Mesh::SetGeoObject(std::shared_ptr<geometry::GeoObject> const &g) { m_pimpl_->m_geo_obj_ = g; }
std::shared_ptr<geometry::GeoObject> const &Mesh::GetGeoObject() const { return m_pimpl_->m_geo_obj_; }
void Mesh::PushData(std::shared_ptr<Patch> p) {
    AttributeBundle::PushData(p);
    m_pimpl_->m_mesh_block_ = p->PopMeshBlock();
};
std::shared_ptr<Patch> Mesh::PopData() {
    auto res = AttributeBundle::PopData();
    res->PushMeshBlock(m_pimpl_->m_mesh_block_);
    m_pimpl_->m_mesh_block_.reset();
    return res;
}

id_type Mesh::GetBlockId() const {
    return m_pimpl_->m_mesh_block_ == nullptr ? NULL_ID : m_pimpl_->m_mesh_block_->GetGUID();
}
std::shared_ptr<MeshBlock> const &Mesh::GetBlock() const { return m_pimpl_->m_mesh_block_; }

void Mesh::Initialize() {}
void Mesh::Finalize() {}

// Real Mesh::GetDt() const { return 1.0; }

}  // {namespace mesh
}  // namespace simpla
