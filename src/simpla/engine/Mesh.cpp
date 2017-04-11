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
Mesh::Mesh() : concept::Configurable(), m_pimpl_(new pimpl_s) {}
Mesh::Mesh(Mesh const &other) : m_pimpl_(new pimpl_s), concept::Configurable(other.db()) {
    m_pimpl_->m_mesh_block_ = other.m_pimpl_->m_mesh_block_;
    m_pimpl_->m_geo_obj_ = other.m_pimpl_->m_geo_obj_;
};

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

void Mesh::Register(AttributeGroup *) {}
void Mesh::Deregister(AttributeGroup *) {}

void Mesh::SetBlock(const std::shared_ptr<MeshBlock> &m) { m_pimpl_->m_mesh_block_ = m; }
std::shared_ptr<MeshBlock> const &Mesh::GetBlock() const { return m_pimpl_->m_mesh_block_; }
id_type Mesh::GetBlockId() const {
    return m_pimpl_->m_mesh_block_ == nullptr ? NULL_ID : m_pimpl_->m_mesh_block_->GetGUID();
}
//
// Range<mesh::MeshEntityId> Mesh::GetRange(std::shared_ptr<geometry::GeoObject> const &, int iform) const {
//    return Range<mesh::MeshEntityId>();
//};

void Mesh::SetGeoObject(std::shared_ptr<geometry::GeoObject> const &g) { m_pimpl_->m_geo_obj_ = g; }
std::shared_ptr<geometry::GeoObject> const &Mesh::GetGeoObject() const { return m_pimpl_->m_geo_obj_; }
void Mesh::SetChart(std::shared_ptr<Chart> const &c) { m_pimpl_->m_chart_ = c; }
std::shared_ptr<Chart> const &Mesh::GetChart() const { return m_pimpl_->m_chart_; }

// Range<mesh::EntityId> Mesh::range(int IFORM) const { return m_pimpl_->m_mesh_block_->range(IFORM); }

void Mesh::Initialize() {}
void Mesh::Finalize() {}

struct MeshViewFactory {
    std::map<std::string, std::function<Mesh *()>> m_mesh_factory_;
};

bool Mesh::RegisterCreator(std::string const &k, std::function<Mesh *()> const &fun) {
    auto res = SingletonHolder<MeshViewFactory>::instance().m_mesh_factory_.emplace(k, fun).second;
    if (res) { LOGGER << "Mesh Creator [ " << k << " ] is registered!" << std::endl; }
    return res;
}
Mesh *Mesh::Create(std::shared_ptr<data::DataTable> const &config) {
    Mesh *res = nullptr;
    try {
        if (config != nullptr) {
            res = SingletonHolder<MeshViewFactory>::instance().m_mesh_factory_.at(
                config->GetValue<std::string>("name", ""))();
            res->db() = config;
        }

    } catch (std::out_of_range const &) {
        RUNTIME_ERROR << "Mesh creator  [] is missing!" << std::endl;
        return nullptr;
    }
    if (res != nullptr) { LOGGER << "Mesh [" << res->name() << "] is created!" << std::endl; }
    return res;
}
// Real Mesh::GetDt() const { return 1.0; }

void Mesh::Push(const std::shared_ptr<Patch> &p) {
    m_pimpl_->m_mesh_block_ = p->GetBlock();
    Initialize();
}
std::shared_ptr<Patch> Mesh::Pop() const { return nullptr; }
}  // {namespace mesh
}  // namespace simpla
