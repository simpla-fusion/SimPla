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
    std::map<std::string, std::function<std::shared_ptr<Mesh>(std::shared_ptr<data::DataTable> const &,
                                                              std::shared_ptr<geometry::GeoObject> const &)>>
        m_mesh_factory_;
};

MeshViewFactory::MeshViewFactory() : m_pimpl_(new pimpl_s){};
MeshViewFactory::~MeshViewFactory(){};

bool MeshViewFactory::RegisterCreator(
    std::string const &k,
    std::function<std::shared_ptr<Mesh>(std::shared_ptr<data::DataTable> const &,
                                        std::shared_ptr<geometry::GeoObject> const &)> const &fun) {
    auto res = m_pimpl_->m_mesh_factory_.emplace(k, fun).second;
    if (res) { LOGGER << "Mesh Creator [ " << k << " ] is registered!" << std::endl; }
    return res;
};

std::shared_ptr<Mesh> MeshViewFactory::Create(std::shared_ptr<data::DataTable> const &config,
                                              std::shared_ptr<geometry::GeoObject> const &g) {
    std::shared_ptr<Mesh> res = nullptr;
    try {
        std::string key = "";
        std::shared_ptr<data::DataTable> t = nullptr;
        if (config == nullptr) {
            WARNING << "Create Mesh failed!" << std::endl;
            return res;
        } else if (config->value_type_info() == typeid(std::string)) {
            key = data::data_cast<std::string>(*config);
        } else if (config->isTable()) {
            t = std::dynamic_pointer_cast<data::DataTable>(config);
            key = t->GetValue<std::string>("name");
        }
        ASSERT(key != "");
        res = m_pimpl_->m_mesh_factory_.at(key)(t, g);
        res->db()->SetValue("name", key);

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
Mesh::Mesh(std::shared_ptr<data::DataTable> const &t, const std::shared_ptr<geometry::GeoObject> &geo_obj)
    : concept::Configurable(t), m_pimpl_(new pimpl_s) {
    AttributeBundle::SetMesh(this);
    m_pimpl_->m_geo_obj_ = geo_obj;
    if (m_pimpl_->m_geo_obj_ == nullptr) {
        m_pimpl_->m_geo_obj_ = GLOBAL_GEO_OBJECT_FACTORY.Create(db()->Get("GeometryObject"));
    }
    db()->SetValue("GeometryObject", m_pimpl_->m_geo_obj_);
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

void Mesh::PushData(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<data::DataTable> const &p) {
    ASSERT(m != nullptr);
    m_pimpl_->m_mesh_block_ = m;
    AttributeBundle::SetMesh(this);
    AttributeBundle::PushData(m, p);
};
std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataTable>> Mesh::PopData() {
    return AttributeBundle::PopData();
}

id_type Mesh::GetMeshBlockId() const {
    return m_pimpl_->m_mesh_block_ == nullptr ? NULL_ID : m_pimpl_->m_mesh_block_->GetGUID();
}
std::shared_ptr<MeshBlock> const &Mesh::GetMeshBlock() const { return m_pimpl_->m_mesh_block_; }

void Mesh::Initialize() {}

// Real Mesh::GetDt() const { return 1.0; }

}  // {namespace mesh
}  // namespace simpla
