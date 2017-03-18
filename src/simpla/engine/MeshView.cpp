//
// Created by salmon on 16-11-24.
//
#include "MeshView.h"
#include <simpla/geometry/GeoObject.h>
#include "AttributeView.h"
#include "DomainView.h"
#include "MeshBlock.h"
#include "Model.h"
#include "Patch.h"
namespace simpla {
namespace engine {

struct MeshViewFactory::pimpl_s {
    std::map<std::string, std::function<std::shared_ptr<MeshView>(std::shared_ptr<data::DataEntity> const &,
                                                                  std::shared_ptr<geometry::GeoObject> const &)>>
        m_mesh_factory_;
};

MeshViewFactory::MeshViewFactory() : m_pimpl_(new pimpl_s){};
MeshViewFactory::~MeshViewFactory(){};

bool MeshViewFactory::RegisterCreator(
    std::string const &k,
    std::function<std::shared_ptr<MeshView>(std::shared_ptr<data::DataEntity> const &,
                                            std::shared_ptr<geometry::GeoObject> const &)> const &fun) {
    return m_pimpl_->m_mesh_factory_.emplace(k, fun).second;
};

std::shared_ptr<MeshView> MeshViewFactory::Create(std::shared_ptr<data::DataEntity> const &config,
                                                  std::shared_ptr<geometry::GeoObject> const &g) {
    std::shared_ptr<MeshView> res = nullptr;
    if (config == nullptr) {
        WARNING << "Create Mesh failed!" << std::endl;
        return res;
    } else if (config->value_type_info() == typeid(std::string)) {
        res = m_pimpl_->m_mesh_factory_.at(data::data_cast<std::string>(*config))(nullptr, g);
    } else if (config->isTable()) {
        auto const &t = config->cast_as<data::DataTable>();
        res = m_pimpl_->m_mesh_factory_.at(t.GetValue<std::string>("name"))(config, g);
    }

    if (res != nullptr) { LOGGER << "MeshView [" << res->name() << "] is created!" << std::endl; }
    return res;
}

struct MeshView::pimpl_s {
    std::shared_ptr<MeshBlock> m_mesh_block_;
    std::shared_ptr<geometry::GeoObject> m_geo_obj_;
};
MeshView::MeshView(std::shared_ptr<data::DataEntity> const &t, std::shared_ptr<geometry::GeoObject> const &geo_obj)
    : AttributeViewBundle(t), m_pimpl_(new pimpl_s) {
    m_pimpl_->m_geo_obj_ = geo_obj;
    if (m_pimpl_->m_geo_obj_ == nullptr) {
        m_pimpl_->m_geo_obj_ = GLOBAL_GEO_OBJECT_FACTORY.Create(db()->Get("GeometryObject"));
    }
    db()->SetValue("GeometryObject", *m_pimpl_->m_geo_obj_);
}
MeshView::~MeshView() {}

std::ostream &MeshView::Print(std::ostream &os, int indent) const {
    os << std::setw(indent + 1) << "value_type_info = \"" << getClassName() << "\",";
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


bool MeshView::Update() { return SPObject::Update(); }
std::shared_ptr<geometry::GeoObject> MeshView::GetGeoObject() const { return m_pimpl_->m_geo_obj_; }
void MeshView::PushPatch(std::shared_ptr<Patch> const &p) {
    m_pimpl_->m_mesh_block_ = p->GetMeshBlock();
    AttributeViewBundle::PushPatch(p);
};
std::shared_ptr<Patch> MeshView::PopPatch() const {
    auto res = AttributeViewBundle::PopPatch();
    res->SetMeshBlock(m_pimpl_->m_mesh_block_);
    return res;
}
id_type MeshView::GetMeshBlockId() const {
    return m_pimpl_->m_mesh_block_ == nullptr ? NULL_ID : m_pimpl_->m_mesh_block_->GetGUID();
}
std::shared_ptr<MeshBlock> const &MeshView::GetMeshBlock() const { return m_pimpl_->m_mesh_block_; }
void MeshView::SetMeshBlock(std::shared_ptr<MeshBlock> const &m) {
    if (m == m_pimpl_->m_mesh_block_) {
        return;
    } else
        m_pimpl_->m_mesh_block_ = m;
    Click();
}
void MeshView::Initialize() {}

Real MeshView::GetDt() const { return 1.0; }

}  // {namespace mesh
}  // namespace simpla
