//
// Created by salmon on 16-6-2.
//

#include "Model.h"
#include <simpla/engine/AttributeView.h>
#include <simpla/engine/MeshBlock.h>
#include <simpla/mesh/EntityId.h>
namespace simpla {
namespace model {

struct Model::pimpl_s {
    std::multimap<id_type, std::shared_ptr<geometry::GeoObject>> m_g_obj_;
    box_type m_bound_box_{{0, 0, 0}, {1, 1, 1}};
};

Model::Model() : m_pimpl_(new pimpl_s) {
    db().CreateTable("Material");
    concept::Configurable::Click();
}

Model::~Model() {}

std::ostream& Model::Print(std::ostream& os, int indent) const {
    os << db() << std::endl;
    return os;
}
void Model::Update() {
    // TODO: update bound box
    concept::Configurable::Update();
};
box_type const& Model::bound_box() const { return m_pimpl_->m_bound_box_; };
data::DataTable const& Model::GetMaterial(std::string const& s) const { return db().asTable("Material." + s); }
data::DataTable& Model::GetMaterial(std::string const& s) {
    std::string url = "Material." + s;
    if (!db().has(url)) { db().CreateTable(url)->SetValue("GUID", std::hash<std::string>{}(s)); }
    return db().asTable(url);
}

void Model::AddObject(std::string const& key, std::shared_ptr<geometry::GeoObject> const& g_obj) {
    m_pimpl_->m_g_obj_.emplace(GetMaterial(key).GetValue<id_type>("GUID"), g_obj);
    concept::Configurable::Click();
}

void Model::RemoveObject(std::string const& key) {
    m_pimpl_->m_g_obj_.erase(GetMaterial(key).GetValue<id_type>("GUID"));
    concept::Configurable::Click();
}
}  // namespace model
}  // namespace simpla{;