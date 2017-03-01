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
    std::multimap<id_type, geometry::GeoObject> m_g_obj_;
    box_type m_bound_box_{{0, 0, 0}, {1, 1, 1}};
};

Model::Model() : m_pimpl_(new pimpl_s) {
    db().CreateTable("Material");
    SPObject::Click();
}

Model::~Model() {}

std::ostream& Model::Print(std::ostream& os, int indent) const {
    os << db() << std::endl;
    return os;
}
bool Model::Update() {
    // TODO: update bound box
    return SPObject::Update();
};
box_type const& Model::bound_box() const { return m_pimpl_->m_bound_box_; };
data::DataTable const& Model::GetMaterial(std::string const& s) const { return db().asTable("Material." + s); }
data::DataTable& Model::GetMaterial(std::string const& s) {
    std::string url = "Material." + s;
    if (!db().has(url)) { db().CreateTable(url)->SetValue("GetGUID", std::hash<std::string>{}(s)); }
    return db().asTable(url);
}
id_type Model::AddObject(id_type id, geometry::GeoObject const &g_obj) {
    Click();
    m_pimpl_->m_g_obj_.emplace(id, g_obj);
}

id_type Model::AddObject(std::string const &key, geometry::GeoObject const &g_obj) {
    Click();
    m_pimpl_->m_g_obj_.emplace(GetMaterial(key).GetValue<id_type>("GetGUID"), g_obj);
}

size_type Model::RemoveObject(std::string const& key) {
    Click();
    return m_pimpl_->m_g_obj_.erase(GetMaterial(key).GetValue<id_type>("GetGUID"));
}
}  // namespace model
}  // namespace simpla{;