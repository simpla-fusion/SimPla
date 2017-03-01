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
id_type Model::AddObject(geometry::GeoObject const& g_obj, id_type id) {
    Click();
    m_pimpl_->m_g_obj_.emplace(id, g_obj);
}

id_type Model::AddObject(geometry::GeoObject const& g_obj, std::string const& key) {
    Click();
    m_pimpl_->m_g_obj_.emplace(GetMaterial(key).GetValue<id_type>("GetGUID"), g_obj);
}
size_type Model::RemoveObject(id_type id) {
    Click();
    return m_pimpl_->m_g_obj_.erase(id);
}

size_type Model::RemoveObject(std::string const& key) {
    return RemoveObject(GetMaterial(key).GetValue<id_type>("GetGUID"));
}


}  // namespace model
}  // namespace simpla{;