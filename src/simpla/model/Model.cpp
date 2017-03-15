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
    db().Set("Material/");
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
data::DataTable const& Model::GetMaterial(std::string const& s) const { return db().GetTable(s); }
data::DataTable& Model::GetMaterial(std::string const& s) {
    Click();
    auto& t = db().GetTable("/Material/" + s);
    t.GetValue("GUID", std::hash<std::string>{}(s));
    return t;
}
// id_type Model::GetMaterialId(std::string const& k) const { return GetMaterial(k).GetValue<id_type>("GUID", NULL_ID);
// }
// id_type Model::GetMaterialId(std::string const& k) { return GetMaterial(k).GetValue<id_type>("GUID"); }

// id_type Model::AddObject(id_type id, geometry::GeoObject const &g_obj) {
//    Click();
//    m_pimpl_->m_g_obj_.emplace(id, g_obj);
//}

id_type Model::AddObject(std::string const& key, geometry::GeoObject const& g_obj) {
    Click();
    m_pimpl_->m_g_obj_.emplace(GetMaterial(key).GetValue<id_type>("GUID"), g_obj);
}
size_type Model::RemoveObject(id_type id) {
    Click();
    return m_pimpl_->m_g_obj_.erase(id);
}

size_type Model::RemoveObject(std::string const& key) {
    try {
        return RemoveObject(GetMaterial(key).GetValue<id_type>("GUID"));
    } catch (...) { return 0; }
}

}  // namespace model
}  // namespace simpla{;