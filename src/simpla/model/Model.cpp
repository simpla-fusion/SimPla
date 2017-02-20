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
    std::map<std::string, id_type> m_g_name_map_;
    std::multimap<id_type, std::shared_ptr<geometry::GeoObject>> m_g_obj_;
    box_type m_bound_box_{{0, 0, 0}, {1, 1, 1}};
};

Model::Model() : m_pimpl_(new pimpl_s) {
    auto d = db().CreateTable("Material");
    d->insert("VACUUM"_ = {"GUID"_ = std::hash<std::string>{}("VACUUM")});
    d->insert("PLASMA"_ = {"GUID"_ = std::hash<std::string>{}("PLASMA")});
    concept::Configurable::Click();
}

Model::~Model() {}

std::ostream& Model::Print(std::ostream& os, int indent) const {
    os << db() << std::endl;
    return os;
}
void Model::Update() { concept::Configurable::Update(); };
box_type const& Model::bound_box() const { return m_pimpl_->m_bound_box_; };
id_type Model::GetMaterialId(std::string const& s) const { return std::hash<std::string>{}(s); }

void Model::AddObject(std::string const& key, std::shared_ptr<geometry::GeoObject> const& g_obj) {
    int id = 0;
    //    auto it = m_pimpl_->m_g_name_map_.find(key);
    //    if (it != m_pimpl_->m_g_name_map_.end()) {
    //        id = it->second;
    //    } else {
    //         m_pimpl_->m_g_name_map_[key] = id;
    //    }
    m_pimpl_->m_g_obj_.emplace(GetMaterialId(key), g_obj);
    concept::Configurable::Click();
}

void Model::RemoveObject(std::string const& key) {
    m_pimpl_->m_g_obj_.erase(m_pimpl_->m_g_name_map_.at(key));
    concept::Configurable::Click();
}
}  // namespace model
}  // namespace simpla{