//
// Created by salmon on 16-6-2.
//

#include <simpla/geometry/BoxUtilities.h>
#include "simpla/geometry/GeoObject.h"

#include "Model.h"

#include "EngineObject.h"
#include "simpla/utilities/Factory.h"

namespace simpla {
namespace engine {

struct Model::pimpl_s {
    box_type m_bounding_box_{{0, 0, 0}, {0, 0, 0}};

    std::shared_ptr<Model> m_parent_;
    std::map<std::string, std::shared_ptr<geometry::GeoObject>> m_g_objs_;
    std::map<std::string, Model::attr_fun> m_attr_fun_;
    std::map<std::string, Model::vec_attr_fun> m_vec_attr_fun_;
};

Model::Model() : m_pimpl_(new pimpl_s) {}
Model::~Model() { delete m_pimpl_; };
void Model::Load(std::string const& url) {}
std::shared_ptr<data::DataNode> Model::Serialize() const {
    auto tdb = base_type::Serialize();
    //    for (auto const& item : m_pimpl_->m_g_objs_) {
    //        if (item.second != nullptr) { tdb->Set(item.first, item.second->Serialize()); }
    //    }
    return tdb;
};
void Model::Deserialize(std::shared_ptr<data::DataNode> const& tdb) {
    base_type::Deserialize(tdb);
    if (tdb != nullptr) {
        tdb->Foreach([&](std::string const& k, std::shared_ptr<data::DataNode> const& v) {
            if (v != nullptr) { Add(k, geometry::GeoObject::New(v)); }
            return (v != nullptr) ? 1 : 0;
        });
    }
};
void Model::DoSetUp() {
    auto it = m_pimpl_->m_g_objs_.begin();
    if (it == m_pimpl_->m_g_objs_.end() || it->second == nullptr) { return; }
    m_pimpl_->m_bounding_box_ = it->second->GetBoundingBox();
    ++it;
    for (; it != m_pimpl_->m_g_objs_.end(); ++it) {
        if (it->second != nullptr) {
            m_pimpl_->m_bounding_box_ = geometry::Union(m_pimpl_->m_bounding_box_, it->second->GetBoundingBox());
        }
    }
}
void Model::DoUpdate(){

};
void Model::DoTearDown() {}

box_type const& Model::GetBoundingBox() const { return m_pimpl_->m_bounding_box_; };

std::shared_ptr<geometry::GeoObject> Model::Get(std::string const& k) const {
    std::shared_ptr<geometry::GeoObject> res = nullptr;

    auto it = m_pimpl_->m_g_objs_.find(k);
    if (it != m_pimpl_->m_g_objs_.end()) { res = it->second; }

    return res;
}
size_type Model::Delete(std::string const& k) { return m_pimpl_->m_g_objs_.erase(k); }

size_type Model::Add(std::string const& k, std::shared_ptr<geometry::GeoObject> const& g) {
    m_pimpl_->m_g_objs_[k] = (g);
    return 1;
}
size_type Model::AddAttribute(std::string const& model_name, std::string const& function_name, attr_fun) { return 0; }
size_type Model::AddAttribute(std::string const& model_name, std::string const& function_name, vec_attr_fun) {
    return 0;
}

void Model::LoadAttribute(std::string const& k, Attribute* f) const {}

std::shared_ptr<Model> Model::GetParent() const { return m_pimpl_->m_parent_; }
void Model::SetParent(std::shared_ptr<Model> const& m) { m_pimpl_->m_parent_ = m; };

//
// void Model::SetObject(std::string const& key, std::shared_ptr<geometry::GeoObject> const& g_obj) {
//    if (g_obj != nullptr) {
//        VERBOSE << "AddEntity GeoObject [ " << key << " : " << g_obj->TypeName() << " ]";
//        //        m_pimpl_->m_g_objs_[key] = g_obj;
//    }
//}
//
// std::shared_ptr<geometry::GeoObject> Model::GetGeoObject(std::string const& k) const {
//    //    auto it = m_pimpl_->m_g_objs_.find(k);
//    //    return it == m_pimpl_->m_g_objs_.end() ? nullptr : it->second;
//    FIXME;
//    return nullptr;
//}
//
// size_type Model::DeleteObject(std::string const& key) {
//    //    return m_pimpl_->m_g_objs_.erase(key);
//    FIXME;
//    return 0;
//}
//
// std::map<std::string, std::shared_ptr<geometry::GeoObject>> const& Model::GetAll() const {
//    FIXME;
//    return m_pimpl_->m_g_objs_;
//};

}  // namespace engine
}  // namespace simpla{;