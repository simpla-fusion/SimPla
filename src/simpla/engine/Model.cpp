//
// Created by salmon on 16-6-2.
//

#include "simpla/geometry/GeoObject.h"

#include "Model.h"

#include "SPObject.h"
#include "simpla/utilities/Factory.h"

namespace simpla {
namespace engine {
REGISTER_CREATOR(Model, Model)

struct Model::pimpl_s {
    std::map<std::string, std::shared_ptr<geometry::GeoObject>> m_g_objs_;
    box_type m_bound_box_{{0, 0, 0}, {0, 0, 0}};
};

Model::Model() : m_pimpl_(new pimpl_s) {}
Model::~Model() { delete m_pimpl_; };
std::shared_ptr<Model> Model::New() { return std::shared_ptr<Model>(new Model); }

void Model::Serialize(data::DataTable& cfg) const {
    base_type::Serialize(cfg);
    for (auto const& item : m_pimpl_->m_g_objs_) {
        if (item.second != nullptr) { item.second->Serialize(cfg.GetTable(item.first)); }
    }
};
void Model::Deserialize(const DataTable& cfg) {
    base_type::Deserialize(cfg);
    cfg.Foreach([&](std::string const& k, std::shared_ptr<data::DataEntity> v) {
        if (v != nullptr) { SetObject(k, CreateObject<geometry::GeoObject>(v.get())); }
        return (v != nullptr) ? 1 : 0;
    });
};
void Model::DoInitialize() {}
void Model::DoFinalize() {}

void Model::DoUpdate() {
    auto it = m_pimpl_->m_g_objs_.begin();
    if (it == m_pimpl_->m_g_objs_.end() || it->second == nullptr) { return; }
    m_pimpl_->m_bound_box_ = it->second->BoundingBox();
    ++it;
    for (; it != m_pimpl_->m_g_objs_.end(); ++it) {
        if (it->second != nullptr) {
            m_pimpl_->m_bound_box_ = geometry::Union(m_pimpl_->m_bound_box_, it->second->BoundingBox());
        }
    }
};
void Model::DoTearDown() {}

box_type const& Model::BoundingBox() const { return m_pimpl_->m_bound_box_; };

void Model::SetObject(std::string const& key, std::shared_ptr<geometry::GeoObject> const& g_obj) {
    if (g_obj != nullptr) {
        VERBOSE << "Add GeoObject [ " << key << " : " << g_obj->GetFancyTypeName() << " ]" << std::endl;
        m_pimpl_->m_g_objs_[key] = g_obj;
    }
}

std::shared_ptr<geometry::GeoObject> Model::GetGeoObject(std::string const& k) const {
    auto it = m_pimpl_->m_g_objs_.find(k);
    return it == m_pimpl_->m_g_objs_.end() ? nullptr : it->second;
}

size_type Model::DeleteObject(std::string const& key) { return m_pimpl_->m_g_objs_.erase(key); }
std::map<std::string, std::shared_ptr<geometry::GeoObject>> const& Model::GetAll() const {
    return m_pimpl_->m_g_objs_;
};

}  // namespace engine
}  // namespace simpla{;