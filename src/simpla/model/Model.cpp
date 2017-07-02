//
// Created by salmon on 16-6-2.
//
#include "Model.h"
#include <simpla/geometry/Cube.h>
#include <simpla/model/GEqdsk.h>
#include <simpla/engine/SPObject.h>
#include "simpla/engine/Attribute.h"
#include "simpla/engine/MeshBlock.h"
namespace simpla {
namespace model {

struct Model::pimpl_s {
    std::map<std::string, std::shared_ptr<geometry::GeoObject>> m_g_objs_;
    box_type m_bound_box_{{0, 0, 0}, {0, 0, 0}};
};

Model::Model() : m_pimpl_(new pimpl_s) {}
Model::~Model() {}
DataTable Model::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    for (auto const& item : m_pimpl_->m_g_objs_) {
        if (item.second != nullptr) { res->Set(item.first, item.second->Pack()); }
    }
    return res;
};
void Model::Deserialize(const DataTable &cfg) {
    if (cfg == nullptr) { return; }
    cfg->Foreach([&](std::string const& k, std::shared_ptr<data::DataEntity> const& v) {
        if (v != nullptr) { SetObject(k, geometry::GeoObject::Create(v)); }
    });
};
void Model::Initialize() { LOGGER << "Model is initialized " << std::endl; }
void Model::Finalize() {}

void Model::Update() {
    auto it = m_pimpl_->m_g_objs_.begin();
    if (it == m_pimpl_->m_g_objs_.end() || it->second == nullptr) { return; }
    m_pimpl_->m_bound_box_ = it->second->GetBoundBox();
    ++it;
    for (; it != m_pimpl_->m_g_objs_.end(); ++it) {
        if (it->second != nullptr) {
            if (it->second->hasChildren()) { continue; }
            m_pimpl_->m_bound_box_ = geometry::BoundBox(m_pimpl_->m_bound_box_, it->second->GetBoundBox());
        }
    }
};
void Model::TearDown() {}
int Model::GetNDims() const { return 3; }

box_type const& Model::GetBoundBox() const { return m_pimpl_->m_bound_box_; };

void Model::SetObject(std::string const& key, std::shared_ptr<geometry::GeoObject> const& g_obj) {
    if (g_obj != nullptr) {
        VERBOSE << "Add GeoObject [ " << key << " : " << g_obj->GetRegisterName() << " ]" << std::endl;
        m_pimpl_->m_g_objs_[key] = g_obj;
        if (g_obj->hasChildren()) { g_obj->Register(m_pimpl_->m_g_objs_, key); }
    }
}

std::shared_ptr<geometry::GeoObject> Model::GetObject(std::string const& k) const {
    auto it = m_pimpl_->m_g_objs_.find(k);
    return it == m_pimpl_->m_g_objs_.end() ? nullptr : it->second;
}

size_type Model::DeleteObject(std::string const& key) { return m_pimpl_->m_g_objs_.erase(key); }
std::map<std::string, std::shared_ptr<geometry::GeoObject>> const& Model::GetAll() const {
    return m_pimpl_->m_g_objs_;
};

}  // namespace engine
}  // namespace simpla{;