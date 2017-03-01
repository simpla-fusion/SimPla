//
// Created by salmon on 17-2-16.
//

#include "Manager.h"
namespace simpla {
namespace engine {
struct Manager::pimpl_s {
    std::map<id_type, std::shared_ptr<Patch>> m_patches_;
    std::map<id_type, std::shared_ptr<DomainView>> m_views_;
    Atlas m_atlas_;
    model::Model m_model_;
    std::string m_name_;
    AttributeDict m_attr_db_;
};
Manager::Manager() : m_pimpl_(new pimpl_s) {}
Manager::~Manager() {}

Atlas const &Manager::GetAtlas() const { return m_pimpl_->m_atlas_; }
Atlas &Manager::GetAtlas() { return m_pimpl_->m_atlas_; }
model::Model &Manager::GetModel() { return m_pimpl_->m_model_; }
model::Model const &Manager::GetModel() const { return m_pimpl_->m_model_; }

void Manager::SetDomainView(id_type d_type_id, std::shared_ptr<DomainView> const &p) {
    m_pimpl_->m_views_[d_type_id] = p != nullptr ? p : std::make_shared<DomainView>();
}
void Manager::SetDomainView(std::string const &d_name, std::shared_ptr<DomainView> const &p) {
    SetDomainView(m_pimpl_->m_model_.GetMaterial(d_name).template GetValue<id_type>("GetGUID"), p);
}

DomainView const &Manager::GetDomainView(id_type d_id) const { return *m_pimpl_->m_views_.at(d_id); }
DomainView const &Manager::GetDomainView(std::string const &d_name) const {
    return GetDomainView(m_pimpl_->m_model_.GetMaterial(d_name).GetValue<id_type>("GetGUID"));
}
DomainView &Manager::GetDomainView(id_type d_id) {
    Click();
    auto it = m_pimpl_->m_views_.find(d_id);
    if (it == m_pimpl_->m_views_.end()) { m_pimpl_->m_views_.emplace(d_id, std::make_shared<DomainView>()); }
    return *m_pimpl_->m_views_.at(d_id);
}
DomainView &Manager::GetDomainView(std::string const &d_name) {
    return GetDomainView(m_pimpl_->m_model_.GetMaterial(d_name).GetValue<id_type>("GetGUID"));
}

AttributeDict &Manager::GetAttributeDatabase() { return m_pimpl_->m_attr_db_; }
AttributeDict const &Manager::GetAttributeDatabase() const { return m_pimpl_->m_attr_db_; }
Real Manager::GetTime() const { return 1.0; }
void Manager::Run(Real dt) {}
bool Manager::Update() { return SPObject::Update(); };
void Manager::Evaluate() {
    Update();
    //    for (auto &item : m_pimpl_->m_atlas_) {}
};
}
}