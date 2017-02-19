//
// Created by salmon on 17-2-16.
//

#include "Manager.h"
namespace simpla {
namespace engine {
struct Manager::pimpl_s {
    std::map<id_type, std::shared_ptr<Patch>> m_patches_;
    std::map<id_type, std::unique_ptr<DomainView>> m_views_;
    Atlas m_atlas_;
    model::Model m_model_;
    std::string m_name_;
};
Manager::Manager() : m_pimpl_(new pimpl_s) {}
Manager::~Manager() {}
std::string const &Manager::name() const { return m_pimpl_->m_name_; }
model::Model &Manager::GetModel() { return m_pimpl_->m_model_; }
model::Model const &Manager::GetModel() const { return m_pimpl_->m_model_; }

void Manager::SetDomainView(id_type geo_obj_id, std::unique_ptr<DomainView> &&p) {}
DomainView &Manager::GetDomainView(id_type d_id) { return *m_pimpl_->m_views_.at(d_id); }
void Manager::Update(){};
void Manager::Evaluate() {
//    for (auto &item : m_pimpl_->m_atlas_) {}
};
}
}