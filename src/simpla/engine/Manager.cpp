//
// Created by salmon on 17-2-16.
//
#include "Manager.h"
#include <simpla/data/all.h>
#include "MeshView.h"
#include "Worker.h"
namespace simpla {
namespace engine {

struct Manager::pimpl_s {
    std::map<id_type, std::shared_ptr<Patch>> m_patches_;
    std::map<std::string, std::shared_ptr<DomainView>> m_views_;
    Atlas m_atlas_;
    Model m_model_;
};

Manager::Manager() : m_pimpl_(new pimpl_s) {
    db()->Link("Model", m_pimpl_->m_model_.db());
    db()->Link("Atlas", m_pimpl_->m_atlas_.db());
}

Manager::~Manager() {}

std::ostream &Manager::Print(std::ostream &os, int indent) const { return db()->Print(os, indent); }

Atlas &Manager::GetAtlas() const { return m_pimpl_->m_atlas_; }

Model &Manager::GetModel() const { return m_pimpl_->m_model_; }

std::shared_ptr<DomainView> Manager::GetDomainView(std::string const &d_name) const {
    return m_pimpl_->m_views_.at(d_name);
}

void Manager::SetDomainView(std::string const &d_name, std::shared_ptr<data::DataTable> const &p) {
    db()->Set("DomainView/" + d_name, *p, false);
}

Real Manager::GetTime() const { return 1.0; }
void Manager::Run(Real dt) { Update(); }
bool Manager::Update() { return SPObject::Update(); };

void Manager::Initialize() {
    LOGGER << "Manager " << name() << " is initializing!" << std::endl;
    GetModel().Initialize();
    GetAtlas().Initialize();
    auto domain_view_list = db()->Get("DomainView");
    if (domain_view_list == nullptr || !domain_view_list->isTable()) { return; }

    auto &domain_t = domain_view_list->cast_as<data::DataTable>();
    domain_t.Foreach([&](std::string const &s_key, std::shared_ptr<data::DataEntity> const &item) {
        auto res = m_pimpl_->m_views_.emplace(s_key, nullptr);

        if (res.first->second == nullptr) {
            res.first->second = std::make_shared<DomainView>(
                (item != nullptr && item->isTable()) ? std::dynamic_pointer_cast<data::DataTable>(item) : nullptr);
        } else {
            if (item != nullptr && item->isTable()) {
                res.first->second->db()->Set(*std::dynamic_pointer_cast<data::DataTable>(item));
            }
        }
        domain_t.Set(s_key, res.first->second->db(), true);
        res.first->second->name(s_key);
//        res.first->second->SetMesh(GetModel().GetMesh(s_key));
        res.first->second->Initialize();
    });
    SPObject::Tag();
    LOGGER << "Manager " << name() << " is initialized!" << std::endl;
}
}  // namespace engine {
}  // namespace simpla {
