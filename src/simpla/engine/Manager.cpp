//
// Created by salmon on 17-2-16.
//
#include "Manager.h"
#include <simpla/data/all.h>
#include "MeshView.h"
#include "Worker.h"
namespace simpla {
namespace engine {
static std::map<std::string, std::function<std::shared_ptr<MeshView>(std::shared_ptr<data::DataTable> const &)>>
    g_mesh_factory_;
static std::map<std::string, std::function<std::shared_ptr<Worker>(std::shared_ptr<data::DataTable> const &)>>
    g_worker_factory_;

bool Manager::RegisterMeshCreator(
    std::string const &k,
    std::function<std::shared_ptr<MeshView>(std::shared_ptr<data::DataTable> const &)> const &fun) {
    return g_mesh_factory_.emplace(k, fun).second;
};
bool Manager::RegisterWorkerCreator(
    std::string const &k, std::function<std::shared_ptr<Worker>(std::shared_ptr<data::DataTable> const &)> const &fun) {
    return g_worker_factory_.emplace(k, fun).second;
};

std::shared_ptr<MeshView> CreateMesh(std::shared_ptr<data::DataEntity> const &config) {
    std::shared_ptr<MeshView> res = nullptr;
    if (config == nullptr) {
        return res;
    } else if (config->type() == typeid(std::string)) {
        res = g_mesh_factory_.at(data::data_cast<std::string>(*config))(nullptr);
    } else if (config->isTable()) {
        auto const &t = config->cast_as<data::DataTable>();
        res = g_mesh_factory_.at(t.GetValue<std::string>("name"))(std::dynamic_pointer_cast<data::DataTable>(config));
    }

    if (res != nullptr) { LOGGER << "MeshView [" << res->name() << "] is created!" << std::endl; }
    return res;
}

std::shared_ptr<Worker> CreateWorker(std::shared_ptr<MeshView> const &m,
                                     std::shared_ptr<data::DataEntity> const &config) {
    std::shared_ptr<Worker> res = nullptr;
    std::string s_name = "";
    if (config == nullptr || config->isNull()) {
        return res;
    } else if (config->type() == typeid(std::string)) {
        s_name = m->name() + "." + data::data_cast<std::string>(*config);
        res = g_worker_factory_.at(s_name)(nullptr);
    } else if (config->isTable()) {
        auto const &t = config->cast_as<data::DataTable>();
        s_name = m->name() + "." + t.GetValue<std::string>("name");
        res = g_worker_factory_.at(s_name)(std::dynamic_pointer_cast<data::DataTable>(config));
    }
    LOGGER << "Worker [" << s_name << "] is created!" << std::endl;
    return res;
}

struct Manager::pimpl_s {
    std::map<id_type, std::shared_ptr<Patch>> m_patches_;
    std::map<std::string, std::shared_ptr<DomainView>> m_views_;
    Atlas m_atlas_;
    model::Model m_model_;
};

Manager::Manager() : m_pimpl_(new pimpl_s) {
    db()->Link("Model", m_pimpl_->m_model_.db());
    db()->Link("Atlas", m_pimpl_->m_atlas_.db());
}

Manager::~Manager() {}

std::ostream &Manager::Print(std::ostream &os, int indent) const { return db()->Print(os, indent); }

Atlas &Manager::GetAtlas() const { return m_pimpl_->m_atlas_; }

model::Model &Manager::GetModel() const { return m_pimpl_->m_model_; }

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

    auto domain_view_list = db()->Get("DomainView");
    if (domain_view_list == nullptr || !domain_view_list->isTable()) { return; }

    std::dynamic_pointer_cast<data::DataTable>(domain_view_list)
        ->ForEach([&](std::string const &s_key, std::shared_ptr<data::DataEntity> const &item) {
            auto res = m_pimpl_->m_views_.emplace(s_key, nullptr);
            if (res.first->second == nullptr) { res.first->second = std::make_shared<DomainView>(); }
            auto d_view = res.first->second;
            auto &view_table = *d_view->db();
            view_table.Link("", item);
            auto t_mesh = d_view->SetMesh(CreateMesh(view_table.Get("Mesh")));
            auto t_worker = view_table.Get("Worker");
            if (t_worker != nullptr) {
                t_worker->cast_as<data::DataArray>().ForEach([&](std::shared_ptr<data::DataEntity> const &c) {
                    auto w = d_view->AddWorker(CreateWorker(t_mesh, c));
                });
            }
            LOGGER << "Domain View [" << s_key << "] is created!" << std::endl;
        });
    SPObject::Tag();
    LOGGER << "Manager " << name() << " is initialized!" << std::endl;
}
}  // namespace engine {
}  // namespace simpla {
