//
// Created by salmon on 17-2-16.
//
#include "Manager.h"
#include <simpla/data/all.h>
#include "MeshView.h"
#include "Worker.h"
namespace simpla {
namespace engine {
static std::map<std::string, std::function<std::shared_ptr<MeshView>()>> g_mesh_factory_;
static std::map<std::string, std::function<std::shared_ptr<Worker>()>> g_worker_factory_;

bool Manager::RegisterMeshCreator(std::string const &k, std::function<std::shared_ptr<MeshView>()> const &fun) {
    return g_mesh_factory_.emplace(k, fun).second;
};
bool Manager::RegisterWorkerCreator(std::string const &k, std::function<std::shared_ptr<Worker>()> const &fun) {
    return g_worker_factory_.emplace(k, fun).second;
};

std::shared_ptr<MeshView> CreateMesh(std::shared_ptr<data::DataEntity> const &config) {
    if (config->type() == typeid(std::string)) {
        return g_mesh_factory_.at(data::data_cast<std::string>(*config))();
    } else if (config->isTable()) {
        auto const &t = config->cast_as<data::DataTable>();
        auto m = g_mesh_factory_.at(t.GetValue<std::string>("name"))();
        m->db()->Link("", t);
        return m;
    }
}

std::shared_ptr<Worker> CreateWorker(std::shared_ptr<MeshView> const &m,
                                     std::shared_ptr<data::DataEntity> const &config) {
    if (config == nullptr || config->isNull()) {
        return nullptr;
    } else if (config->type() == typeid(std::string)) {
        return g_worker_factory_.at(m->name() + "." + data::data_cast<std::string>(*config))();
    } else if (config->isTable()) {
        auto const &t = config->cast_as<data::DataTable>();
        auto w = g_worker_factory_.at(m->name() + "." + t.GetValue<std::string>("name"))();
        w->db()->Link("", t);
        return w;
    }
}

struct Manager::pimpl_s {
    std::map<id_type, std::shared_ptr<Patch>> m_patches_;
    std::map<id_type, std::shared_ptr<DomainView>> m_views_;
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

std::shared_ptr<data::DataTable> Manager::GetAttributeDatabase() const { return db()->GetTable("AttributeDesc"); }

DomainView &Manager::GetDomainView(std::string const &d_name) const {
    return *m_pimpl_->m_views_.at(m_pimpl_->m_model_.GetMaterialId(d_name));
}

std::shared_ptr<DomainView> Manager::SetDomainView(std::string const &d_name,
                                                   std::shared_ptr<data::DataEntity> const &p) {
    auto p0 = db()->Set(d_name, p, false);
    auto &view_table = p0.first->cast_as<data::DataTable>();
    auto d_view = std::make_shared<DomainView>();
    auto t_mesh = d_view->SetMesh(CreateMesh(view_table.Get("Mesh")));

    auto attr_table = GetAttributeDatabase();
    view_table.Get("Worker")->cast_as<data::DataArray>().ForEach([&](std::shared_ptr<data::DataEntity> const &item) {
        auto w = d_view->AddWorker(CreateWorker(t_mesh, item));
        w.first->db()
            ->GetTable("Attributes")
            ->ForEach([&](std::string const &k, std::shared_ptr<data::DataEntity> const &v) {
                auto res = attr_table->Set(k, v, false);
                if (res.second) { v->cast_as<data::DataTable>().Link("", attr_table->GetTable(k)); }
            });

    });
}
std::shared_ptr<DomainView> Manager::SetDomainView(std::string const &d_name, std::shared_ptr<DomainView> const &p,
                                                   bool overwrite) {
    auto d_id = m_pimpl_->m_model_.GetMaterialId(d_name);
    auto res = m_pimpl_->m_views_.emplace(d_id, p);
    if (!res.second) { return res.first->second; }
    Click();
    auto t_table = db()->GetTable("Domains/" + d_name);
    if (res.first->second == nullptr) {
        res.first->second = std::make_shared<DomainView>();
        res.first->second->db()->Link("", db()->GetTable("Domains/" + d_name));
    } else {
        res.first->second->db()->Set(*db()->GetTable("Domains/" + d_name));
    }
    db()->Link("Domains/" + d_name, res.first->second->db());
    return res.first->second;
}

Real Manager::GetTime() const { return 1.0; }
void Manager::Run(Real dt) { Update(); }
bool Manager::Update() { return SPObject::Update(); };

void Manager::Initialize() {
    db()->GetTable("DomainView")->ForEach([&](std::string const &s_key, std::shared_ptr<data::DataEntity> const &item) {
        SetDomainView(s_key, item);
    });
}
}  // namespace engine {
}  // namespace simpla {
