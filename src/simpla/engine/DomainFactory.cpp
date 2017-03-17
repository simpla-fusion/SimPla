//
// Created by salmon on 17-3-17.
//
#include "DomainFactory.h"
#include "MeshView.h"
#include "Worker.h"
namespace simpla {
namespace engine {
struct DomainFactory::pimpl_s {
    std::map<std::string, std::function<std::shared_ptr<MeshView>(std::shared_ptr<data::DataTable> const &)>>
        m_mesh_factory_;
    std::map<std::string, std::function<std::shared_ptr<Worker>(std::shared_ptr<data::DataTable> const &)>>
        m_worker_factory_;
};

DomainFactory::DomainFactory() : m_pimpl_(new pimpl_s){};
DomainFactory::~DomainFactory(){};

bool DomainFactory::RegisterMeshCreator(
    std::string const &k,
    std::function<std::shared_ptr<MeshView>(std::shared_ptr<data::DataTable> const &)> const &fun) {
    return m_pimpl_->m_mesh_factory_.emplace(k, fun).second;
};
bool DomainFactory::RegisterWorkerCreator(
    std::string const &k, std::function<std::shared_ptr<Worker>(std::shared_ptr<data::DataTable> const &)> const &fun) {
    return m_pimpl_->m_worker_factory_.emplace(k, fun).second;
};

std::shared_ptr<MeshView> DomainFactory::CreateMesh(std::shared_ptr<data::DataEntity> const &config) {
    std::shared_ptr<MeshView> res = nullptr;
    if (config == nullptr) {
        return res;
    } else if (config->type() == typeid(std::string)) {
        res = m_pimpl_->m_mesh_factory_.at(data::data_cast<std::string>(*config))(nullptr);
    } else if (config->isTable()) {
        auto const &t = config->cast_as<data::DataTable>();
        res = m_pimpl_->m_mesh_factory_.at(t.GetValue<std::string>("name"))(
            std::dynamic_pointer_cast<data::DataTable>(config));
    }

    if (res != nullptr) { LOGGER << "MeshView [" << res->name() << "] is created!" << std::endl; }
    return res;
}

std::shared_ptr<Worker> DomainFactory::CreateWorker(std::shared_ptr<MeshView> const &m,
                                                    std::shared_ptr<data::DataEntity> const &config) {
    std::shared_ptr<Worker> res = nullptr;
    std::string s_name = "";
    if (config == nullptr || config->isNull()) {
        return res;
    } else if (config->type() == typeid(std::string)) {
        s_name = m->name() + "." + data::data_cast<std::string>(*config);
        res = m_pimpl_->m_worker_factory_.at(s_name)(nullptr);
    } else if (config->isTable()) {
        auto const &t = config->cast_as<data::DataTable>();
        s_name = m->name() + "." + t.GetValue<std::string>("name");
        res = m_pimpl_->m_worker_factory_.at(s_name)(std::dynamic_pointer_cast<data::DataTable>(config));
    }
    LOGGER << "Worker [" << s_name << "] is created!" << std::endl;
    return res;
}
}  // namespace engine{
}  // namespace simpla