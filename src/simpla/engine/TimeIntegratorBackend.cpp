//
// Created by salmon on 17-4-5.
//

#include "TimeIntegratorBackend.h"
#include "Context.h"
namespace simpla {
namespace engine {

TimeIntegratorBackendFactory::TimeIntegratorBackendFactory(){};

TimeIntegratorBackendFactory::~TimeIntegratorBackendFactory(){};

bool TimeIntegratorBackendFactory::RegisterCreator(std::string const &k,
                                                   std::function<TimeIntegratorBackend *()> const &fun) {
    auto res = m_TimeIntegrator_factory_.emplace(k, fun).second;
    if (res) { LOGGER << "TimeIntegrator Creator [ " << k << " ] is registered!" << std::endl; }
    return res;
};

TimeIntegratorBackend *TimeIntegratorBackendFactory::Create(std::shared_ptr<data::DataTable> const &config) {
    std::string k = "";
    if (config != nullptr && !config->isNull()) {
        k = config->cast_as<data::DataTable>().GetValue<std::string>("Backend", k);
    }
    auto it = m_TimeIntegrator_factory_.find(k);
    if (it == m_TimeIntegrator_factory_.end()) {
        LOGGER << "Can not find time integrator creator[" << k << "], \"Dummy\" TimeIntegrator is created!"
               << std::endl;
        return new DummyTimeIntegratorBackend;
    } else {
        LOGGER << "TimeIntegrator [" << k << "] is created!" << std::endl;
        return it->second();
    }
}

TimeIntegratorBackend::TimeIntegratorBackend(std::shared_ptr<Context> const &ctx)
    : m_ctx_(ctx != nullptr ? ctx : std::make_shared<Context>()) {}
TimeIntegratorBackend::~TimeIntegratorBackend() {}
void TimeIntegratorBackend::SetContext(std::shared_ptr<Context> const &ctx) {
    if (ctx == nullptr) { return; }
    m_ctx_ = ctx;
    db()->Link("Context", m_ctx_->db());
}
std::shared_ptr<Context> const &TimeIntegratorBackend::GetContext() const { return m_ctx_; }
std::shared_ptr<Context> &TimeIntegratorBackend::GetContext() { return m_ctx_; }
}  // namespace engine{
}  // namespace simpla{