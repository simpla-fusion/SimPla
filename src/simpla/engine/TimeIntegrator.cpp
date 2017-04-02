//
// Created by salmon on 17-3-17.
//
#include "TimeIntegrator.h"

#include <simpla/algebra/nTupleExt.h>
#include <simpla/concept/Configurable.h>
#include <simpla/data/all.h>
#include "Context.h"
namespace simpla {
namespace engine {

struct TimeIntegratorFactory::pimpl_s {
    std::map<std::string, std::function<std::shared_ptr<TimeIntegrator>(std::shared_ptr<Context> const &,
                                                                        std::shared_ptr<data::DataTable> const &)>>
        m_TimeIntegrator_factory_;
};

TimeIntegratorFactory::TimeIntegratorFactory()
    : m_pimpl_(new pimpl_s){

      };
TimeIntegratorFactory::~TimeIntegratorFactory(){};
bool TimeIntegratorFactory::RegisterCreator(
    std::string const &k,
    std::function<std::shared_ptr<TimeIntegrator>(std::shared_ptr<Context> const &,
                                                  std::shared_ptr<data::DataTable> const &)> const &fun) {
    auto res = m_pimpl_->m_TimeIntegrator_factory_.emplace(k, fun).second;

    if (res) { LOGGER << "TimeIntegrator Creator [ " << k << " ] is registered!" << std::endl; }

    return res;
};

std::shared_ptr<TimeIntegrator> TimeIntegratorFactory::Create(std::string const &s_name,
                                                              std::shared_ptr<Context> const &m,
                                                              std::shared_ptr<data::DataTable> const &d) {
    std::string k = s_name;
    auto it = m_pimpl_->m_TimeIntegrator_factory_.find(k);
    if (it == m_pimpl_->m_TimeIntegrator_factory_.end()) {
        LOGGER << "Can not find time integrator creator[" << k << "], \"Default\" TimeIntegrator is applied!"
               << std::endl;
        return std::make_shared<TimeIntegrator>(m, d);
    } else {
        LOGGER << "TimeIntegrator [" << k << "] is created!" << std::endl;
        return it->second(m, d);
    }
};

std::shared_ptr<TimeIntegrator> TimeIntegratorFactory::Create(std::shared_ptr<data::DataTable> const &config) {
    std::string s_name = "Default";

    if (config == nullptr || config->isNull()) {
    } else if (config->value_type_info() == typeid(std::string)) {
        s_name = data::data_cast<std::string>(*config);
    } else if (config->isTable()) {
        s_name = config->cast_as<data::DataTable>().GetValue<std::string>("name", s_name);
    }
    return Create(s_name, nullptr, config);
}

struct TimeIntegrator::pimpl_s {
    std::shared_ptr<Context> m_ctx_;
    Real m_time_;
};

TimeIntegrator::TimeIntegrator(std::shared_ptr<Context> const &m, std::shared_ptr<data::DataTable> const &t)
    : concept::Configurable(t), m_pimpl_(new pimpl_s) {
    m_pimpl_->m_ctx_ = m != nullptr ? m : std::make_shared<Context>(db()->GetTable("Context"));
}
TimeIntegrator::~TimeIntegrator() { Finalize(); }

void TimeIntegrator::Initialize() {
    if (m_pimpl_->m_ctx_ != nullptr) { m_pimpl_->m_ctx_->Initialize(); }
}
void TimeIntegrator::Finalize() { m_pimpl_->m_ctx_.reset(); }
void TimeIntegrator::SetContext(std::shared_ptr<Context> const &c) {
    Finalize();
    m_pimpl_->m_ctx_ = c;
}
std::shared_ptr<Context> const &TimeIntegrator::GetContext() const { return m_pimpl_->m_ctx_; }

size_type TimeIntegrator::NextTimeStep(Real dt) { return 0; };
size_type TimeIntegrator::step() const { return 0; };
bool TimeIntegrator::remainingSteps() const { return 0; };
Real TimeIntegrator::timeNow() const { return 0.0; }

Real TimeIntegrator::Advance(Real dt, int level) {
    if (level >= m_pimpl_->m_ctx_->GetAtlas().GetNumOfLevels()) { return m_pimpl_->m_time_; }
    auto &atlas = m_pimpl_->m_ctx_->GetAtlas();
    for (auto const &id : atlas.GetBlockList(level)) {
        auto mblk = atlas.GetBlock(id);
        for (auto &v : m_pimpl_->m_ctx_->GetAllDomains()) {
            if (!v.second->GetGeoObject()->CheckOverlap(mblk->GetBoundBox())) { continue; }
            auto res = m_pimpl_->m_ctx_->GetPatches()->GetTable(std::to_string(id));
            if (res == nullptr) { res = std::make_shared<data::DataTable>(); }
            v.second->PushData(mblk, res);
            LOGGER << " Domain [ " << std::setw(10) << std::left << v.second->name() << " ] is applied on "
                   << mblk->GetIndexBox() << " id= " << id << std::endl;
            v.second->Run(dt);
            auto t = v.second->PopData().second;
            m_pimpl_->m_ctx_->GetPatches()->Set(std::to_string(id), t);
        }
    }
    m_pimpl_->m_time_ += dt;
    return m_pimpl_->m_time_;
    //    for (auto const &item : atlas.GetLayer(level)) {
    //        for (auto &v : m_pimpl_->m_domains_) {
    //            auto b_box = v.second->GetMesh()->inner_bound_box();
    //            if (!geometry::check_overlap(item.second->GetBox(), b_box)) { continue; }
    //            v.second->Dispatch(m_pimpl_->m_patches_[item.first]);
    //            v.second->Run(dt);
    //        }
    //    }
    //    for (int i = 0; i < m_pimpl_->m_refine_ratio_; ++i) { Run(dt / m_pimpl_->m_refine_ratio_, level + 1); }
    //    for (auto const &item : atlas.GetLayer(level)) {
    //        for (auto &v : m_pimpl_->m_domains_) {
    //            auto b_box = v.second->GetMesh()->GetGeoObject()->GetBoundBox();
    //            if (!geometry::check_overlap(item.second->GetBox(), b_box)) { continue; }
    //            v.second->Dispatch(m_pimpl_->m_patches_[item.first]);
    //            v.second->Run(dt);
    //        }
    //    }
}

}  // namespace engine{
}  // namespace simpla{