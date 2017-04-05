//
// Created by salmon on 17-3-17.
//
#include "TimeIntegrator.h"

#include <simpla/algebra/nTupleExt.h>
#include <simpla/concept/Configurable.h>
#include <simpla/data/all.h>
#include "Context.h"
#include "TimeIntegratorBackend.h"
namespace simpla {
namespace engine {

TimeIntegrator::TimeIntegrator(std::string const &s_engine, std::shared_ptr<Context> const &ctx)
    : TimeIntegrator(std::make_shared<data::DataTable>("Backend"_ = s_engine), ctx){};

TimeIntegrator::TimeIntegrator(std::shared_ptr<data::DataTable> const &t, std::shared_ptr<Context> const &ctx)
    : Schedule(t), m_backend_(GLOBAL_TIME_INTEGRATOR_FACTORY.Create(Schedule::db())) {
    m_backend_->SetContext(ctx);
}

TimeIntegrator::~TimeIntegrator() { Finalize(); }

std::shared_ptr<Context> &TimeIntegrator::GetContext() { return m_backend_->GetContext(); }
std::shared_ptr<Context> const &TimeIntegrator::GetContext() const { return m_backend_->GetContext(); }
void TimeIntegrator::Initialize() { m_backend_->Initialize(); }
void TimeIntegrator::Finalize() { m_backend_->Finalize(); }
void TimeIntegrator::NextTimeStep(Real dt) { m_backend_->NextTimeStep(dt); };
bool TimeIntegrator::RemainingSteps() const { return m_backend_->RemainingSteps(); }
Real TimeIntegrator::CurrentTime() const { return m_backend_->CurrentTime(); }

//    if (level >= m_pimpl_->m_ctx_->GetAtlas().GetNumOfLevels()) { return m_pimpl_->m_time_; }
//    auto &atlas = m_pimpl_->m_ctx_->GetAtlas();
//    //    for (auto const &id : atlas.GetBlockList(level)) {
//    //        auto mblk = atlas.GetBlock(id);
//    //        for (auto &v : m_pimpl_->m_ctx_->GetAllDomains()) {
//    //            if (!v.second->GetGeoObject()->CheckOverlap(mblk->GetBoundBox())) { continue; }
//    //            auto res = m_pimpl_->m_ctx_->GetPatches()->GetTable(std::to_string(id));
//    //            if (res == nullptr) { res = std::make_shared<data::DataTable>(); }
//    //            v.second->PushData(mblk, res);
//    //            LOGGER << " Domain [ " << std::setw(10) << std::left << v.second->name() << " ] is applied on "
//    //                   << mblk->GetIndexBox() << " id= " << id << std::endl;
//    //            v.second->Run(dt);
//    //            auto t = v.second->PopData().second;
//    //            m_pimpl_->m_ctx_->GetPatches()->Set(std::to_string(id), t);
//    //        }
//    //    }
//    m_pimpl_->m_time_ += dt;
//    return m_pimpl_->m_time_;
//    //    for (auto const &item : atlas.GetLayer(level)) {
//    //        for (auto &v : m_pimpl_->m_domains_) {
//    //            auto b_box = v.second->GetMesh()->inner_bound_box();
//    //            if (!geometry::check_overlap(item.second->GetBox(), b_box)) { continue; }
//    //            v.second->Dispatch(m_pimpl_->m_patches_[item.first]);
//    //            v.second->Run(dt);
//    //        }
//    //    }
//    //    for (int i = 0; i < m_pimpl_->m_refine_ratio_; ++i) { Run(dt / m_pimpl_->m_refine_ratio_, level + 1); }
//    //    for (auto const &item : atlas.GetLayer(level)) {
//    //        for (auto &v : m_pimpl_->m_domains_) {
//    //            auto b_box = v.second->GetMesh()->GetGeoObject()->GetBoundBox();
//    //            if (!geometry::check_overlap(item.second->GetBox(), b_box)) { continue; }
//    //            v.second->Dispatch(m_pimpl_->m_patches_[item.first]);
//    //            v.second->Run(dt);
//    //        }
//    //    }

}  // namespace engine{
}  // namespace simpla{