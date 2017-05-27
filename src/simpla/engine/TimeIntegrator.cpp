//
// Created by salmon on 17-3-17.
//
#include "TimeIntegrator.h"
#include "Context.h"
#include "simpla/data/all.h"
namespace simpla {
namespace engine {

std::shared_ptr<data::DataTable> TimeIntegrator::Serialize() const {
    auto p = Schedule::Serialize();
    p->SetValue("Type", GetRegisterName());
    p->SetValue("TimeBegin", GetTimeNow());
    p->SetValue("TimeEnd", GetTimeEnd());
    p->SetValue("TimeStep", GetTimeStep());
    p->SetValue("MaxStep", GetMaxStep());
    return p;
}

void TimeIntegrator::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    Schedule::Deserialize(cfg);
    SetTimeNow(cfg->GetValue("TimeBegin", 0.0));
    SetTimeEnd(cfg->GetValue("TimeEnd", 1.0));
    SetTimeStep(cfg->GetValue("TimeStep", 0.5));
    SetMaxStep(cfg->GetValue<size_type>("MaxStep", 0));
    SetCheckPointInterval(cfg->GetValue("CheckPointInterval", 1));
    SetDumpInterval(cfg->GetValue("DumpInterval", 0));
};
void TimeIntegrator::Synchronize() { Schedule::Synchronize(); }

void TimeIntegrator::NextStep() {
    Advance(m_time_step_);
    Schedule::NextStep();
}

Real TimeIntegrator::Advance(Real time_dt) {
    if (std::abs(time_dt) < std::numeric_limits<Real>::min()) { time_dt = m_time_step_; }
    time_dt = std::min(std::min(time_dt, m_time_step_), m_time_end_ - m_time_now_);
    m_time_now_ += time_dt;
    return m_time_now_;
};

//    if (level >= m_pimpl_->m_ctx_->GetAtlas().GetNumOfLevels()) { return m_pimpl_->m_time_; }
//    auto &atlas = m_pimpl_->m_ctx_->GetAtlas();
//    for (auto const &id : atlas.GetBlockList(level)) {
//        auto mblk = atlas.GetBlock(id);
//        for (auto &v : m_pimpl_->m_ctx_->GetAllDomains()) {
//            if (!v.second->GetGeoObject()->CheckOverlap(mblk->GetBoundBox())) { continue; }
//            auto res = m_pimpl_->m_ctx_->GetPatches()->GetTable(std::to_string(id));
//            if (res == nullptr) { res = std::make_shared<data::DataTable>(); }
//            v.second->ConvertPatchFromSAMRAI(mblk, res);
//            LOGGER << " Domain [ " << std::setw(10) << std::left << v.second->name() << " ] is applied on "
//                   << mblk->GetIndexBox() << " id= " << id << std::endl;
//            v.second->Run(dt);
//            auto t = v.second->PopPatch().second;
//            m_pimpl_->m_ctx_->GetPatches()->Push(std::to_string(id), t);
//        }
//    }
//    m_pimpl_->m_time_ += dt;
//    return m_pimpl_->m_time_;
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

}  // namespace engine{
}  // namespace simpla{