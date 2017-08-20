//
// Created by salmon on 17-3-17.
//

#include "simpla/SIMPLA_config.h"

#include "TimeIntegrator.h"

#include "simpla/data/Data.h"

#include "Context.h"
#include "TimeIntegrator.h"

namespace simpla {
namespace engine {

TimeIntegrator::TimeIntegrator() = default;
TimeIntegrator::~TimeIntegrator() = default;

void TimeIntegrator::Serialize(std::shared_ptr<data::DataNode> const &cfg) const {
    base_type::Serialize(cfg);
    auto tdb = std::dynamic_pointer_cast<data::DataTable>(cfg);
    if (tdb != nullptr) {
        tdb->SetValue("Name", GetName());
        tdb->SetValue("TimeBegin", GetTimeNow());
        tdb->SetValue("TimeEnd", GetTimeEnd());
        tdb->SetValue("TimeStep", GetTimeStep());
        tdb->SetValue("MaxStep", GetMaxStep());
    }
}

void TimeIntegrator::Deserialize(std::shared_ptr<const data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    auto tdb = std::dynamic_pointer_cast<const data::DataTable>(cfg);
    if (tdb != nullptr) {
        SetTimeNow(tdb->GetValue("TimeBegin", 0.0));
        SetTimeEnd(tdb->GetValue("TimeEnd", 1.0));
        SetTimeStep(tdb->GetValue("TimeStep", 0.5));
        SetMaxStep(tdb->GetValue<size_type>("MaxStep", 0UL));
    }
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

//    if (level >= m_pack_->m_ctx_->GetAtlas().GetNumOfLevels()) { return m_pack_->m_time_; }
//    auto &atlas = m_pack_->m_ctx_->GetAtlas();
//    for (auto const &id : atlas.GetBlockList(level)) {
//        auto mblk = atlas.GetMeshBlock(id);
//        for (auto &v : m_pack_->m_ctx_->GetAllDomains()) {
//            if (!v.second->GetGeoObject()->CheckOverlap(mblk->BoundingBox())) { continue; }
//            auto res = m_pack_->m_ctx_->GetPatches()->GetTable(std::to_string(id));
//            if (res == nullptr) { res = std::make_shared<data::DataTable>(); }
//            v.second->GetPatch(mblk, res);
//            LOGGER << " DomainBase [ " << std::setw(10) << std::left << v.second->name() << " ] is applied on "
//                   << mblk->IndexBox() << " id= " << id << std::endl;
//            v.second->Run(dt);
//            auto t = v.second->Serialize().second;
//            m_pack_->m_ctx_->GetPatches()->Deserialize(std::to_string(id), t);
//        }
//    }
//    m_pack_->m_time_ += dt;
//    return m_pack_->m_time_;
//    for (auto const &item : atlas.GetLayer(level)) {
//        for (auto &v : m_pack_->m_domains_) {
//            auto b_box = v.second->GetBaseMesh()->inner_bound_box();
//            if (!geometry::check_overlap(item.second->GetBox(), b_box)) { continue; }
//            v.second->Dispatch(m_pack_->m_patches_[item.first]);
//            v.second->Run(dt);
//        }
//    }
//    for (int i = 0; i < m_pack_->m_refine_ratio_; ++i) { Run(dt / m_pack_->m_refine_ratio_, level + 1); }
//    for (auto const &item : atlas.GetLayer(level)) {
//        for (auto &v : m_pack_->m_domains_) {
//            auto b_box = v.second->GetBaseMesh()->GetGeoObject()->BoundingBox();
//            if (!geometry::check_overlap(item.second->GetBox(), b_box)) { continue; }
//            v.second->Dispatch(m_pack_->m_patches_[item.first]);
//            v.second->Run(dt);
//        }
//    }

}  // namespace engine{
}  // namespace simpla{