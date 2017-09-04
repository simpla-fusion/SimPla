//
// Created by salmon on 17-3-17.
//

#include "simpla/SIMPLA_config.h"

#include "TimeIntegrator.h"

#include "TimeIntegrator.h"
#include "simpla/data/Data.h"

namespace simpla {
namespace engine {

TimeIntegrator::TimeIntegrator() = default;
TimeIntegrator::~TimeIntegrator() = default;

std::shared_ptr<data::DataNode> TimeIntegrator::Serialize() const {
    auto tdb = base_type::Serialize();
    tdb->SetValue("Name", GetName());
    tdb->SetValue("TimeBegin", GetTimeNow());
    tdb->SetValue("TimeEnd", GetTimeEnd());
    tdb->SetValue("TimeStep", GetTimeStep());
    tdb->SetValue("MaxStep", GetMaxStep());
    return tdb;
}

void TimeIntegrator::Deserialize(std::shared_ptr<const data::DataNode>const & tdb) {
    base_type::Deserialize(tdb);
    if (tdb != nullptr) {
        SetTimeNow(tdb->GetValue("TimeBegin", 0.0));
        SetTimeEnd(tdb->GetValue("TimeEnd", 1.0));
        SetTimeStep(tdb->GetValue("TimeStep", 0.5));
        SetMaxStep(tdb->GetValue<size_type>("MaxStep", 0UL));
    }
};
void TimeIntegrator::Synchronize() { Schedule::Synchronize(); }

void TimeIntegrator::NextStep() {
    Advance(GetTimeStep());
    Schedule::NextStep();
}

Real TimeIntegrator::Advance(Real time_dt) {
    if (std::abs(time_dt) < std::numeric_limits<Real>::min()) { time_dt = GetTimeStep(); }
    time_dt = std::min(std::min(time_dt, GetTimeStep()), GetTimeEnd() - GetTimeNow());
    SetTimeNow(GetTimeNow() + time_dt);
    return GetTimeNow();
};

//    if (level >= m_pack_->m_ctx_->GetAtlas().GetNumOfLevels()) { return m_pack_->m_time_; }
//    auto &atlas = m_pack_->m_ctx_->GetAtlas();
//    for (auto const &id : atlas.GetBlockList(level)) {
//        auto mblk = atlas.GetMeshBlock(id);
//        for (auto &v : m_pack_->m_ctx_->GetAllDomains()) {
//            if (!v.m_node_->GetGeoObject()->CheckOverlap(mblk->BoundingBox())) { continue; }
//            auto res = m_pack_->m_ctx_->GetPatches()->GetTable(std::to_string(id));
//            if (res == nullptr) { res = std::make_shared<data::DataNode>(); }
//            v.m_node_->GetPatch(mblk, res);
//            LOGGER << " DomainBase [ " << std::setw(10) << std::left << v.m_node_->name() << " ] is applied on "
//                   << mblk->IndexBox() << " id= " << id << std::endl;
//            v.m_node_->Run(dt);
//            auto t = v.m_node_->Serialize().m_node_;
//            m_pack_->m_ctx_->GetPatches()->Deserialize(std::to_string(id), t);
//        }
//    }
//    m_pack_->m_time_ += dt;
//    return m_pack_->m_time_;
//    for (auto const &item : atlas.GetLayer(level)) {
//        for (auto &v : m_pack_->m_domains_) {
//            auto b_box = v.m_node_->GetBaseMesh()->inner_bound_box();
//            if (!geometry::check_overlap(item.m_node_->GetBox(), b_box)) { continue; }
//            v.m_node_->Dispatch(m_pack_->m_patches_[item.first]);
//            v.m_node_->Run(dt);
//        }
//    }
//    for (int i = 0; i < m_pack_->m_refine_ratio_; ++i) { Run(dt / m_pack_->m_refine_ratio_, level + 1); }
//    for (auto const &item : atlas.GetLayer(level)) {
//        for (auto &v : m_pack_->m_domains_) {
//            auto b_box = v.m_node_->GetBaseMesh()->GetGeoObject()->BoundingBox();
//            if (!geometry::check_overlap(item.m_node_->GetBox(), b_box)) { continue; }
//            v.m_node_->Dispatch(m_pack_->m_patches_[item.first]);
//            v.m_node_->Run(dt);
//        }
//    }

}  // namespace engine{
}  // namespace simpla{