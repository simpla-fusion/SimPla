//
// Created by salmon on 17-3-17.
//
#include "TimeIntegrator.h"
#include <simpla/concept/Configurable.h>
#include <simpla/data/all.h>
#include "Manager.h"
namespace simpla {
namespace engine {
struct TimeIntegrator::pimpl_s {
    Manager* m_manager_;
    Real m_time_;
};

TimeIntegrator::TimeIntegrator(Manager* m, std::shared_ptr<data::DataEntity> const& t)
    : concept::Configurable(t), m_pimpl_(new pimpl_s) {
    m_pimpl_->m_manager_ = m;
}
TimeIntegrator::~TimeIntegrator() {}

size_type TimeIntegrator::NextTimeStep(Real dt) { return 0; };
size_type TimeIntegrator::step() const { return 0; };
bool TimeIntegrator::remainingSteps() const { return 0; };
Real TimeIntegrator::timeNow() const { return 0.0; }

Real TimeIntegrator::Advance(Real dt, int level) {
    if (level >= m_pimpl_->m_manager_->GetAtlas().GetNumOfLevels()) { return m_pimpl_->m_time_; }
    auto& atlas = m_pimpl_->m_manager_->GetAtlas();
    for (auto const& id : atlas.GetBlockList(level)) {
        auto mblk = atlas.GetBlock(id);
        for (auto& v : m_pimpl_->m_manager_->GetAllDomainViews()) {
            if (!v.second->GetMesh()->GetGeoObject()->CheckOverlap(mblk->GetBoundBox())) { continue; }
            auto res = m_pimpl_->m_manager_->GetPatches()->Get(std::to_string(id));
            if (res == nullptr) { res = std::make_shared<data::DataTable>(); }
            v.second->PushData(mblk, res);
            LOGGER << " DomainView [ " << std::setw(10) << std::left << v.second->name() << " ] is applied on "
                   << mblk->GetIndexBox() << " id= " << id << std::endl;
            v.second->Run(dt);
            auto t = v.second->PopData().second;
            m_pimpl_->m_manager_->GetPatches()->Set(std::to_string(id), t);
        }
    }
    m_pimpl_->m_time_ += dt;

    //    for (auto const &item : atlas.GetLayer(level)) {
    //        for (auto &v : m_pimpl_->m_views_) {
    //            auto b_box = v.second->GetMesh()->inner_bound_box();
    //            if (!geometry::check_overlap(item.second->GetBox(), b_box)) { continue; }
    //            v.second->Dispatch(m_pimpl_->m_patches_[item.first]);
    //            v.second->Run(dt);
    //        }
    //    }
    //    for (int i = 0; i < m_pimpl_->m_refine_ratio_; ++i) { Run(dt / m_pimpl_->m_refine_ratio_, level + 1); }
    //    for (auto const &item : atlas.GetLayer(level)) {
    //        for (auto &v : m_pimpl_->m_views_) {
    //            auto b_box = v.second->GetMesh()->GetGeoObject()->GetBoundBox();
    //            if (!geometry::check_overlap(item.second->GetBox(), b_box)) { continue; }
    //            v.second->Dispatch(m_pimpl_->m_patches_[item.first]);
    //            v.second->Run(dt);
    //        }
    //    }
}

}  // namespace engine{
}  // namespace simpla{