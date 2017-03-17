//
// Created by salmon on 17-3-17.
//
#include "TimeIntegrator.h"
namespace simpla {
namespace engine {

Real TimeIntegrator::Advance(Real dt, int level) {
    auto &atlas = m_manger_->GetAtlas();

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
//            auto b_box = v.second->GetMesh()->GetGeoObject()->bound_box();
//            if (!geometry::check_overlap(item.second->GetBox(), b_box)) { continue; }
//            v.second->Dispatch(m_pimpl_->m_patches_[item.first]);
//            v.second->Run(dt);
//        }
//    }
}

}  // namespace engine{
}  // namespace simpla{