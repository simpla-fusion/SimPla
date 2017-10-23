//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_BOUNDEDSURFACE_H
#define SIMPLA_BOUNDEDSURFACE_H

#include "Surface.h"
namespace simpla {
namespace geometry {

template <typename TBaseSurface>
struct BoundedSurface : public TBaseSurface {
    SP_GEO_OBJECT_HEAD(BoundedSurface, TBaseSurface);

    BoundedSurface() : base_type(), m_min_(base_type::GetMinParameter()), m_max_(base_type::GetMaxParameter()){};
    BoundedSurface(BoundedSurface const&) = default;
    ~BoundedSurface() override = default;

    template <typename... Args>
    BoundedSurface(nTuple<Real, 2> min, nTuple<Real, 2> max, Args&&... args)
        : base_type(std::forward<Args>(args)...), m_min_(std::move(min)), m_max_(std::move(max)) {}

    void SetMinParameter(nTuple<Real, 2> const& min) { m_min_ = min; };
    void SetMaxParameter(nTuple<Real, 2> const& max) { m_max_ = max; };
    nTuple<Real, 2> GetMinParameter() const { return m_min_; }
    nTuple<Real, 2> GetMaxParameter() const { return m_max_; }
    point_type GetStartPoint() const { return base_type::Value(GetMinParameter()); }
    point_type GetEndPoint() const { return base_type::Value(GetMaxParameter()); }

   private:
    nTuple<Real, 2> m_min_{0, 0};
    nTuple<Real, 2> m_max_{1, 1};
};
template <typename>
void BoundedSurface<TBaseSurface>::Deserialize(std::shared_ptr<simpla::data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    m_min_ = cfg->GetValue<Real>("MinParameter", m_min_);
    m_max_ = cfg->GetValue<Real>("MaxParameter", m_max_);
};
template <typename>
std::shared_ptr<simpla::data::DataNode> BoundedSurface<TBaseSurface>::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("MinParameter", m_min_);
    res->SetValue<Real>("MaxParameter", m_max_);
    return res;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BOUNDEDSURFACE_H
