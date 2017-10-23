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

    BoundedSurface() : base_type(), m_min_u_(base_type::MinParameter()), m_max_u_(base_type::MaxParameter()){};
    BoundedSurface(BoundedSurface const&) = default;
    ~BoundedSurface() override = default;

    //    BoundedSurface(Real start, Real end, base_type const& base) : base_type(base), m_start_u_(start),
    //    m_end_u_(end)
    //    {}
    template <typename... Args>
    BoundedSurface(Real min, Real max, Args&&... args)
        : base_type(std::forward<Args>(args)...), m_start_u_(start), m_end_u_(end) {}

    void SetMinParameter(Real u) { m_min_u_ = u; };
    void SetMaxParameter(Real u) { m_max_u_ = u; };
    Real GetMinParameter() const { return m_min_u_; }
    Real GetMaxParameter() const { return m_max_u_; }
    point_type GetStartPoint() const { return base_type::Value(GetStartParameter()); }
    point_type GetEndPoint() const { return base_type::Value(GetEndParameter()); }

   private:
    Real m_min_u_ = 0;
    Real m_max_u_ = 1;
};
template <typename>
void BoundedSurface<TBaseSurface>::Deserialize(std::shared_ptr<simpla::data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    m_min_u_ = cfg->GetValue<Real>("MinParameter", m_min_u_);
    m_max_u_ = cfg->GetValue<Real>("MaxParameter", m_max_u_);
};
template <typename>
std::shared_ptr<simpla::data::DataNode> BoundedSurface<TBaseSurface>::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("MinParameter", m_min_u_);
    res->SetValue<Real>("MaxParameter", m_max_u_);
    return res;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BOUNDEDSURFACE_H
