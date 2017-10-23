//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_CURVESEGMENT_H
#define SIMPLA_CURVESEGMENT_H

#include "Curve.h"
namespace simpla {
namespace geometry {

template <typename TBaseCurve>
struct BoundedCurve : public TBaseCurve {
    SP_GEO_OBJECT_HEAD(BoundedCurve, TBaseCurve);

    BoundedCurve() : base_type(), m_min_u_(base_type::MinParameter()), m_max_u_(base_type::MaxParameter()){};
    BoundedCurve(BoundedCurve const&) = default;
    ~BoundedCurve() override = default;

    template <typename... Args>
    BoundedCurve(Real min, Real max, Args&&... args)
        : base_type(std::forward<Args>(args)...), m_min_u_(min), m_max_u_(max) {}

    void SetMinParameter(Real u) { m_min_u_ = u; };
    void SetMaxParameter(Real u) { m_max_u_ = u; };
    Real GetMinParameter() const override { return m_min_u_; }
    Real GetMaxParameter() const override { return m_max_u_; }

    point_type GetStartPoint() const { return Value(MinParameter()); }
    point_type GetEndPoint() const { return Value(MaxParameter()); }

   private:
    Real m_min_u_ = 0;
    Real m_max_u_ = 1;
};
template <typename>
void BoundedCurve<TBaseCurve>::Deserialize(std::shared_ptr<simpla::data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    m_min_u_ = cfg->GetValue<Real>("GetMinParameter", m_min_u_);
    m_max_u_ = cfg->GetValue<Real>("GetMaxParameter", m_max_u_);
};
template <typename>
std::shared_ptr<simpla::data::DataNode> BoundedCurve<TBaseCurve>::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("GetMinParameter", m_min_u_);
    res->SetValue<Real>("GetMaxParameter", m_max_u_);
    return res;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_CURVESEGMENT_H
