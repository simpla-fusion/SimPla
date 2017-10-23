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

    BoundedCurve() : base_type(), m_min_(base_type::GetMinParameter()), m_max_(base_type::GetMaxParameter()){};
    BoundedCurve(BoundedCurve const&) = default;
    ~BoundedCurve() override = default;

    template <typename... Args>
    BoundedCurve(Real min, Real max, Args&&... args)
        : base_type(std::forward<Args>(args)...), m_min_(min), m_max_(max) {}

    void SetMinParameter(Real u) { m_min_ = u; };
    void SetMaxParameter(Real u) { m_max_ = u; };
    Real GetMinParameter() const override { return m_min_; }
    Real GetMaxParameter() const override { return m_max_; }

    point_type GetStartPoint() const { return TBaseCurve::Value(GetMinParameter()); }
    point_type GetEndPoint() const { return TBaseCurve::Value(GetMaxParameter()); }

   private:
    Real m_min_ = 0;
    Real m_max_ = 1;
};
template <typename TBaseCurve>
void BoundedCurve<TBaseCurve>::Deserialize(std::shared_ptr<simpla::data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    m_min_ = cfg->GetValue("GetMinParameter", m_min_);
    m_max_ = cfg->GetValue("GetMaxParameter", m_max_);
};
template <typename TBaseCurve>
std::shared_ptr<simpla::data::DataNode> BoundedCurve<TBaseCurve>::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("GetMinParameter", m_min_);
    res->SetValue("GetMaxParameter", m_max_);
    return res;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_CURVESEGMENT_H
