//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_CURVESEGMENT_H
#define SIMPLA_CURVESEGMENT_H

#include "Curve.h"
namespace simpla {
namespace geometry {

template <typename TBaseCurve>
struct CurveSegment : public TBaseCurve {
    SP_GEO_OBJECT_HEAD(CurveSegment, TBaseCurve);

    CurveSegment() : base_type(), m_start_u_(base_type::MinParameter()), m_end_u_(base_type::MaxParameter()){};
    CurveSegment(CurveSegment const&) = default;
    ~CurveSegment() override = default;

    //    CurveSegment(Real start, Real end, base_type const& base) : base_type(base), m_start_u_(start), m_end_u_(end)
    //    {}
    template <typename... Args>
    CurveSegment(Real start, Real end, Args&&... args)
        : base_type(std::forward<Args>(args)...), m_start_u_(start), m_end_u_(end) {}

    void SetStartParameter(Real u) { m_start_u_ = u; };
    void SetEndParameter(Real u) { m_end_u_ = u; };
    Real GetStartParameter() const { return m_start_u_; }
    Real GetEndParameter() const { return m_end_u_; }
    point_type GetStartPoint() const { return base_type::Value(GetStartParameter()); }
    point_type GetEndPoint() const { return base_type::Value(GetEndParameter()); }

   private:
    Real m_start_u_ = 0;
    Real m_end_u_ = 1;
};
template <typename>
void CurveSegment<TBaseCurve>::Deserialize(std::shared_ptr<simpla::data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
    m_start_u_ = res->GetValue<Real>("StartU", m_start_u_);
    m_end_u_ = res->GetValue<Real>("EndU", m_end_u_);
};
template <typename>
std::shared_ptr<simpla::data::DataNode> CurveSegment<TBaseCurve>::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("StartU", m_start_u_);
    res->SetValue<Real>("EndU", m_end_u_);
    return res;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_CURVESEGMENT_H
