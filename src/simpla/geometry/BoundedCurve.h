//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_BOUNDEDCURVE_H
#define SIMPLA_BOUNDEDCURVE_H

#include "Curve.h"
namespace simpla {
namespace geometry {

struct BoundedCurve {
    BoundedCurve() = default;
    BoundedCurve(const BoundedCurve& other) = default;
    ~BoundedCurve() = default;

    BoundedCurve(Real start, Real end) : m_start_u_(start), m_end_u_(end) {}

    virtual point_type Value(Real u) const = 0;

    point_type StartPoint() const { return Value(GetStartParameter()); }
    point_type EndPoint() const { return Value(GetEndParameter()); }

    void SetStartParameter(Real u) { m_start_u_ = u; };
    Real SetEndParameter(Real u) { m_end_u_ = u; };
    Real GetStartParameter() const { return m_start_u_; }
    Real GetEndParameter() const { return m_end_u_; }

   private:
    Real m_start_u_ = 0;
    Real m_end_u_ = 1;
};

template <typename TBase>
struct BoundedCurveT : public TBase, public BoundedCurve {
    SP_GEO_OBJECT_HEAD(BoundedCurveT, TBase);

    BoundedCurveT() = default;
    BoundedCurveT(BoundedCurveT const&) = default;
    BoundedCurveT(TBase const& base, Real start, Real end) : TBase(base), BoundedCurve(start, end) {}
};
}  //    namespace geometry{

}  // namespace simpla{

#endif  // SIMPLA_BOUNDEDCURVE_H
