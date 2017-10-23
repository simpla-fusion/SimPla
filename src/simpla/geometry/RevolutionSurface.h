//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_REVOLUTIONSURFACE_H
#define SIMPLA_REVOLUTIONSURFACE_H
#include "SweptSurface.h"
namespace simpla {
namespace geometry {
struct RevolutionSurface : public SweptSurface {
    SP_GEO_OBJECT_HEAD(RevolutionSurface, SweptSurface);

   protected:
    RevolutionSurface() = default;
    RevolutionSurface(RevolutionSurface const &other) = default;  //: SweptSurface(other) {}
    RevolutionSurface(std::shared_ptr<Axis> const &axis, std::shared_ptr<Curve> const &c)
        : SweptSurface(c), m_r_axis_(axis) {
        SetParameterRange(GetMinParameter(), GetMaxParameter());
    }

   public:
    ~RevolutionSurface() override = default;

    std::tuple<bool, bool> IsClosed() const override { return std::make_tuple(true, GetBasisCurve()->IsClosed()); };
    std::tuple<bool, bool> IsPeriodic() const override { return std::make_tuple(true, GetBasisCurve()->IsPeriodic()); };
    nTuple<Real, 2> GetPeriod() const override { return nTuple<Real, 2>{TWOPI, GetBasisCurve()->GetPeriod()}; };
    nTuple<Real, 2> GetMinParameter() const override { return nTuple<Real, 2>{0, GetBasisCurve()->GetMinParameter()}; }
    nTuple<Real, 2> GetMaxParameter() const override {
        return nTuple<Real, 2>{TWOPI, GetBasisCurve()->GetMaxParameter()};
    }
    point_type Value(Real u, Real v) const override {
        vector_type P = GetBasisCurve()->Value(u) - m_r_axis_->o;

        return m_r_axis_->o + (dot(P, m_r_axis_->z) * (1.0 - std::cos(v))) * m_r_axis_->z +
               cross(P, m_r_axis_->x) * std::sin(v) + P * std::cos(v);
    };

   private:
    std::shared_ptr<Axis> m_r_axis_ = nullptr;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_REVOLUTIONSURFACE_H
