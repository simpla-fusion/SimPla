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
    RevolutionSurface() = default;
    RevolutionSurface(RevolutionSurface const &) = default;
    ~RevolutionSurface() override = default;

    template <typename... Args>
    explicit RevolutionSurface(point_type const &origin, Args &&... args)
        : SweptSurface(std::forward<Args>(args)...), m_origin_(origin) {}

    std::tuple<bool, bool> IsClosed() const override { return std::make_tuple(true, GetBasisCurve()->IsClosed()); };
    std::tuple<bool, bool> IsPeriodic() const override { return std::make_tuple(true, GetBasisCurve()->IsPeriodic()); };
    nTuple<Real, 2> GetPeriod() const override { return nTuple<Real, 2>{TWOPI, GetBasisCurve()->GetPeriod()}; };
    nTuple<Real, 2> GetMinParameter() const override { return nTuple<Real, 2>{0, GetBasisCurve()->GetMinParameter()}; }
    nTuple<Real, 2> GetMaxParameter() const override {
        return nTuple<Real, 2>{TWOPI, GetBasisCurve()->GetMaxParameter()};
    }

    point_type Value(Real u, Real v) const override {
        vector_type P = GetBasisCurve()->Value(u) - m_origin_;
        return m_origin_ + (dot(P, m_direction_) * (1.0 - std::cos(v))) * m_direction_ +
               cross(P, m_direction_) * std::sin(v) + P * std::cos(v);
    };

    void SetOrigin(point_type const &p) { m_origin_ = p; }
    point_type const &GetOrigin() const { return m_origin_; }

   private:
    point_type m_origin_{0, 0, 0};
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_REVOLUTIONSURFACE_H
