//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_SPHERE_H
#define SIMPLA_SPHERE_H

#include <simpla/utilities/Constants.h>
#include <simpla/utilities/macro.h>
#include "Body.h"
#include "GeoObject.h"
#include "simpla/SIMPLA_config.h"
namespace simpla {
namespace geometry {

struct Sphere : public Body {
    SP_GEO_OBJECT_HEAD(Sphere, Body)
    Sphere() = default;
    ~Sphere() override = default;

   protected:
    explicit Sphere(Axis const &axis, Real r0 = SP_SNaN, Real r1 = SP_SNaN, Real phi0 = SP_SNaN, Real phi1 = SP_SNaN,
                    Real theta0 = SP_SNaN, Real theta1 = SP_SNaN)
        : Body(axis) {
        auto min = GetMinParameter();
        auto max = GetMaxParameter();
        TRY_ASSIGN(min[0], r0);
        TRY_ASSIGN(max[0], r1);
        TRY_ASSIGN(min[1], phi0);
        TRY_ASSIGN(max[1], phi1);
        TRY_ASSIGN(min[2], theta0);
        TRY_ASSIGN(min[2], theta1);

        SetParameterRange(min, max);
    }
    Sphere(Real r) : Sphere(Axis{}, 0, r) {}

   public:
    bool CheckInside(point_type const &x, Real tolerance) const override { return true; }

    std::tuple<bool, bool, bool> IsClosed() const override { return std::make_tuple(false, true, true); };
    std::tuple<bool, bool, bool> IsPeriodic() const override { return std::make_tuple(false, true, true); };
    nTuple<Real, 3> GetPeriod() const override { return nTuple<Real, 3>{SP_INFINITY, TWOPI, PI}; };
    nTuple<Real, 3> GetMinParameter() const override { return nTuple<Real, 3>{0, 0, -PI / 2}; }
    nTuple<Real, 3> GetMaxParameter() const override { return nTuple<Real, 3>{SP_INFINITY, TWOPI, PI / 2}; }
    /**
     *
     * @param u R
     * @param v phi
     * @param w theta
     * @return
     */
    point_type Value(Real r, Real phi, Real theta) const override {
        Real cos_theta = std::cos(theta);
        return m_axis_.Coordinates(r * cos_theta * std::cos(phi), r * cos_theta * std::sin(phi), r * std::sin(theta));
        //        return m_axis_.o + r * std::cos(theta) * (std::cos(phi) * m_axis_.x + std::sin(phi) * m_axis_.y) +
        //               r * std::sin(theta) * m_axis_.z;
    };
    int CheckOverlap(box_type const &, Real tolerance) const override;
    int FindIntersection(std::shared_ptr<const Curve> const &, std::vector<Real> &, Real tolerance) const override;
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_SPHERE_H
