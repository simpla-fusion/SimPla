//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_SPHERE_H
#define SIMPLA_SPHERE_H

#include <simpla/utilities/Constants.h>
#include "simpla/SIMPLA_config.h"

#include "Body.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Sphere : public Body {
    SP_GEO_OBJECT_HEAD(Sphere, Body)
    Sphere() = default;
    ~Sphere() override = default;

   protected:
    explicit Sphere(Axis const &axis) : Body(axis) {}

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
        return m_axis_.o + r * std::cos(theta) * (std::cos(phi) * m_axis_.x + std::sin(phi) * m_axis_.y) +
               r * std::sin(theta) * m_axis_.z;
    };
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_SPHERE_H
