//
// Created by salmon on 17-7-22.
//

#ifndef SIMPLA_CYLINDRICAL_H
#define SIMPLA_CYLINDRICAL_H

#include <simpla/utilities/Constants.h>
#include <simpla/utilities/macro.h>
#include "simpla/SIMPLA_config.h"

#include "Body.h"
#include "GeoObject.h"
#include "Surface.h"
namespace simpla {
namespace geometry {

struct Cylindrical : public Body {
    SP_GEO_OBJECT_HEAD(Cylindrical, Body)

   protected:
    Cylindrical() = default;
    explicit Cylindrical(Axis const &axis, Real r0 = SP_SNaN, Real r1 = SP_SNaN, Real phi0 = SP_SNaN,
                         Real phi1 = SP_SNaN, Real z0 = SP_SNaN, Real z1 = SP_SNaN)
        : Body(axis) {
        auto min = GetMinParameter();
        auto max = GetMaxParameter();
        TRY_ASSIGN(min[0], r0);
        TRY_ASSIGN(max[0], r1);
        TRY_ASSIGN(min[1], phi0);
        TRY_ASSIGN(max[1], phi1);
        TRY_ASSIGN(min[2], z0);
        TRY_ASSIGN(min[2], z1);

        SetParameterRange(min, max);
    }

   public:
    ~Cylindrical() override = default;

    std::tuple<bool, bool, bool> IsClosed() const override { return std::make_tuple(false, true, false); };
    std::tuple<bool, bool, bool> IsPeriodic() const override { return std::make_tuple(false, true, false); };
    nTuple<Real, 3> GetPeriod() const override { return nTuple<Real, 3>{SP_INFINITY, TWOPI, SP_INFINITY}; };
    nTuple<Real, 3> GetMinParameter() const override { return nTuple<Real, 3>{0, 0, -SP_INFINITY}; }
    nTuple<Real, 3> GetMaxParameter() const override { return nTuple<Real, 3>{SP_INFINITY, TWOPI, -SP_INFINITY}; }
    /**
     *
     * @param u R
     * @param v phi
     * @param w Z
     * @return
     */
    point_type Value(Real u, Real v, Real w) const override {
        return m_axis_.Coordinates(u * std::cos(v), u * std::sin(v), w);
    };

    bool TestIntersection(box_type const &) const override;
    bool TestInside(point_type const &x) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;
};

struct CylindricalSurface : public Surface {
    SP_GEO_OBJECT_HEAD(CylindricalSurface, Surface);

   protected:
    CylindricalSurface() = default;
    CylindricalSurface(CylindricalSurface const &other) = default;  // : Surface(other), m_radius_(other.m_radius_) {}
    CylindricalSurface(Axis const &axis, Real R, Real phi0 = SP_SNaN, Real phi1 = SP_SNaN, Real z0 = SP_SNaN,
                       Real z1 = SP_SNaN)
        : Surface(axis), m_radius_(R) {
        auto min = GetMinParameter();
        auto max = GetMaxParameter();

        TRY_ASSIGN(min[0], phi0);
        TRY_ASSIGN(max[0], phi1);
        TRY_ASSIGN(min[1], z0);
        TRY_ASSIGN(min[1], z1);

        SetParameterRange(min, max);
    }

   public:
    ~CylindricalSurface() override = default;

    std::tuple<bool, bool> IsClosed() const override { return std::make_tuple(true, false); };
    std::tuple<bool, bool> IsPeriodic() const override { return std::make_tuple(true, false); };
    nTuple<Real, 2> GetPeriod() const override { return nTuple<Real, 2>{TWOPI, SP_INFINITY}; };
    nTuple<Real, 2> GetMinParameter() const override { return nTuple<Real, 2>{0, -SP_INFINITY}; }
    nTuple<Real, 2> GetMaxParameter() const override { return nTuple<Real, 2>{TWOPI, SP_INFINITY}; }

    void SetRadius(Real r) { m_radius_ = r; }
    Real GetRadius() const { return m_radius_; }

    /**
     *
     * @param u  \phi
     * @param v  z
     * @return
     */
    point_type Value(Real u, Real v) const override {
        return m_axis_.Coordinates(m_radius_ * std::cos(u), m_radius_ * std::sin(u), v);
    };
    bool TestIntersection(box_type const &) const override;
    bool TestInside(point_type const &x) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   private:
    Real m_radius_ = 1.0;
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CYLINDRICAL_H
