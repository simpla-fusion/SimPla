//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_TOROIDAL_H
#define SIMPLA_TOROIDAL_H

#include <simpla/utilities/Constants.h>
#include <simpla/utilities/macro.h>
#include "simpla/SIMPLA_config.h"

#include "Body.h"
#include "GeoObject.h"
#include "Surface.h"
namespace simpla {
namespace geometry {

struct Toroidal : public Body {
    SP_GEO_OBJECT_HEAD(Toroidal, Body)

   protected:
    Toroidal() = default;

    explicit Toroidal(Real major_radius, Real r0 = 0, Real r1 = 1, Real phi0 = SP_SNaN, Real phi1 = SP_SNaN,
                      Real theta0 = SP_SNaN, Real theta1 = SP_SNaN)
        : m_major_radius_(major_radius) {
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
    template <typename... Args>
    explicit Toroidal(Axis const &axis, Args &&... args) : Toroidal(std::forward<Args>(args)...) {
        SetAxis(axis);
    }

   public:
    ~Toroidal() override = default;

    std::tuple<bool, bool, bool> IsClosed() const override { return std::make_tuple(false, true, false); };
    std::tuple<bool, bool, bool> IsPeriodic() const override { return std::make_tuple(false, true, false); };
    nTuple<Real, 3> GetPeriod() const override { return nTuple<Real, 3>{SP_INFINITY, TWOPI, SP_INFINITY}; };
    nTuple<Real, 3> GetMinParameter() const override { return nTuple<Real, 3>{0, 0, -SP_INFINITY}; }
    nTuple<Real, 3> GetMaxParameter() const override { return nTuple<Real, 3>{SP_INFINITY, TWOPI, -SP_INFINITY}; }
    /**
     *
     * @param phi R
     * @param theta phi
     * @param w Z
     * @return
     */
    point_type Value(Real phi, Real theta, Real r) const override {
        r = (m_major_radius_ + r * std::cos(theta));
        return m_axis_.Coordinates(r * std::cos(phi), r * std::sin(phi), r * std::sin(theta));
    };

    bool TestIntersection(box_type const &) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   private:
    Real m_major_radius_ = 1.0;
};
struct ToroidalSurface : public Surface {
    SP_GEO_OBJECT_HEAD(ToroidalSurface, Surface);

   protected:
    ToroidalSurface() = default;
    ToroidalSurface(ToroidalSurface const &) = default;
    ToroidalSurface(Axis const &axis, Real major_radius, Real minor_radius, Real phi0 = SP_SNaN, Real phi1 = SP_SNaN,
                    Real theta0 = SP_SNaN, Real theta1 = SP_SNaN)
        : Surface(axis), m_major_radius_(major_radius), m_minor_radius_(minor_radius) {
        auto min = GetMinParameter();
        auto max = GetMaxParameter();

        TRY_ASSIGN(min[0], phi0);
        TRY_ASSIGN(max[0], phi1);
        TRY_ASSIGN(min[1], theta0);
        TRY_ASSIGN(min[1], theta1);

        SetParameterRange(min, max);
    }

   public:
    ~ToroidalSurface() override = default;

    std::tuple<bool, bool> IsClosed() const override { return std::make_tuple(true, true); };
    std::tuple<bool, bool> IsPeriodic() const override { return std::make_tuple(true, true); };
    nTuple<Real, 2> GetPeriod() const override { return nTuple<Real, 2>{TWOPI, TWOPI}; };
    nTuple<Real, 2> GetMinParameter() const override { return nTuple<Real, 2>{0, 0}; }
    nTuple<Real, 2> GetMaxParameter() const override { return nTuple<Real, 2>{TWOPI, TWOPI}; }

    void GetMajorRadius(Real r) { m_major_radius_ = r; }
    void GetMinorRadius(Real r) { m_minor_radius_ = r; }
    Real GetMajorRadius() const { return m_major_radius_; }
    Real GetMinorRadius() const { return m_minor_radius_; }

    point_type Value(Real u, Real v) const override {
        Real r = (m_major_radius_ + m_minor_radius_ * std::cos(v));
        return m_axis_.Coordinates(r * std::cos(u), r * std::sin(u), m_minor_radius_ * std::sin(v));
        //        return m_axis_.o +
        //               (m_major_radius_ + m_minor_radius_ * std::cos(v)) * (std::cos(u) * m_axis_.x + std::sin(u) *
        //               m_axis_.y) +
        //               m_minor_radius_ * std::sin(v) * m_axis_.z;
    };
    bool TestIntersection(box_type const &) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   protected:
    Real m_major_radius_ = 1;
    Real m_minor_radius_ = 1;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_TOROIDAL_H
