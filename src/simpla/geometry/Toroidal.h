//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_TOROIDAL_H
#define SIMPLA_TOROIDAL_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Constants.h>
#include <simpla/utilities/SPDefines.h>
#include <simpla/utilities/macro.h>

#include "Body.h"
#include "GeoObject.h"
#include "ShapeFunction.h"
#include "Surface.h"
namespace simpla {
namespace geometry {
struct sfToroidal : public ShapeFunction {
    sfToroidal(Real major_radius, Real minor_radius) : m_major_radius_(major_radius), m_minor_radius_(minor_radius) {
        m_min_value_[0] = -major_radius - m_minor_radius_;
        m_max_value_[0] = major_radius + m_minor_radius_;
        m_min_value_[1] = -major_radius - m_minor_radius_;
        m_max_value_[1] = major_radius + m_minor_radius_;
        m_min_value_[2] = -m_minor_radius_;
        m_max_value_[2] = m_minor_radius_;
    }
    sfToroidal(sfToroidal const &) = default;
    ~SfToroidal() override = default;
    void swap(sfToroidal &other) {
        std::swap(m_minor_radius_, other.m_minor_radius_);
        std::swap(m_major_radius_, other.m_major_radius_);
        std::swap(m_min_parameter_, other.m_min_parameter_);
        std::swap(m_max_parameter_, other.m_max_parameter_);
        std::swap(m_min_value_, other.m_min_value_);
        std::swap(m_max_value_, other.m_max_value_);
    };
    int GetDimension() const override { return 3; }
    Real GetMinParameter(int n) const override { return m_min_parameter_[n]; };
    Real GetMaxParameter(int n) const override { return m_max_parameter_[n]; };
    Real GetMaxValue(int n) const override { return SP_INFINITY; };
    Real GetMinValue(int n) const override { return -SP_INFINITY; };

    point_type Value(Real phi, Real theta, Real r) const override {
        Real R = (m_major_radius_ + r * m_minor_radius_ * std::cos(theta));
        return point_type{R * std::cos(phi), R * std::sin(phi), m_minor_radius_ * std::sin(theta)};
    };
    point_type InvValue(point_type const &xyz) const override { return point_type{SP_SNaN, SP_SNaN, SP_SNaN}; }
    Real Distance(point_type const &xyz) const override { return SP_SNaN; }
    bool TestBoxIntersection(point_type const &x_min, point_type const &x_max) const override { return false; }
    int LineIntersection(point_type const &p0, point_type const &p1, Real *u) const override { return false; }

   private:
    Real m_major_radius_ = 1;
    Real m_minor_radius_ = 0.1;
    nTuple<Real, 3> m_min_parameter_{0, 0, 0};
    nTuple<Real, 3> m_max_parameter_{TWOPI, TWOPI, SP_INFINITY};
    nTuple<Real, 3> m_min_value_{0, 0, 0};
    nTuple<Real, 3> m_max_value_{1, 1, 1};
};
struct Toroidal : public Body, public sfToroidal {
    SP_GEO_OBJECT_HEAD(Toroidal, Body)

   protected:
    Toroidal() = default;
    template <typename... Args>
    explicit Toroidal(Args &&... args) : sfToroidal(std::forward<Args>(args)...) {}
    template <typename... Args>
    explicit Toroidal(Axis const &axis, Args &&... args) : Body(Axis), sfToroidal(std::forward<Args>(args)...) {}

   public:
    ~Toroidal() override = default;

    /**
     *
     * @param phi R
     * @param theta phi
     * @param w Z
     * @return
     */
    point_type xyz(Real phi, Real theta, Real r) const override { return m_axis_.xyz(sfToroidal(phi, theta, r)); };

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
