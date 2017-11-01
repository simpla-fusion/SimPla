//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_TOROIDAL_H
#define SIMPLA_TOROIDAL_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Constants.h>
#include <simpla/utilities/SPDefines.h>
#include <simpla/utilities/macro.h>

#include "GeoObject.h"
#include "ParametricBody.h"
#include "ParametricSurface.h"
#include "ShapeFunction.h"
#include "Surface.h"

namespace simpla {
namespace geometry {
struct sfToroidal : public ShapeFunction {
    explicit sfToroidal(Real major_radius = 10, Real minor_radius = 1.0)
        : m_major_radius_(major_radius), m_minor_radius_(minor_radius) {
        //        m_min_value_[0] = -major_radius - m_minor_radius_;
        //        m_max_value_[0] = major_radius + m_minor_radius_;
        //        m_min_value_[1] = -major_radius - m_minor_radius_;
        //        m_max_value_[1] = major_radius + m_minor_radius_;
        //        m_min_value_[2] = -m_minor_radius_;
        //        m_max_value_[2] = m_minor_radius_;
    }
    sfToroidal(sfToroidal const &) = default;
    ~sfToroidal() = default;
    void swap(sfToroidal &other){
        //        std::swap(m_minor_radius_, other.m_minor_radius_);
        //        std::swap(m_major_radius_, other.m_major_radius_);
        //        std::swap(m_min_parameter_, other.m_min_parameter_);
        //        std::swap(m_max_parameter_, other.m_max_parameter_);
        //        std::swap(m_min_value_, other.m_min_value_);
        //        std::swap(m_max_value_, other.m_max_value_);
    };
    int GetDimension() const override { return 3; }

    point_type Value(Real phi, Real theta, Real r) const override {
        Real R = (m_major_radius_ + r * m_minor_radius_ * std::cos(theta));
        return point_type{R * std::cos(phi), R * std::sin(phi), m_minor_radius_ * std::sin(theta)};
    };
    point_type Value(Real phi, Real theta) const {
        Real R = (m_major_radius_ + m_minor_radius_ * std::cos(theta));
        return point_type{R * std::cos(phi), R * std::sin(phi), m_minor_radius_ * std::sin(theta)};
    };
    point_type InvValue(point_type const &xyz) const override { return point_type{SP_SNaN, SP_SNaN, SP_SNaN}; }
    Real Distance(point_type const &xyz) const override { return SP_SNaN; }
    bool TestBoxGetIntersectionion(point_type const &x_min, point_type const &x_max) const override { return false; }
    int LineGetIntersectionion(point_type const &p0, point_type const &p1, Real *u) const override { return false; }

   private:
    Real m_major_radius_ = 1;
    Real m_minor_radius_ = 0.1;

    static constexpr Real m_parameter_range_[2][3] = {{0, 0, 0}, {TWOPI, TWOPI, 1}};
    static constexpr Real m_value_range_[2][3] = {{-1, -1, -1}, {1, 1, 1}};
};
struct Toroidal : public ParametricBody {
    SP_GEO_OBJECT_HEAD(Toroidal, ParametricBody)

   protected:
    Toroidal();
    Toroidal(Toroidal const &);
    explicit Toroidal(Axis const &axis);

   public:
    ~Toroidal() override;

    point_type xyz(Real phi, Real theta, Real r) const override { return m_axis_.xyz(m_shape_.Value(phi, theta, r)); };
    point_type uvw(Real x, Real y, Real z) const override { return m_shape_.InvValue(m_axis_.uvw(x, y, z)); };

    //    bool CheckIntersection(box_type const &, Real tolerance) const override;
    //    std::shared_ptr<Curve> GetIntersection(std::shared_ptr<const Curve> const &g, Real tolerance) const override;
    //    std::shared_ptr<Surface> GetIntersection(std::shared_ptr<const Surface> const &g, Real tolerance) const override;
    //    std::shared_ptr<Body> GetIntersection(std::shared_ptr<const Body> const &g, Real tolerance) const override;
    //    using base_type::GetIntersection;

   private:
    sfToroidal m_shape_;
};
struct ToroidalSurface : public ParametricSurface {
    SP_GEO_OBJECT_HEAD(ToroidalSurface, ParametricSurface);

   protected:
    ToroidalSurface() = default;
    ToroidalSurface(ToroidalSurface const &) = default;

    explicit ToroidalSurface(Axis const &axis, Real major_radius, Real minor_radius)
        : ParametricSurface(axis), m_shape_(minor_radius / major_radius) {}

   public:
    ~ToroidalSurface() override = default;
    bool IsClosed() const override { return true; };
    bool IsSimpleConnected() const override { return false; }
    point_type xyz(Real phi, Real theta) const override { return m_axis_.xyz(m_shape_.Value(phi, theta)); };
    point_type uvw(Real x, Real y, Real z) const override { return m_shape_.InvValue(m_axis_.uvw(x, y, z)); };
    //    bool CheckIntersection(box_type const &b, Real tolerance) const override;
    //    bool CheckIntersection(point_type const &x, Real tolerance) const override;

   protected:
    sfToroidal m_shape_;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_TOROIDAL_H
