//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_TOROIDAL_H
#define SIMPLA_TOROIDAL_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Constants.h>
#include <simpla/utilities/SPDefines.h>
#include <simpla/utilities/macro.h>

#include "PrimitiveShape.h"
#include "Surface.h"

namespace simpla {
namespace geometry {
struct Torus : public PrimitiveShape {
    SP_GEO_OBJECT_HEAD(Torus, PrimitiveShape)
   protected:
    explicit Torus(Real major_radius, Real minor_radius);
    explicit Torus(Axis const& axis, Real major_radius, Real minor_radius);

   public:
    point_type xyz(Real phi, Real theta, Real r) const override {
        Real R = (m_major_radius_ + r * m_minor_radius_ * std::cos(theta));
        return point_type{R * std::cos(phi), R * std::sin(phi), m_minor_radius_ * std::sin(theta)};
    };
    point_type xyz(Real phi, Real theta) const {
        Real R = (m_major_radius_ + m_minor_radius_ * std::cos(theta));
        return point_type{R * std::cos(phi), R * std::sin(phi), m_minor_radius_ * std::sin(theta)};
    };
    point_type uvw(Real x, Real y, Real z) const override { return point_type{SP_SNaN, SP_SNaN, SP_SNaN}; }

    //    Real Distance(point_type const &xyz) const override { return SP_SNaN; }
    //    bool TestBoxGetIntersection(point_type const &x_min, point_type const &x_max) const override { return false; }
    //    int LineGetIntersection(point_type const &p0, point_type const &p1, Real *u) const override { return false; }

    Real GetMajorRadius() const;
    Real GetMinorRadius() const;

   private:
    Real m_major_radius_ = 1;
    Real m_minor_radius_ = 0.1;

    static constexpr Real m_parameter_range_[2][3] = {{0, 0, 0}, {TWOPI, TWOPI, 1}};
    static constexpr Real m_value_range_[2][3] = {{-1, -1, -1}, {1, 1, 1}};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_TOROIDAL_H
