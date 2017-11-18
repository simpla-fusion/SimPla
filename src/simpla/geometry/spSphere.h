//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_SPHERE_H
#define SIMPLA_SPHERE_H

#include <simpla/SIMPLA_config.h>
#include "Shape.h"
namespace simpla {
namespace geometry {

struct spSphere : public Shape {
    SP_SHAPE_HEAD(Shape, spSphere, Sphere)

    explicit spSphere(Real radius);
    Real GetRadius() const { return m_radius_; }
    void SetRadius(Real const &r) { m_radius_ = r; }

    point_type xyz(Real r, Real phi, Real theta) const {
        Real cos_theta = std::cos(theta);
        return point_type{r * cos_theta * std::cos(phi), r * cos_theta * std::sin(phi), r * std::sin(theta)};
    };
    point_type uvw(Real x, Real y, Real z) const { return point_type{x, y, z}; };

   private:
    Real m_radius_ = 1;
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_SPHERE_H
