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

   protected:
   public:
    point_type xyz(Real r, Real phi, Real theta) const override {
        Real cos_theta = std::cos(theta);
        return point_type{r * cos_theta * std::cos(phi), r * cos_theta * std::sin(phi), r * std::sin(theta)};
    };
    point_type uvw(Real x, Real y, Real z) const override { return point_type{x, y, z}; };

    box_type GetBoundingBox(box_type const &uvw_box) const override {
        Real r = std::get<1>(uvw_box)[0];
        return box_type{{-r, -r, -r}, {r, r, r}};
    };
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_SPHERE_H
