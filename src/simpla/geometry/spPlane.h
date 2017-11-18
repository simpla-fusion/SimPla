//
// Created by salmon on 17-10-20.
//

#ifndef SIMPLA_PLANE_H
#define SIMPLA_PLANE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/data/Serializable.h>
#include "GeoObject.h"
#include "Shape.h"
namespace simpla {
namespace geometry {
struct spPlane : public Shape {
    SP_SHAPE_HEAD(Shape, spPlane, Plane)

   public:
    //    point_type xyz(Real u, Real v, Real w) const override { return point_type{u, v, w}; }
    //    point_type uvw(Real x, Real y, Real z) const override { return point_type{x, y, z}; }
    //    point_type GetOrigin() const { return point_type{0, 0, 0}; }
    //    vector_type GetNormal() const { return point_type{0, 0, 1}; }
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_PLANE_H
