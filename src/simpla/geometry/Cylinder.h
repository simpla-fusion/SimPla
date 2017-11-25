//
// Created by salmon on 17-11-19.
//

#ifndef SIMPLA_CYLINDER_H
#define SIMPLA_CYLINDER_H

#include "Face.h"
#include "Solid.h"
namespace simpla {
namespace geometry {

struct Cylinder : public Solid {
    SP_GEO_OBJECT_HEAD(Solid, Cylinder)
    explicit Cylinder(Axis const &axis, box_type const &);
};
struct CylinderSurface : public Face {
    SP_GEO_OBJECT_HEAD(Face, CylinderSurface)
    explicit CylinderSurface(Axis const &axis, Real radius, std::tuple<point2d_type, point2d_type> const &b);
};

}  // namespace geometry {
}  // namespace simpla {
#endif  // SIMPLA_CYLINDER_H
