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
   protected:
    explicit Cylinder(Axis const &axis, Real r0, Real r1, Real a0, Real a1, Real h0, Real h1);
    explicit Cylinder(Axis const &axis, box_type const &);
};
struct CylinderSurface : public Face {
    SP_GEO_OBJECT_HEAD(Face, CylinderSurface)
   protected:
    explicit CylinderSurface(Axis const &axis, Real radius, Real a0, Real a1, Real h0, Real h1);
};

}  // namespace geometry {
}  // namespace simpla {
#endif  // SIMPLA_CYLINDER_H
