//
// Created by salmon on 17-11-19.
//

#ifndef SIMPLA_SPHERE_H
#define SIMPLA_SPHERE_H
#include "Face.h"
#include "Solid.h"
namespace simpla {
namespace geometry {

struct Sphere : public Solid {
    SP_GEO_OBJECT_HEAD(Solid, Sphere)
   protected:
    explicit Sphere(Axis const &axis, box_type const &range);
};
struct SphereSurface : public Face {
    SP_GEO_OBJECT_HEAD(Face, SphereSurface)
   protected:
    explicit SphereSurface(Axis const &axis, Real radius, std::tuple<point2d_type, point2d_type> const &b);
};

}  // namespace geometry {
}  // namespace simpla {
#endif  // SIMPLA_SPHERE_H
