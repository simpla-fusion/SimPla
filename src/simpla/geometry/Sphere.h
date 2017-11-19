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
    explicit Sphere(Axis const &axis, Real r0, Real r1, Real a0, Real a1, Real h0, Real h1);
};
struct SphereSurface : public Face {
    SP_GEO_OBJECT_HEAD(Face, SphereSurface)
   protected:
    explicit SphereSurface(Axis const &axis, Real radius, Real a0, Real a1, Real h0, Real h1);
};

}  // namespace geometry {
}  // namespace simpla {
#endif  // SIMPLA_SPHERE_H
