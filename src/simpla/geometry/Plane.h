//
// Created by salmon on 17-10-20.
//

#ifndef SIMPLA_PLANE_H
#define SIMPLA_PLANE_H

#include <simpla/SIMPLA_config.h>
#include "GeoObject.h"
#include "ShapeFunction.h"
#include "Surface.h"
namespace simpla {
namespace geometry {
struct Plane : public GeoObject {
    SP_GEO_OBJECT_HEAD(Plane, GeoObject)
   protected:
    Plane(point_type const &o, vector_type const &x, vector_type const &y);

   public:
    point_type xyz(Real u, Real v, Real w) const { return m_axis_.xyz(u, v, w); }
    point_type uvw(Real x, Real y, Real z) const { return m_axis_.uvw(x, y, z); }
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_PLANE_H
