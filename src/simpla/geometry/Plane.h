//
// Created by salmon on 17-10-20.
//

#ifndef SIMPLA_PLANE_H
#define SIMPLA_PLANE_H

#include <simpla/SIMPLA_config.h>
#include "GeoObject.h"
#include "Surface.h"

namespace simpla {
namespace geometry {
struct Plane : public Surface {
    SP_GEO_OBJECT_HEAD(Plane, Surface);

   protected:
    Plane() = default;
    Plane(Plane const &) = default;

    explicit Plane( Axis  const &axis) : Surface(axis) {
        SetParameterRange(std::make_tuple(GetMinParameter(), GetMaxParameter()));
    }

   public:
    ~Plane() override = default;

    point_type Value(Real u, Real v) const override { return m_axis_.Coordinates(u, v); };
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_PLANE_H
