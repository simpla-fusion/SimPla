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

    template <typename... Args>
    explicit Plane(Args &&... args) : Surface(std::forward<Args>(args)...) {}

   public:
    ~Plane() override = default;



    point_type Value(Real u, Real v) const override { return u * m_axis_.x + v * m_axis_.y; };
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_PLANE_H
