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
    Plane();
    Plane(Plane const &);
    explicit Plane(Axis const &axis);
    Plane(point_type const &o, vector_type const &x, vector_type const &y);

   public:
    ~Plane() override;
    point_type Value(Real u, Real v) const override;
    std::shared_ptr<GeoObject> GetBoundary() const override;
    box_type GetBoundingBox() const override;
    bool TestIntersection(box_type const &) const override;
    bool TestInside(point_type const &x, Real tolerance) const override;
    bool TestInsideUV(Real u, Real v, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const override;
    point_type Value(point_type const &x) const override;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_PLANE_H
