//
// Created by salmon on 17-10-17.
//

#ifndef SIMPLA_BOX_H
#define SIMPLA_BOX_H

#include <simpla/SIMPLA_config.h>
#include "GeoEntity.h"
#include "Solid.h"
namespace simpla {
namespace geometry {

struct Box : public Solid {
    SP_GEO_OBJECT_HEAD(Solid, Box)

   protected:
    Box(std::initializer_list<std::initializer_list<Real>> const &v);
    explicit Box(point_type const &p0, point_type const &p1);
    explicit Box(point_type const &p0, Real u, Real v, Real w);
    explicit Box(box_type const &v);
    explicit Box(Axis const &axis, vector_type const &extents);

   public:
    static std::shared_ptr<Box> New(std::initializer_list<std::initializer_list<Real>> const &box) {
        return std::shared_ptr<Box>(new Box(box));
    }

    SP_PROPERTY(vector_type, Extents);
    //    point_type xyz(Real u, Real v, Real w) const override;
    //    point_type uvw(Real x, Real y, Real z) const override;
    //    box_type GetBoundingBox() const override;
    //    bool CheckIntersection(box_type const &, Real tolerance) const override;
    //    bool CheckIntersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;
    //    std::shared_ptr<Point> GetIntersectionion(std::shared_ptr<const Point> const &g, Real tolerance) const
    //    override;
    //    std::shared_ptr<gCurve> GetIntersectionion(std::shared_ptr<const gCurve> const &g, Real tolerance) const
    //    override;
    //    std::shared_ptr<gSurface> GetIntersectionion(std::shared_ptr<const gSurface> const &g, Real tolerance) const
    //    override;
    //    std::shared_ptr<Body> GetIntersectionion(std::shared_ptr<const Body> const &g, Real tolerance) const override;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BOX_H
