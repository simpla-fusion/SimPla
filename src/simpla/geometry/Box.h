//
// Created by salmon on 17-10-17.
//

#ifndef SIMPLA_BOX_H
#define SIMPLA_BOX_H

#include <simpla/SIMPLA_config.h>
#include "Shape.h"
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
    explicit Box(vector_type const &extents);

   public:
    static std::shared_ptr<Box> New(std::initializer_list<std::initializer_list<Real>> const &box) {
        return std::shared_ptr<Box>(new Box(box));
    }

    //    point_type xyz(Real u, Real v, Real w) const override;
    //    point_type uvw(Real x, Real y, Real z) const override;
    vector_type const &GetExtents() const { return m_extents_; }
    void SetExtents(vector_type const &extents) { m_extents_ = extents; }

    box_type GetBoundingBox() const override;
    bool CheckIntersection(box_type const &, Real tolerance) const override;
    bool CheckIntersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

    //    std::shared_ptr<Point> GetIntersectionion(std::shared_ptr<const Point> const &g, Real tolerance) const
    //    override;
    //    std::shared_ptr<Curve> GetIntersectionion(std::shared_ptr<const Curve> const &g, Real tolerance) const
    //    override;
    //    std::shared_ptr<Surface> GetIntersectionion(std::shared_ptr<const Surface> const &g, Real tolerance) const
    //    override;
    //    std::shared_ptr<Body> GetIntersectionion(std::shared_ptr<const Body> const &g, Real tolerance) const override;

   protected:
    vector_type m_extents_{1, 1, 1};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BOX_H
