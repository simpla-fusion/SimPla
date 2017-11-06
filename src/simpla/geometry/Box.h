//
// Created by salmon on 17-10-17.
//

#ifndef SIMPLA_BOX_H
#define SIMPLA_BOX_H

#include <simpla/SIMPLA_config.h>
#include "PrimitiveShape.h"
namespace simpla {
namespace geometry {

struct Box : public PrimitiveShape {
    SP_GEO_OBJECT_HEAD(Box, PrimitiveShape)

   protected:
    Box(std::initializer_list<std::initializer_list<Real>> const &v);
    explicit Box(point_type const &p0, point_type const &p1);
    explicit Box(box_type const &v);

   public:
    static std::shared_ptr<Box> New(std::initializer_list<std::initializer_list<Real>> const &box) {
        return std::shared_ptr<Box>(new Box(box));
    }

    point_type xyz(Real u, Real v, Real w) const override;
    point_type uvw(Real x, Real y, Real z) const override;

    point_type GetMinPoint() const;
    point_type GetMaxPoint() const;

    box_type GetBoundingBox() const override { return std::make_tuple(GetMinPoint(), GetMaxPoint()); }
    bool CheckIntersection(box_type const &, Real tolerance) const override;
    //    std::shared_ptr<Point> GetIntersectionion(std::shared_ptr<const Point> const &g, Real tolerance) const
    //    override;
    //    std::shared_ptr<Curve> GetIntersectionion(std::shared_ptr<const Curve> const &g, Real tolerance) const
    //    override;
    //    std::shared_ptr<Surface> GetIntersectionion(std::shared_ptr<const Surface> const &g, Real tolerance) const
    //    override;
    //    std::shared_ptr<Body> GetIntersectionion(std::shared_ptr<const Body> const &g, Real tolerance) const override;

   protected:
    static constexpr Real m_parameter_range_[2][3] = {{0, 0, 0}, {1, 1, 1}};
    static constexpr Real m_value_range_[2][3] = {{0, 0, 0}, {1, 1, 1}};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_BOX_H
