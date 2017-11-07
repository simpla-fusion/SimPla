//
// Created by salmon on 17-11-7.
//

#ifndef SIMPLA_WEDGE_H
#define SIMPLA_WEDGE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Constants.h>
#include "PrimitiveShape.h"
namespace simpla {
namespace geometry {

struct Wedge : public PrimitiveShape {
    SP_GEO_OBJECT_HEAD(Wedge, PrimitiveShape)

   protected:
    explicit Wedge(vector_type const &v, Real ltx);
    explicit Wedge(Axis const &axis, vector_type const &v, Real ltx);

   public:
    point_type xyz(Real u, Real v, Real w) const override;
    point_type uvw(Real x, Real y, Real z) const override;

    vector_type GetExtents() const { return m_extents_; }
    Real GetLTX() const { return m_ltx_; }
    box_type GetBoundingBox() const override;
    //    std::shared_ptr<Point> GetIntersectionion(std::shared_ptr<const Point> const &g, Real tolerance) const
    //    override;
    //    std::shared_ptr<Curve> GetIntersectionion(std::shared_ptr<const Curve> const &g, Real tolerance) const
    //    override;
    //    std::shared_ptr<Surface> GetIntersectionion(std::shared_ptr<const Surface> const &g, Real tolerance) const
    //    override;
    //    std::shared_ptr<Body> GetIntersectionion(std::shared_ptr<const Body> const &g, Real tolerance) const override;

   protected:
    vector_type m_extents_{1, 1, 1};
    Real m_ltx_ = PI / 2;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_WEDGE_H
