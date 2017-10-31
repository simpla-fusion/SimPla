//
// Created by salmon on 17-10-31.
//

#ifndef SIMPLA_PARAMETERICSURFACE_H
#define SIMPLA_PARAMETERICSURFACE_H

#include "Surface.h"
namespace simpla {
template <typename T, int... N>
struct nTuple;
namespace geometry {
struct ParametricCurve;
struct ShapeFunction;
struct ParametricSurface : public Surface {
    SP_GEO_ABS_OBJECT_HEAD(ParametricSurface, Surface);

   protected:
    ParametricSurface();
    ParametricSurface(ParametricSurface const &other);
    explicit ParametricSurface(Axis const &axis);

   public:
    ~ParametricSurface() override;
    bool IsClosed() const override;
    box_type GetBoundingBox() const override;
    std::shared_ptr<Curve> GetBoundaryCurve() const override;
    std::shared_ptr<PolyPoints> Intersection(std::shared_ptr<const Curve> const &g, Real tolerance) const override;
    std::shared_ptr<Curve> Intersection(std::shared_ptr<const Surface> const &g, Real tolerance) const override;
    bool TestIntersection(point_type const &, Real tolerance) const override override;
    bool TestIntersection(box_type const &, Real tolerance) const override override;

    box_type const &GetParameterRange() const;
    box_type const &GetValueRange() const;

    virtual ShapeFunction const &shape() const = 0;

    virtual point_type xyz(Real u, Real v) const;
    point_type xyz(point_type const &u) const;
    point_type xyz(point2d_type const &u) const;
    point_type uvw(point_type const &x) const;

   protected:
    std::shared_ptr<ParametricCurve> m_curve_ = nullptr;
};
}  // namespace geometry{
}  // namespace simpla{impla
#endif  // SIMPLA_PARAMETERICSURFACE_H
