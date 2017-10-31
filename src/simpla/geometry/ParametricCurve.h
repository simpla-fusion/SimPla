//
// Created by salmon on 17-10-31.
//

#ifndef SIMPLA_PARAMETRICCURVE_H
#define SIMPLA_PARAMETRICCURVE_H

#include "Curve.h"
namespace simpla {
template <typename T, int... N>
struct nTuple;
namespace geometry {
struct ParametricCurve;
struct ShapeFunction;
struct ParametricCurve : public Curve {
    SP_GEO_ABS_OBJECT_HEAD(ParametricCurve, Curve);

   protected:
    ParametricCurve();
    ParametricCurve(ParametricCurve const &other);
    explicit ParametricCurve(Axis const &axis);

   public:
    ~ParametricCurve() override;
    bool IsClosed() const override;

    box_type GetBoundingBox() const override;
    std::shared_ptr<PolyPoints> GetBoundaryPoints() const override;
    std::shared_ptr<PolyPoints> Intersection(std::shared_ptr<const Curve> const &g, Real tolerance) const override;
    std::shared_ptr<PolyPoints> Intersection(std::shared_ptr<const Plane> const &g, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const Box> const &g, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const override;

    virtual point_type xyz(Real u) const;
    point_type xyz(point_type const &u) const;
    point_type uvw(point_type const &x) const;
};
}  // namespace geometry{
}  // namespace simpla{impla
#endif  // SIMPLA_PARAMETRICCURVE_H
