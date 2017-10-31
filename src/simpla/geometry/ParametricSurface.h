//
// Created by salmon on 17-10-31.
//

#ifndef SIMPLA_PARAMETERICSURFACE_H
#define SIMPLA_PARAMETERICSURFACE_H

#include "Surface.h"
namespace simpla {
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

    virtual box_type GetParameterRange() const {
        return box_type{{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY}, {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
    };
    virtual box_type GetValueRange() const {
        return box_type{{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY}, {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
    }
    virtual point_type xyz(Real u, Real v) const = 0;
    virtual point_type uvw(Real x, Real y, Real z) const = 0;

    point_type xyz(point_type const &u) const;
    point_type uvw(point_type const &x) const;
    box_type GetBoundingBox() const override;
    bool IsClosed() const override;
};
}  // namespace geometry{
}  // namespace simpla{impla
#endif  // SIMPLA_PARAMETERICSURFACE_H
