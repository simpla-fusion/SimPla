//
// Created by salmon on 17-10-31.
//

#ifndef SIMPLA_PARAMETRICBODY_H
#define SIMPLA_PARAMETRICBODY_H
#include "Body.h"
namespace simpla {
namespace geometry {
struct ParametricBody : public Body {
    SP_GEO_ABS_OBJECT_HEAD(ParametricBody, Body);

   protected:
    ParametricBody();
    ParametricBody(ParametricBody const &other);
    explicit ParametricBody(Axis const &axis);

   public:
    ~ParametricBody() override;

    virtual box_type GetParameterRange() const = 0;
    virtual box_type GetValueRange() const = 0;
    virtual point_type xyz(Real u, Real v, Real w) const = 0;
    virtual point_type uvw(Real x, Real y, Real z) const = 0;

    point_type xyz(point_type const &u) const;
    point_type uvw(point_type const &x) const;
    box_type GetBoundingBox() const override;
};
}  // namespace geometry{
}  // namespace simpla{impla
#endif  // SIMPLA_PARAMETRICBODY_H
