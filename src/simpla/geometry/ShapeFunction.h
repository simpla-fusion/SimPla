//
// Created by salmon on 17-10-31.
//

#ifndef SIMPLA_SHAPEFUNCTION_H
#define SIMPLA_SHAPEFUNCTION_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/SPDefines.h>
#include <tuple>

namespace simpla {
namespace geometry {

struct ShapeFunction {
    virtual bool IsClosedSurface() const { return false; }
    virtual int GetDimension() const { return 3; }
    virtual box_type const& GetParameterRange() const = 0;
    virtual box_type const& GetValueRange() const = 0;

    virtual point_type Value(Real u, Real v, Real w) const { return point_type{u, v, w}; }

    virtual point_type InvValue(point_type const& xyz) const { return point_type{SP_SNaN, SP_SNaN, SP_SNaN}; }
    virtual Real Distance(point_type const& xyz) const { return SP_SNaN; }
    virtual bool TestBoxIntersection(point_type const& x_min, point_type const& x_max) const { return false; }
    virtual int LineIntersection(point_type const& p0, point_type const& p1, Real* u) const { return 0; }
};
}  // namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_SHAPEFUNCTION_H
