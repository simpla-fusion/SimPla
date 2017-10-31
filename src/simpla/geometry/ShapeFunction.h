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

    virtual box_type GetParameterRange() const {
        return box_type{{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY}, {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
    }
    virtual box_type GetValueRange() const {
        return box_type{{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY}, {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
    }

    virtual point_type Value(Real u, Real v, Real w) const { return point_type{u, v, w}; }
    virtual point_type InvValue(Real x, Real y, Real z) const { return point_type{x, y, z}; }

    virtual point_type InvValue(point_type const& xyz) const { return point_type{SP_SNaN, SP_SNaN, SP_SNaN}; }
    virtual Real Distance(point_type const& xyz) const { return SP_SNaN; }
    virtual bool TestBoxIntersection(point_type const& x_min, point_type const& x_max) const { return false; }
    virtual int LineIntersection(point_type const& p0, point_type const& p1, Real* u) const { return 0; }
};

#define SP_DEF_SHAPE_FUNCTION_PARA_VALUE_RANGE(_NAME_)                                                          \
    constexpr Real sf##_NAME_::m_parameter_range_[2][3];                                                        \
    constexpr Real sf##_NAME_::m_value_range_[2][3];                                                            \
    box_type sf##_NAME_::GetParameterRange() const { return utility::make_box(m_parameter_range_); }            \
    box_type sf##_NAME_::GetValueRange() const {                                                                \
        return std::make_tuple(utility::make_point(m_value_range_[0]), utility::make_point(m_value_range_[1])); \
    };

#define SP_DEF_PARA_VALUE_RANGE(_NAME_)                                                  \
    box_type _NAME_::GetParameterRange() const { return m_shape_.GetParameterRange(); }; \
    box_type _NAME_::GetValueRange() const {                                             \
        point_type lo, hi;                                                               \
        std::tie(lo, hi) = m_shape_.GetValueRange();                                     \
        return std::make_tuple(m_axis_.xyz(lo), m_axis_.xyz(hi));                        \
    };

}  // namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_SHAPEFUNCTION_H
