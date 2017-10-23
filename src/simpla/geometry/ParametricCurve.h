//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_PARAMETRICSURFACE_H
#define SIMPLA_PARAMETRICSURFACE_H

#include "Curve.h"
namespace simpla {
namespace geometry {

struct ParametricCurve : public Curve {
    SP_GEO_ABS_OBJECT_HEAD(ParametricCurve, Curve);

   public:
    ParametricCurve(ParametricCurve const &other) = default;
    ParametricCurve() = default;
    ~ParametricCurve() override = default;

    ParametricCurve(point_type origin, point_type x_axis, point_type y_axis)
        : m_origin_(std::move(origin)), m_x_axis_(std::move(x_axis)), m_y_axis_(std::move(y_axis)) {}

    void SetUp(point_type const &origin, point_type const &x_axis, point_type const &y_axis) {
        SetOrigin(origin);
        SetXAxis(x_axis);
        SetYAxis(y_axis);
    }

    void SetOrigin(point_type const &p) { m_origin_ = p; }
    void SetXAxis(point_type const &p) { m_x_axis_ = p; }
    void SetYAxis(point_type const &p) { m_y_axis_ = p; }
    point_type const &GetOrigin() const { return m_origin_; }
    point_type const &GetXAxis() const { return m_x_axis_; }
    point_type const &GetYAxis() const { return m_y_axis_; }

   protected:
    point_type m_origin_{0, 0, 0};
    point_type m_x_axis_{1, 0, 0};
    point_type m_y_axis_{1, 0, 0};
};

}  // namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_PARAMETRICSURFACE_H
