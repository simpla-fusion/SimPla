//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_PARAMETRICSURFACE_H
#define SIMPLA_PARAMETRICSURFACE_H

#include "Surface.h"
namespace simpla {
namespace geometry {
struct ParametricSurface : public Surface {
    SP_GEO_ABS_OBJECT_HEAD(ParametricSurface, Surface);
    ParametricSurface() = default;
    ParametricSurface(ParametricSurface const &) = default;
    ~ParametricSurface() override = default;

    ParametricSurface(point_type origin, point_type x_axis, point_type y_axis, point_type z_axis)
        : m_origin_(std::move(origin)),
          m_x_axis_(std::move(x_axis)),
          m_y_axis_(std::move(y_axis)),
          m_z_axis_(std::move(z_axis)) {}

    void SetUp(point_type const &origin, point_type const &x_axis, point_type const &y_axis, point_type const &z_axis) {
        SetOrigin(origin);
        SetXAxis(x_axis);
        SetYAxis(y_axis);
        SetZAxis(z_axis);
    }

    void SetOrigin(point_type const &p) { m_origin_ = p; }
    void SetXAxis(point_type const &p) { m_x_axis_ = p; }
    void SetYAxis(point_type const &p) { m_y_axis_ = p; }
    void SetZAxis(point_type const &p) { m_z_axis_ = p; }

    point_type const &GetOrigin() const { return m_origin_; }
    point_type const &GetXAxis() const { return m_x_axis_; }
    point_type const &GetYAxis() const { return m_y_axis_; }
    point_type const &GetZAxis() const { return m_z_axis_; }

   protected:
    point_type m_origin_{0, 0, 0};
    point_type m_x_axis_{1, 0, 0};
    point_type m_y_axis_{0, 1, 0};
    point_type m_z_axis_{0, 0, 1};
};

}  // namespace simpla
}  // namespace geometry

#endif  // SIMPLA_PARAMETRICSURFACE_H
