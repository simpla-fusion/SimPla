//
// Created by salmon on 17-10-20.
//

#ifndef SIMPLA_PLANE_H
#define SIMPLA_PLANE_H

#include <simpla/SIMPLA_config.h>
#include "GeoObject.h"
#include "ParametricSurface.h"

namespace simpla {
namespace geometry {
struct Plane : public Surface {
    SP_GEO_OBJECT_HEAD(Plane, Surface);
    Plane() = default;
    Plane(Plane const &) = default;
    ~Plane() override = default;

    Plane(point_type origin, point_type x_axis, point_type y_axis)
        : m_origin_(std::move(origin)), m_x_axis_(std::move(x_axis)), m_y_axis_(std::move(y_axis)) {}

    std::tuple<bool, bool> IsClosed() const override { return std::make_tuple(false, false); };
    std::tuple<bool, bool> IsPeriodic() const override { return std::make_tuple(false, false); };
    nTuple<Real, 2> GetPeriod() const override { return nTuple<Real, 2>{SP_INFINITY, SP_INFINITY}; };
    nTuple<Real, 2> GetMinParameter() const override { return nTuple<Real, 2>{-SP_INFINITY, -SP_INFINITY}; }
    nTuple<Real, 2> GetMaxParameter() const override { return nTuple<Real, 2>{SP_INFINITY, SP_INFINITY}; }

    void SetOrigin(point_type const &p) { m_origin_ = p; }
    void SetXAxis(point_type const &p) { m_x_axis_ = p; }
    void SetYAxis(point_type const &p) { m_y_axis_ = p; }

    point_type const &GetOrigin() const { return m_origin_; }
    point_type const &GetXAxis() const { return m_x_axis_; }
    point_type const &GetYAxis() const { return m_y_axis_; }
    point_type GetZAxis() const { return cross(m_x_axis_, m_y_axis_); }

    point_type Value(Real u, Real v) const override { return u * m_x_axis_ + v * m_y_axis_; };

   protected:
    point_type m_origin_{0, 0, 0};
    point_type m_x_axis_{1, 0, 0};
    point_type m_y_axis_{0, 1, 0};
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_PLANE_H
