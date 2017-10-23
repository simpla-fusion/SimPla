//
// Created by salmon on 17-10-21.
//

#ifndef SIMPLA_LINE_H
#define SIMPLA_LINE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/SPDefines.h>
#include <memory>
#include <utility>
#include "Curve.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Line : public Curve {
    SP_GEO_OBJECT_HEAD(Line, Curve);

   protected:
    Line() = default;
    Line(Line const &) = default;
    template <typename... Args>
    Line(point_type origin, point_type x_axis, Args &&... args)
        : m_origin_(std::move(origin)), m_x_axis_(std::move(x_axis)){};

    Line(std::initializer_list<Real> const &origin, std::initializer_list<Real> const &x_axis)
        : m_origin_(origin), m_x_axis_(x_axis){};

   public:
    ~Line() override = default;

    static std::shared_ptr<Line> New(std::initializer_list<Real> const &origin,
                                     std::initializer_list<Real> const &x_axis) {
        return std::shared_ptr<Line>(new Line(origin, x_axis));
    }
    bool IsClosed() const override { return false; };
    bool IsPeriodic() const override { return false; };
    Real GetPeriod() const override { return SP_INFINITY; };
    Real GetMinParameter() const override { return -SP_INFINITY; }
    Real GetMaxParameter() const override { return SP_INFINITY; }
    point_type Value(Real u) const override { return m_origin_ + u * m_x_axis_; }

    void SetOrigin(point_type const &p) { m_origin_ = p; }
    void SetXAxis(point_type const &p) { m_x_axis_ = p; }
    point_type const &GetOrigin() const { return m_origin_; }
    point_type const &GetXAxis() const { return m_x_axis_; }

   protected:
    point_type m_origin_{0, 0, 0};
    point_type m_x_axis_{1, 0, 0};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_LINE_H
