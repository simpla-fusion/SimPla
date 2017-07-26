//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_CURVE_H
#define SIMPLA_CURVE_H

#include <simpla/physics/Constants.h>
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Curve : public GeoObject {
    int Dimension() const final { return 1; };
};
struct Conic : public Curve {};

struct Circle : public Conic {
    Circle() = default;
    ~Circle() override = default;

    Circle(point_type x, Real radius, vector_type N, vector_type R)
        : m_origin_(std::move(x)), m_radius_(radius), m_normal_(std::move(N)), m_r_(std::move(R)) {}

    Real Measure() const final { return TWOPI * m_radius_; };

    point_type Origin() const { return m_origin_; }
    void Origin(point_type const &x) { m_origin_ = x; }

    vector_type Normal() const { return m_normal_; }
    void Normal(vector_type const &v) { m_normal_ = v; }

    vector_type XAxis() const { return m_r_; }
    void XAxis(vector_type const &v) { m_r_ = v; }

    Real Radius() const { return m_radius_; }
    void Radius(Real r) { m_radius_ = r; }

   private:
    point_type m_origin_{0, 0, 0};
    vector_type m_normal_{1, 0, 0};
    vector_type m_r_{0, 1, 0};
    Real m_radius_ = 1;
};
struct Arc : public Conic {
    Arc() = default;
    ~Arc() override = default;

    Real Measure() const final { return TWOPI * m_radius_; };

    point_type Origin() const { return m_origin_; }
    void Origin(point_type const &x) { m_origin_ = x; }

    vector_type XAxis() const { return m_XAxis_; }
    void XAxis(vector_type const &v) { m_XAxis_ = v; }

    vector_type YAxis() const { return m_YAxis_; }
    void YAxis(vector_type const &v) { m_YAxis_ = v; }

    vector_type ZAxis() const { return cross(m_XAxis_, m_YAxis_); }

    Real Radius() const { return m_radius_; }
    void Radius(Real const &r) { m_radius_ = r; }

    void AngleStart(Real a0) { m_angle_start_ = a0; }
    void AngleEnd(Real a1) { m_angle_start_ = a1; }
    Real AngleStart() const { return m_angle_start_; }
    Real AngleEnd() const { return m_angle_end_; }

   private:
    point_type m_origin_{0, 0, 0};
    vector_type m_XAxis_{1, 0, 0};
    vector_type m_YAxis_{0, 1, 0};
    Real m_radius_ = 1;
    Real m_angle_start_ = 0, m_angle_end_ = TWOPI;
};

struct Line : public Curve {
    Line(point_type p0, point_type p1) : m_p0_(std::move(p0)), m_p1_(std::move(p1)){};
    Line() = default;
    ~Line() override = default;

    Real Measure() const final { return std::sqrt(dot(m_p1_ - m_p0_, m_p1_ - m_p0_)); };

    void Start(point_type const &p0) { m_p0_ = p0; }
    point_type const &Start() const { return m_p0_; }
    void End(point_type const &p1) { m_p1_ = p1; }
    point_type const &End() const { return m_p1_; }

   private:
    point_type m_p0_{0, 0, 0}, m_p1_{1, 0, 0};
};

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CURVE_H
