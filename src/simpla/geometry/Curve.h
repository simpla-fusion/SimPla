//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_CURVE_H
#define SIMPLA_CURVE_H

#include <simpla/utilities/Constants.h>
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Curve : public GeoObject {
    SP_DEFINE_FANCY_TYPE_NAME(Curve, GeoObject);
    int Dimension() const final { return 1; };
    virtual point_type Value(Real u) const { return point_type{SNaN, SNaN, SNaN}; };
};
struct Conic : public Curve {
    SP_OBJECT_HEAD(Conic, Curve);
};

struct Circle : public Conic {
    SP_OBJECT_HEAD(Circle, Conic);

   protected:
    Circle(point_type x, Real radius, vector_type N, vector_type R)
        : m_origin_(std::move(x)), m_radius_(radius), m_normal_(std::move(N)), m_r_(std::move(R)) {}

   public:
    Real Measure() const final { return TWOPI * m_radius_; };

    point_type Origin() const { return m_origin_; }
    void Origin(point_type const &x) { m_origin_ = x; }

    vector_type Normal() const { return m_normal_; }
    void Normal(vector_type const &v) { m_normal_ = v; }

    vector_type XAxis() const { return m_r_; }
    void XAxis(vector_type const &v) { m_r_ = v; }

    Real Radius() const { return m_radius_; }
    void Radius(Real r) { m_radius_ = r; }

    point_type Value(Real u) const override {
        UNIMPLEMENTED;
        return point_type{SNaN, SNaN, SNaN};
    };

   private:
    point_type m_origin_{0, 0, 0};
    vector_type m_normal_{0, 0, 1};
    vector_type m_r_{1, 0, 0};
    Real m_radius_ = 1;
};

struct Arc : public Conic {
    SP_OBJECT_HEAD(Arc, Conic);

   public:
    Real Measure() const final { return m_radius_ * (m_angle_begin_ - m_angle_end_); };

    point_type Origin() const { return m_origin_; }
    vector_type XAxis() const { return m_XAxis_; }
    vector_type YAxis() const { return m_YAxis_; }
    vector_type ZAxis() const { return cross(m_XAxis_, m_YAxis_); }
    Real Radius() const { return m_radius_; }

    Real AngleStart() const { return m_angle_begin_; }
    Real AngleEnd() const { return m_angle_end_; }

    point_type Value(Real u) const override {
        UNIMPLEMENTED;
        return point_type{SNaN, SNaN, SNaN};
    };

   private:
    point_type m_origin_{0, 0, 0};
    vector_type m_XAxis_{1, 0, 0};
    vector_type m_YAxis_{0, 1, 0};
    Real m_radius_ = 1;
    Real m_angle_begin_ = 0, m_angle_end_ = TWOPI;
};

struct Line : public Curve {
    SP_OBJECT_HEAD(Line, Curve);

   protected:
    Line(point_type p0, point_type p1) : m_p0_(std::move(p0)), m_p1_(std::move(p1)){};

    Line(std::initializer_list<std::initializer_list<Real>> const &v)
        : m_p0_(point_type(*v.begin())), m_p1_(point_type(*(v.begin() + 1))) {}

   public:
    static std::shared_ptr<Line> New(std::initializer_list<std::initializer_list<Real>> const &box) {
        return std::shared_ptr<Line>(new Line(box));
    }
    box_type GetBoundingBox() const override { return std::make_tuple(m_p0_, m_p1_); };
    point_type const &Begin() const { return m_p0_; }
    vector_type const &End() const { return m_p1_; }
    point_type Value(Real r) const override { return m_p0_ + (m_p1_ - m_p0_) * r; }

   protected:
    point_type m_p0_{0, 0, 0}, m_p1_{1, 0, 0};
};

struct AxeLine : public Line {
    SP_OBJECT_HEAD(AxeLine, Line);

   protected:
    AxeLine(int dir, point_type p0, point_type p1) : Line(p0, p1), m_dir_(dir) { m_p1_[dir] = m_p0_[dir]; };

    AxeLine(int dir, std::initializer_list<std::initializer_list<Real>> const &v) : Line(v), m_dir_(dir) {
        m_p1_[(dir + 1) % 3] = m_p0_[(dir + 1) % 3];
        m_p1_[(dir + 2) % 3] = m_p0_[(dir + 2) % 3];
    }

   public:
    static std::shared_ptr<AxeLine> New(int dir, std::initializer_list<std::initializer_list<Real>> const &box) {
        return std::shared_ptr<AxeLine>(new AxeLine(dir, box));
    }
    int GetDirection() const { return m_dir_; }

   private:
    int m_dir_;
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CURVE_H
