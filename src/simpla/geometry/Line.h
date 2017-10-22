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

   public:
    Line() = default;
    Line(Line const &) = default;
    ~Line() = default;
    Line(point_type const &p0, point_type const &p1) : m_p{{p0[0], p0[1], p0[2]}, {p1[0], p1[1], p1[2]}} {};

    Line(std::initializer_list<std::initializer_list<Real>> const &v) {
        SetVertices(point_type(*v.begin()), point_type(*(v.begin() + 1)));
    }

    static std::shared_ptr<Line> New(std::initializer_list<std::initializer_list<Real>> const &box) {
        return std::shared_ptr<Line>(new Line(box));
    }

    point_type Value(Real u) const override { return m_p[0] * (1 - u) + u * m_p[1]; }

    void SetVertices(point_type const &v0, point_type const &v1) {
        m_p[0] = v0;
        m_p[1] = v1;
    }
    nTuple<Real, 2, 3> const &GetVertices() const { return m_p; };

    point_type GetPoint(Real r) const { return m_p[0] + (m_p[1] - m_p[0]) * r; }
    std::tuple<Real, Real> GetParameter(point_type const &p) const {
        return std::tuple<Real, Real>{dot(p - m_p[0], m_p[1] - m_p[0]), dot(p - m_p[0], m_p[2] - m_p[0])};
    };

   protected:
    nTuple<Real, 2, 3> m_p{{0, 0, 0}, {1, 0, 0}};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_LINE_H
