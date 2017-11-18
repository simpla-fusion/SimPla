//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_HYPERBOLA_H
#define SIMPLA_HYPERBOLA_H

#include "Shape.h"
namespace simpla {
namespace geometry {

struct spHyperbola : public Shape {
    SP_SHAPE_HEAD(Shape, spHyperbola, Hyperbola);

   protected:
    spHyperbola(Real major_radius, Real minor_radius) : m_major_radius_(major_radius), m_minor_radius_(minor_radius) {}

   public:
    void GetMajorRadius(Real r) { m_major_radius_ = r; }
    void GetMinorRadius(Real r) { m_minor_radius_ = r; }
    Real GetMajorRadius() const { return m_major_radius_; }
    Real GetMinorRadius() const { return m_minor_radius_; }

    point_type xyz(Real alpha) const {
        return m_axis_.xyz(m_major_radius_ * std::cosh(alpha), m_minor_radius_ * std::sinh(alpha));
    };

   protected:
    Real m_major_radius_ = 1;
    Real m_minor_radius_ = 1;
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_HYPERBOLA_H
