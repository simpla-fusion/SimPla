//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_HYPERBOLA_H
#define SIMPLA_HYPERBOLA_H
#include "ParametricCurve.h"
namespace simpla {
namespace geometry {

struct Hyperbola : public ParametricCurve {
    SP_GEO_OBJECT_HEAD(Hyperbola, ParametricCurve);

   protected:
    Hyperbola() = default;
    Hyperbola(Hyperbola const &other) = default;
    Hyperbola(Axis const &axis, Real major_radius, Real minor_radius, Real alpha0 = SP_SNaN, Real alpha1 = SP_SNaN)
        : ParametricCurve(axis), m_major_radius_(major_radius), m_minor_radius_(minor_radius) {}

   public:
    ~Hyperbola() override = default;

    void GetMajorRadius(Real r) { m_major_radius_ = r; }
    void GetMinorRadius(Real r) { m_minor_radius_ = r; }
    Real GetMajorRadius() const { return m_major_radius_; }
    Real GetMinorRadius() const { return m_minor_radius_; }

    point_type xyz(Real alpha) const override {
        return m_axis_.xyz(m_major_radius_ * std::cosh(alpha), m_minor_radius_ * std::sinh(alpha));
    };
    bool TestIntersection(box_type const &, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   protected:
    Real m_major_radius_ = 1;
    Real m_minor_radius_ = 1;
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_HYPERBOLA_H
