//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_HYPERBOLA_H
#define SIMPLA_HYPERBOLA_H
#include "Curve.h"
namespace simpla {
namespace geometry {

struct Hyperbola : public Curve {
    SP_GEO_OBJECT_HEAD(Hyperbola, Curve);

   protected:
    Hyperbola() = default;
    Hyperbola(Hyperbola const &other) = default;
    Hyperbola( Axis  const &axis, Real major_radius, Real minor_radius, Real alpha0 = SP_SNaN,
              Real alpha1 = SP_SNaN)
        : Curve(axis), m_major_radius_(major_radius), m_minor_radius_(minor_radius) {
        SetParameterRange(std::isnan(alpha0) ? GetMinParameter() : alpha0,
                          std::isnan(alpha1) ? GetMaxParameter() : alpha1);
    }

   public:
    ~Hyperbola() override = default;

    bool IsClosed() const override { return false; };
    bool IsPeriodic() const override { return false; };
    Real GetPeriod() const override { return SP_INFINITY; };
    Real GetMinParameter() const override { return -SP_INFINITY; }
    Real GetMaxParameter() const override { return SP_INFINITY; };

    void GetMajorRadius(Real r) { m_major_radius_ = r; }
    void GetMinorRadius(Real r) { m_minor_radius_ = r; }
    Real GetMajorRadius() const { return m_major_radius_; }
    Real GetMinorRadius() const { return m_minor_radius_; }

    point_type Value(Real alpha) const override {
        return m_axis_.Coordinates(m_major_radius_ * std::cosh(alpha), m_minor_radius_ * std::sinh(alpha));
    };
    int CheckOverlap(box_type const &) const override;                                                                \
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override; \
   protected:
    Real m_major_radius_ = 1;
    Real m_minor_radius_ = 1;
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_HYPERBOLA_H
