//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_HYPERBOLA_H
#define SIMPLA_HYPERBOLA_H
#include "Conic.h"
namespace simpla {
namespace geometry {

struct Hyperbola : public Conic {
    SP_GEO_OBJECT_HEAD(Hyperbola, Conic);

   public:
    Hyperbola() = default;
    Hyperbola(Hyperbola const &other) = default;
    ~Hyperbola() override = default;

    template <typename... Args>
    Hyperbola(Real major_radius, Real minor_radius, Args &&... args)
        : Conic(std::forward<Args>(args)...), m_major_radius_(major_radius), m_minor_radius_(minor_radius) {}

    point_type Value(Real alpha) const override {
        return m_origin_ + m_major_radius_ * std::cosh(alpha) * m_x_axis_ +
               m_minor_radius_ * std::sinh(alpha) * m_y_axis_;
    };
    bool IsClosed() const override { return false; };
    bool IsPeriodic() const override { return false; };
    Real GetPeriod() const override { return INIFITY; };
    Real GetMinParameter() const override { return -INIFITY; }
    Real GetMaxParameter() const override { return INIFITY; };

    void GetMajorRadius(Real r) { m_major_radius_ = r; }
    void GetMinorRadius(Real r) { m_minor_radius_ = r; }
    Real GetMajorRadius() const { return m_major_radius_; }
    Real GetMinorRadius() const { return m_minor_radius_; }

   protected:
    Real m_major_radius_ = 1;
    Real m_minor_radius_ = 1;
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_HYPERBOLA_H
