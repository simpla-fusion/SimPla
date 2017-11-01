//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_ELLIPSE_H
#define SIMPLA_ELLIPSE_H
#include "ParametricCurve.h"
namespace simpla {
namespace geometry {
struct Ellipse : public ParametricCurve {
    SP_GEO_OBJECT_HEAD(Ellipse, ParametricCurve);

   protected:
    Ellipse();
    Ellipse(Ellipse const &other);
    Ellipse(Axis const &axis, Real major_radius, Real minor_radius);

   public:
    ~Ellipse() override;

    bool IsClosed() const override { return true; };

    void GetMajorRadius(Real r) { m_major_radius_ = r; }
    void GetMinorRadius(Real r) { m_minor_radius_ = r; }
    Real GetMajorRadius() const { return m_major_radius_; }
    Real GetMinorRadius() const { return m_minor_radius_; }

    point_type xyz(Real alpha) const override {
        return m_axis_.Coordinates(m_major_radius_ * std::cos(alpha), m_minor_radius_ * std::sin(alpha));
    };
    //    bool CheckIntersection(point_type const &, Real tolerance) override;
    //    bool CheckIntersection(box_type const &, Real tolerance) override;

   protected:
    Real m_major_radius_ = 1;
    Real m_minor_radius_ = 1;
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_ELLIPSE_H
