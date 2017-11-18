//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_PARABOLA_H
#define SIMPLA_PARABOLA_H
#include "Curve.h"
#include "Shape.h"
namespace simpla {
namespace geometry {

struct spParabola : public Shape {
    SP_SHAPE_HEAD(Shape, spParabola, Parabola);

   protected:
    explicit spParabola(Real focal) : m_focal_(focal) {}

   public:
    void SetFocal(Real f) { m_focal_ = f; }
    Real GetFocal() const { return m_focal_; }

    //    point2d_type xy(Real u) const override { return point2d_type{u * u / (4. * m_focal_), u}; };
    //    Real u(point2d_type const &xy) const override { return xy[1]; };

   protected:
    Real m_focal_ = 1;
};

}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_PARABOLA_H
