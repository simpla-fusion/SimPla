//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_PARABOLA_H
#define SIMPLA_PARABOLA_H
#include "Curve.h"
namespace simpla {
namespace geometry {

struct Parabola : public Curve {
    SP_GEO_OBJECT_HEAD(Parabola, Curve);

   protected:
    Parabola(Axis const &axis, Real focal, Real alpha0 = SP_SNaN, Real alpha1 = SP_SNaN)
        : Curve(axis), m_focal_(focal) {}

   public:
    bool IsClosed() const override { return false; };

    void SetFocal(Real f) { m_focal_ = f; }
    Real GetFocal() const { return m_focal_; }

    point_type xyz(Real u) const override { return m_axis_.Coordinates(u * u / (4. * m_focal_), u, 0); };

   protected:
    Real m_focal_ = 1;
};

}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_PARABOLA_H
