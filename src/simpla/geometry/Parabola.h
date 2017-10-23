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
    Parabola() = default;
    Parabola(Parabola const &other) = default;
    Parabola(std::shared_ptr<Axis> const &axis, Real focal) : Curve(axis), m_focal_(focal) {}

   public:
    ~Parabola() override = default;

    bool IsClosed() const override { return false; };
    bool IsPeriodic() const override { return false; };
    Real GetPeriod() const override { return SP_INFINITY; };
    Real GetMinParameter() const override { return -SP_INFINITY; }
    Real GetMaxParameter() const override { return SP_INFINITY; };

    void SetFocal(Real f) { m_focal_ = f; }
    Real GetFocal() const { return m_focal_; }

    point_type Value(Real u) const override { return m_axis_->Coordinates(u * u / (4. * m_focal_), u, 0); };

   protected:
    Real m_focal_ = 1;
};

}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_PARABOLA_H
