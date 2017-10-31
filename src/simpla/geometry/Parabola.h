//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_PARABOLA_H
#define SIMPLA_PARABOLA_H
#include "ParametricCurve.h"
namespace simpla {
namespace geometry {

struct Parabola : public ParametricCurve {
    SP_GEO_OBJECT_HEAD(Parabola, ParametricCurve);

   protected:
    Parabola() = default;
    Parabola(Parabola const &other) = default;
    Parabola(Axis const &axis, Real focal, Real alpha0 = SP_SNaN, Real alpha1 = SP_SNaN)
        : ParametricCurve(axis), m_focal_(focal) {}

   public:
    ~Parabola() override = default;

    bool IsClosed() const override { return false; };

    void SetFocal(Real f) { m_focal_ = f; }
    Real GetFocal() const { return m_focal_; }

    point_type xyz(Real u) const override { return m_axis_.Coordinates(u * u / (4. * m_focal_), u, 0); };
    bool TestIntersection(box_type const &, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   protected:
    Real m_focal_ = 1;
};

}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_PARABOLA_H
