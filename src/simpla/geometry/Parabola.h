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

   public:
    Parabola() = default;
    Parabola(Parabola const &other) = default;
    ~Parabola() override = default;

    template <typename... Args>
    explicit Parabola(Real focal, Args &&... args) : ParametricCurve(std::forward<Args>(args)...), m_focal_(focal) {}

    bool IsClosed() const override { return false; };
    bool IsPeriodic() const override { return false; };
    Real GetPeriod() const override { return SP_INFINITY; };
    Real GetMinParameter() const override { return -SP_INFINITY; }
    Real GetMaxParameter() const override { return SP_INFINITY; };

    void SetFocal(Real f) { m_focal_ = f; }
    Real GetFocal() const { return m_focal_; }

    point_type Value(Real u) const override { return m_origin_ + u * u / (4. * m_focal_) * m_x_axis_ + u * m_y_axis_; };

   protected:
    Real m_focal_ = 1;
};

}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_PARABOLA_H
