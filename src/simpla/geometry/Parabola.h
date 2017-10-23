//
// Created by salmon on 17-10-22.
//

#ifndef SIMPLA_PARABOLA_H
#define SIMPLA_PARABOLA_H
#include "Conic.h"
namespace simpla {
namespace geometry {

struct Parabola : public Conic {
    SP_GEO_OBJECT_HEAD(Parabola, Conic);

   public:
    Parabola() = default;
    Parabola(Parabola const &other) = default;
    ~Parabola() override = default;

    template <typename... Args>
    explicit Parabola(Real focal, Args &&... args) : Conic(std::forward<Args>(args)...), m_focal_(focal) {}

    point_type Value(Real u) const override { return m_origin_ + u * u / (4. * m_focal_) * m_x_axis_ + u * m_y_axis_; };
    bool IsClosed() const override { return false; };
    bool IsPeriodic() const override { return false; };
    Real GetPeriod() const override { return INFINITY; };
    Real GetMinParameter() const override { return -INFINITY; }
    Real GetMaxParameter() const override { return INFINITY; };

    void SetFocal(Real f) { m_focal_ = f; }
    Real GetFocal() const { return m_focal_; }

   protected:
    Real m_focal_ = 1;
};

}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_PARABOLA_H
