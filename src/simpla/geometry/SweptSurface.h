//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_SWEPTSURFACE_H
#define SIMPLA_SWEPTSURFACE_H
#include <simpla/utilities/Constants.h>
#include "Curve.h"
#include "Surface.h"
namespace simpla {
namespace geometry {

struct SweptSurface : public Surface {
    SP_GEO_ABS_OBJECT_HEAD(SweptSurface, Surface);

   protected:
    SweptSurface() = default;
    SweptSurface(SweptSurface const &) = default;
    explicit SweptSurface(std::shared_ptr<Curve> const &c, vector_type const &d)
        : Surface(), m_basis_curve_(c), m_direction_(d) {}

   public:
    ~SweptSurface() override = default;

    std::shared_ptr<Curve> GetBasisCurve() const { return m_basis_curve_; }
    void SetBasisCurve(std::shared_ptr<Curve> const &c) { m_basis_curve_ = c; }
    vector_type GetDirection() const { return m_direction_; }
    void SetDirection(vector_type const &d) { m_direction_ = d; }

   protected:
    std::shared_ptr<Curve> m_basis_curve_;
    vector_type m_direction_{0, 0, 1};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SWEPTSURFACE_H
