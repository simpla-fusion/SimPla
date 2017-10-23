//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_SWEPTSURFACE_H
#define SIMPLA_SWEPTSURFACE_H
#include <simpla/algebra/nTuple.ext.h>
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
    template <typename... Args>
    explicit SweptSurface(std::shared_ptr<Curve> const &c, Args &&... args)
        : Surface(std::forward<Args>(args)...), m_basis_curve_(c) {}

   public:
    ~SweptSurface() override = default;

    std::shared_ptr<Curve> GetBasisCurve() const { return m_basis_curve_; }
    void SetBasisCurve(std::shared_ptr<Curve> const &c) { m_basis_curve_ = c; }

   protected:
    std::shared_ptr<Curve> m_basis_curve_;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SWEPTSURFACE_H
