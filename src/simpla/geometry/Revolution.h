//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_REVOLUTIONBODY_H
#define SIMPLA_REVOLUTIONBODY_H
#include <simpla/utilities/Constants.h>
#include "Revolution.h"
#include "Swept.h"
namespace simpla {
namespace geometry {
struct Curve;
struct Surface;
struct Revolution : public Swept {
    SP_GEO_OBJECT_HEAD(Revolution, Swept);

   protected:
    Revolution();
    Revolution(Revolution const &other);
    explicit Revolution(Axis const &axis, std::shared_ptr<const Surface> const &s);

   public:
    ~Revolution() override;
    std::shared_ptr<const Surface> GetBasisSurface() const { return m_basis_surface_; }

   private:
    std::shared_ptr<const Surface> m_basis_surface_;
};

struct RevolutionSurface : public SweptSurface {
    SP_GEO_OBJECT_HEAD(RevolutionSurface, SweptSurface);

   protected:
    RevolutionSurface();
    RevolutionSurface(RevolutionSurface const &other);
    explicit RevolutionSurface(Axis const &axis, std::shared_ptr<Curve> const &c);

   public:
    ~RevolutionSurface() override;
    bool IsClosed() const override;
    std::shared_ptr<const Curve> GetBasisCurve() const { return m_basis_curve_; }

   private:
    std::shared_ptr<const Curve> m_basis_curve_;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_REVOLUTIONBODY_H
