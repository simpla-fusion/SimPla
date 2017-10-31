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
    explicit Revolution(std::shared_ptr<const Surface> const &s, point_type const &origin, vector_type const &axe_z);

   public:
    ~Revolution() override;

    bool TestIntersection(box_type const &, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;
};

struct RevolutionSurface : public SweptSurface {
    SP_GEO_OBJECT_HEAD(RevolutionSurface, SweptSurface);

   protected:
    RevolutionSurface();
    RevolutionSurface(RevolutionSurface const &other);
    RevolutionSurface(Axis const &axis, std::shared_ptr<Curve> const &c);

   public:
    ~RevolutionSurface() override;
    std::shared_ptr<const Curve> GetBasisCurve() const { return m_basis_curve_; }
    bool IsClosed() const override;

    bool TestIntersection(box_type const &, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   private:
    std::shared_ptr<const Curve> m_basis_curve_;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_REVOLUTIONBODY_H
